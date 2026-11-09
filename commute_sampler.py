#!/usr/bin/env python3
"""Live "leave now" commute sampler using commute_settings.json.

Setup (once):
    pip install -r requirements.txt
    playwright install chromium

Run (leave it running; macOS: `caffeinate -i python3 commute_sampler.py`):
    python3 commute_sampler.py            # samples every 10 min, around the clock, until stopped
    python3 commute_sampler.py --once     # one sample now, then exit

Google Maps "leave now" directions are loaded with your exact intermediate turns as waypoints, so the
times are for your route, not Google's preferred one. Each sample appends one row per direction to
route-specific CSV files (local time, direction, minutes, miles, route label, traffic note).
Addresses, waypoints and windows come from the app's Settings window (commute_settings.json).
"""
import argparse, asyncio, csv, fcntl, os, re, signal, statistics, sys, time
from contextlib import contextmanager
from datetime import datetime, timedelta
try:
    import commute_model
except Exception as exc:  # sampling is the lifeline: a problem in the learning code must never stop it
    commute_model = None
    print(f"model code unavailable, training disabled: {exc}", file=sys.stderr)
from commute_config import (CSV, HERE, TZ, SettingsError, gmaps_url, load_settings, route_points,
                            sample_path, window)

LOCK = os.path.join(HERE, ".commute-sampler.lock")
REFRESH = os.path.join(HERE, ".commute-refresh")
INTERVAL_MIN = 10
RETRY_SECONDS = 60      # after a failed sample, retry soon (doubling up to MAX_RETRY_SECONDS) instead of waiting for the next slot
MAX_RETRY_SECONDS = 300
SAMPLE_TIMEOUT = 240    # a wedged Chromium/Maps load must never stall the loop
CLOSE_TIMEOUT = 15
WAKE_GAP_SECONDS = 60   # the loop ticks every 5 s; a bigger wall-clock gap means the Mac slept
BROWSER_MAX_AGE = 3 * 3600  # recycle a long-lived Chromium (its in-memory cache and heap only grow)
VIEWPORT = {"width": 1400, "height": 1000}
DIRECTIONS = ("to_work", "to_home")
# A scraped card outside these bounds means a changed Maps page or the wrong route, not traffic.
MIN_MINUTES, MAX_MINUTES = 1, 180
MIN_MILES, MAX_MILES = 0.3, 60
MILES_BAND = (0.5, 1.6)  # accepted ratio to this route's recent median distance
ACCEPT_AFTER = 6         # consecutive cards that agree on a new distance (a real detour) become the baseline
MODEL_SYNC = commute_model is not None   # train the Predicted model after each finished day (tests switch this off)
MODEL_CHECK_SECONDS = 1800   # how often to ask whether a training run is due (also the retry delay after a failure)
MODEL_SYNC_TIMEOUT = 1800
MODEL_SCRIPT = os.path.join(HERE, "commute_model.py")
# (weekday-only) commute windows, local time, come from the settings file. Sampling runs around the clock so
# the latest value is never stale, but charts and recommendations only use samples inside these windows. Both
# directions are sampled so you also see what the reverse trip looks like; filter on `direction` when reading.
WEEKDAYS_ONLY = True


JS = """
() => {
  const root = document.querySelector('[aria-label="Directions"]') || document;
  const out = [];
  for (const el of root.querySelectorAll('div[tabindex="0"]')) {
    if (!el.querySelector('h1')) continue;
    out.push(el.innerText.replace(/\\n/g, ' | '));
  }
  return out;
}
"""


async def accept_consent(page):
    for sel in ['button[aria-label="Accept all"]', 'form[action*="consent"] button', 'button:has-text("Accept all")']:
        try:
            b = page.locator(sel).first
            if await b.is_visible(timeout=1000):
                await b.click()
                await page.wait_for_timeout(1500)
                return
        except Exception:
            pass


async def fetch_routes(page, points):
    await page.goto(gmaps_url(points), wait_until="domcontentloaded", timeout=60000)
    await accept_consent(page)
    for _ in range(60):
        routes = await page.evaluate(JS)
        if routes and any(parse(r) for r in routes):
            return routes
        await page.wait_for_timeout(500)
    return routes or []


DURATION = re.compile(r"(?:(\d+)\s*hrs?(?:\s+(\d+)\s*min)?|(\d+)\s*min)")
DISTANCE = re.compile(r"(\d[\d,]*(?:\.\d+)?)\s*(miles?|mi|km)")


def parse(route):
    # e.g. "29 min | 3.4 miles | via Route name | Fastest route | Details". Fields must be whole
    # segments so banner text like "5 min slower than usual" is never mistaken for the duration.
    segments = [segment.strip() for segment in route.split("|")]
    minutes = None
    for segment in segments:
        m = DURATION.fullmatch(segment)
        if m:
            minutes = 60 * int(m.group(1) or 0) + int(m.group(2) or m.group(3) or 0)
            break
    if minutes is None:
        return None
    miles = ""
    for segment in segments:
        m = DISTANCE.fullmatch(segment)
        if m:
            miles = float(m.group(1).replace(",", "")) * (0.621371 if m.group(2) == "km" else 1)
            miles = round(miles, 1)
            break
    via = re.search(r"via ([^|]+)", route)
    note = re.search(r"\| ((?:Fastest|Slower|Similar|Best|This route)[^|]*)", route)
    return minutes, miles, via.group(1).strip() if via else "", note.group(1).strip() if note else ""


def recent_miles(path, direction, tail_bytes=65536):
    """Median distance of this route's latest rows, or None until there are enough to trust."""
    try:
        with open(path, "rb") as handle:
            handle.seek(0, os.SEEK_END)
            handle.seek(max(0, handle.tell() - tail_bytes))
            text = handle.read().decode("utf-8", "ignore")
    except OSError:
        return None
    values = []
    for row in csv.reader(text.splitlines()[1:]):  # the first line is a partial row or the header
        try:
            if len(row) >= 6 and row[3] == direction and row[5]:
                values.append(float(row[5]))
        except ValueError:
            pass
    return statistics.median(values) if len(values) >= 10 else None


def check_card(top, typical_miles=None):
    """Why a parsed card cannot be trusted, or None."""
    minutes, miles = top[0], top[1]
    if not MIN_MINUTES <= minutes <= MAX_MINUTES:
        return f"{minutes} min is outside {MIN_MINUTES}-{MAX_MINUTES}"
    if miles != "":
        if not MIN_MILES <= miles <= MAX_MILES:
            return f"{miles} miles is outside {MIN_MILES}-{MAX_MILES}"
        if typical_miles and not MILES_BAND[0] * typical_miles <= miles <= MILES_BAND[1] * typical_miles:
            return f"{miles} miles vs. a typical {typical_miles:g} for this route"
    return None


class SampleError(RuntimeError):
    """A sample where some directions failed; `succeeded` rows were already saved."""

    def __init__(self, message, succeeded=()):
        super().__init__(message)
        self.succeeded = set(succeeded)


_band_rejects = {}  # direction -> distances of consecutive cards rejected only for the distance band


def torn_tail(csv_path):
    """True when the file's last row has no line ending (a write cut short); appending would fuse the next row onto it."""
    try:
        with open(csv_path, "rb") as handle:
            handle.seek(-1, os.SEEK_END)
            return handle.read(1) not in (b"\n", b"\r")
    except OSError:  # missing or empty
        return False


def append_rows(csv_path, rows):
    new = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0
    torn = not new and torn_tail(csv_path)
    with open(csv_path, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["date", "time", "dow", "direction", "minutes", "miles", "via", "note"])
        elif torn:
            f.write(w.dialect.lineterminator)  # the fragment stays one unreadable row; the new rows start clean
        w.writerows(rows)
    for r in rows:
        print(f"{r[0]} {r[1]} {r[3]:8s} {r[4]:>3} min  {r[5]} mi  via {r[6]}  ({r[7]})", flush=True)


def vet_card(direction, top, baseline):
    """Why a card is rejected, or None. A distance that keeps agreeing with itself eventually wins."""
    problem = check_card(top, baseline)
    if problem and baseline and check_card(top, None) is None:  # only the distance band failed
        streak = _band_rejects.setdefault(direction, [])
        streak.append(top[1])
        del streak[:-ACCEPT_AFTER]
        if len(streak) == ACCEPT_AFTER and max(streak) <= 1.15 * min(streak):
            print(f"{direction}: {ACCEPT_AFTER} cards in a row say {top[1]} miles (typical {baseline:g}); "
                  "accepting the new route distance", file=sys.stderr)
            streak.clear()
            return None
    else:
        _band_rejects.pop(direction, None)
    return problem


async def sample_once(browser, settings=None, directions=DIRECTIONS, context=None, done=None):
    """Sample `directions`, appending each success's row as soon as it validates; raise SampleError if
    any fail. `done` (a set) collects the directions already saved, even if the sample is cancelled.

    With a `context`, pages share its cookies and HTTP cache (Maps is ~3 MB cold, ~0.2 MB warm).
    """
    settings = settings if settings is not None else load_settings()
    csv_path = sample_path(settings, CSV)
    done = done if done is not None else set()
    page = await (context.new_page() if context is not None else browser.new_page(viewport=VIEWPORT))
    failures = []
    now = datetime.now(TZ)
    try:
        for direction in directions:
            pts = route_points(settings, direction)
            try:
                routes = await fetch_routes(page, pts)
            except Exception as exc:
                failures.append(f"{direction}: {exc}")
                continue
            top = next((parse(r) for r in routes if parse(r)), None)
            if not top:
                failures.append(f"{direction}: no route parsed")
                print(f"{now:%H:%M} {direction}: no route parsed ({routes[:1]})", file=sys.stderr)
                continue
            problem = vet_card(direction, top, recent_miles(csv_path, direction))
            if problem:
                failures.append(f"{direction}: rejected card ({problem})")
                print(f"{now:%H:%M} {direction}: rejected card, {problem}: {routes[:1]}", file=sys.stderr)
                continue
            if top[1] == "":
                print(f"{now:%H:%M} {direction}: card has no distance: {routes[:1]}", file=sys.stderr)
            append_rows(csv_path, [[now.strftime("%Y-%m-%d"), now.strftime("%H:%M"), now.strftime("%a"), direction, *top]])
            done.add(direction)
    finally:
        try:
            await asyncio.wait_for(page.close(), CLOSE_TIMEOUT)  # a wedged renderer must not outlive SAMPLE_TIMEOUT
        except Exception:
            pass
    if failures:
        raise SampleError("; ".join(failures), succeeded=done)


def in_window(now, settings=None):
    if WEEKDAYS_ONLY and now.weekday() >= 5:
        return False
    t = (now.hour, now.minute)
    settings = settings if settings is not None else load_settings()
    return any(a <= t <= b for a, b in (window(settings, "to_work"), window(settings, "to_home")))


def sample_slot(now=None):
    return int((time.time() if now is None else now) // (INTERVAL_MIN * 60))


def next_sample_at(now, failures):
    """Next 10-minute slot after a good sample; a short, growing retry delay after consecutive failures."""
    if not failures:
        step = INTERVAL_MIN * 60
        return now + step - (now % step)
    return now + min(RETRY_SECONDS * 2 ** (failures - 1), MAX_RETRY_SECONDS)


async def close_browser(browser):
    try:
        await asyncio.wait_for(browser.close(), CLOSE_TIMEOUT)
    except Exception:
        pass  # already dead or wedged; the process is released when the sampler exits


async def launch(p):
    """A fresh headless Chromium and the shared context its samples reuse (None if unsupported)."""
    browser = await p.chromium.launch(headless=True)
    new_context = getattr(browser, "new_context", None)
    try:
        return browser, (await new_context(viewport=VIEWPORT) if new_context else None)
    except BaseException:
        await close_browser(browser)
        raise


async def run_model_sync():
    """The nightly training job as a child process, so a long rebuild never stalls sampling."""
    proc = await asyncio.create_subprocess_exec(sys.executable, MODEL_SCRIPT, "--sync")
    try:
        return await asyncio.wait_for(proc.wait(), MODEL_SYNC_TIMEOUT)
    except BaseException:
        if proc.returncode is None:
            proc.kill()
        raise


async def run(once):
    from playwright.async_api import async_playwright
    loop = asyncio.get_running_loop()
    task = asyncio.current_task()
    loop.add_signal_handler(signal.SIGTERM, task.cancel)
    async with async_playwright() as p:
        browser = context = None
        browser_since = 0
        next_sample = 0
        last_tick = time.time()
        failures = 0
        pending = set(DIRECTIONS)  # directions still owed for this slot; a retry skips ones that succeeded
        pending_slot = None
        model_task = None
        next_model_check = 0
        model_identity = None
        settings_error = None  # the settings error already printed; the 5 s loop must not repeat it
        try:
            while True:
                now = datetime.now(TZ)
                woke = time.time() - last_tick > WAKE_GAP_SECONDS
                if woke:
                    # The Mac slept (the loop ticks every 5 s): the data is stale and the network may be changing. Sample
                    # again right away, from a clean slate, whatever was scheduled or backing off before.
                    next_sample, failures, pending = 0, 0, set(DIRECTIONS)
                if browser is not None and (woke or time.time() - browser_since > BROWSER_MAX_AGE):
                    await close_browser(browser)  # Chromium does not survive sleep reliably
                    browser = context = None
                try:
                    settings = load_settings()
                    settings_error = None
                except SettingsError as exc:
                    if str(exc) != settings_error:
                        settings_error = str(exc)
                        print(f"{now:%H:%M} settings error: {exc}", file=sys.stderr, flush=True)
                    if once:
                        return 1
                    if browser is not None:
                        await close_browser(browser)
                        browser = context = None
                    last_tick = time.time()
                    await asyncio.sleep(5)
                    continue
                refresh = consume_refresh()
                if once or refresh or time.time() >= next_sample:
                    # Retries within a slot skip directions already saved; a new slot (or a refresh) samples
                    # everything, so one direction failing for hours never starves the other.
                    if once or refresh or failures == 0 or sample_slot() != pending_slot:
                        pending = set(DIRECTIONS)
                    pending_slot = sample_slot()
                    done = set()
                    ok = True
                    try:
                        if browser is None:
                            browser, context = await asyncio.wait_for(launch(p), SAMPLE_TIMEOUT)
                            browser_since = time.time()
                        await asyncio.wait_for(sample_once(browser, settings, tuple(d for d in DIRECTIONS if d in pending),
                                                           context, done), SAMPLE_TIMEOUT)
                    except Exception as e:
                        ok = False
                        pending -= done | getattr(e, "succeeded", set())
                        print(f"{now:%H:%M} sample failed: {e or type(e).__name__}", file=sys.stderr, flush=True)
                        if once:
                            return 1
                        if browser is not None:
                            await close_browser(browser)
                            browser = context = None
                    if once:
                        return 0
                    failures = 0 if ok else failures + 1
                    next_sample = next_sample_at(time.time(), failures)
                if not in_window(now, settings) and browser is not None:
                    await close_browser(browser)
                    browser = context = None
                if MODEL_SYNC:
                    if model_task is not None and model_task.done():
                        if not model_task.cancelled() and (model_task.exception() or model_task.result()):
                            print(f"{now:%H:%M} model training did not finish cleanly: "
                                  f"{model_task.exception() or model_task.result()}", file=sys.stderr, flush=True)
                        model_task = None
                    identity = (commute_model.windows_of(settings), commute_model.skip_key(settings), sample_path(settings, CSV))
                    if identity != model_identity:   # new windows, days off or route: retrain without waiting for the next check
                        model_identity, next_model_check = identity, 0
                    if woke:
                        next_model_check = min(next_model_check, time.time())   # catch up on any day finished while asleep
                    if model_task is None and time.time() >= next_model_check:
                        next_model_check = time.time() + MODEL_CHECK_SECONDS
                        try:
                            due = commute_model.sync_due(settings)
                        except Exception as exc:
                            due = False
                            print(f"{now:%H:%M} model check failed: {exc}", file=sys.stderr, flush=True)
                        if due:
                            model_task = asyncio.create_task(run_model_sync())
                last_tick = time.time()
                await asyncio.sleep(5)
        finally:
            if model_task is not None:
                if not model_task.done():
                    model_task.cancel()
                elif not model_task.cancelled() and model_task.exception():
                    print(f"{datetime.now(TZ):%H:%M} model training did not finish cleanly: "
                          f"{model_task.exception()}", file=sys.stderr, flush=True)
            if browser is not None:
                await close_browser(browser)
            loop.remove_signal_handler(signal.SIGTERM)


def consume_refresh():
    try:
        os.unlink(REFRESH)
        return True
    except FileNotFoundError:
        return False


@contextmanager
def sampler_lock():
    # Keep the file: unlinking a held lock would let another process lock a new inode.
    with open(LOCK, "a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SamplerAlreadyRunning("Another commute sampler is running; use --request-refresh.") from None
        handle.truncate(0)  # record the holder's pid so the app can replace a wedged sampler it did not start
        handle.write(str(os.getpid()))
        handle.flush()
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


class SamplerAlreadyRunning(RuntimeError):
    pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--request-refresh", action="store_true", help="Ask the running sampler to sample now")
    a = ap.parse_args()
    if a.request_refresh:
        with open(REFRESH, "w"):
            pass
    else:
        try:
            with sampler_lock():
                return asyncio.run(run(a.once))
        except SamplerAlreadyRunning as exc:
            print(str(exc), file=sys.stderr)
            return 75
        except (KeyboardInterrupt, asyncio.CancelledError):
            pass
        except Exception as exc:
            print(str(exc), file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Read-only JSON snapshot for the menu bar; no browser or third-party imports.

The curve shown is the self-trained "Predicted" model (commute_model.py) once it has learned enough
and is calibrated; until then (or if it ever scores worse) it is the simple typical curve: a smoothed mean
of the previous 30 calendar days of observations from the same kind of day (normal days, or weekends and days off,
which have their own curve) on a 10-minute grid, with today's samples adjusting a short-horizon forecast. Times describe drive time, not total wait
plus drive time.
"""
import argparse
import csv
import json
import math
from datetime import datetime, timedelta
from statistics import mean

try:
    import commute_model
except Exception:  # the simple curve must keep working whatever happens to the learning code
    commute_model = None
from commute_calendar import day_off_reason, is_off_day
from commute_config import CSV, TZ, gmaps_url, load_settings, route_points, window, sample_path, SettingsError

HISTORY_DAYS = 30
GRID_MINUTES = 10       # spacing of the typical-drive curve
KERNEL_MINUTES = 15     # each grid point averages samples this close, triangle-weighted
# How much of today's deviation from typical still applies this many minutes ahead. Fitted on the first
# 8 days of data (dissimilar to_work/to_home optima, averaged); revisit once a month of history exists.
FORECAST_WEIGHTS = ((10, 0.8), (30, 0.7), (60, 0.5), (120, 0.35), (240, 0.2))
RESIDUAL_SAMPLES = 3    # today's most recent samples that define the deviation
RESIDUAL_MAX_AGE = 45   # minutes; older samples say little about now
# Going home: leave as early as possible within the allowance. Going to work: leave closest to the middle of the
# window (just before or just after the rush), with the allowance pricing extra driving rather than capping it.
STRATEGY = {"to_work": "middle", "to_home": "earliest"}


MAX_MINUTES = 24 * 60   # larger values are corrupt rows, not traffic
MAX_MILES = 1000


def parse_row(row, now):
    """One CSV row as an observation; raises ValueError/TypeError/KeyError when it is unusable."""
    stamp = datetime.strptime(row["date"] + " " + row["time"], "%Y-%m-%d %H:%M").replace(tzinfo=TZ)
    minutes = float(row["minutes"])
    miles = float(row["miles"]) if row.get("miles") else None
    if (row["direction"] not in ("to_work", "to_home") or
            not math.isfinite(minutes) or not 0 < minutes <= MAX_MINUTES or stamp > now or
            (miles is not None and (not math.isfinite(miles) or not 0 <= miles <= MAX_MILES))):
        raise ValueError("invalid observation")
    return {"direction": row["direction"], "stamp": stamp, "observed_at": stamp.isoformat(),
            "minutes": minutes, "miles": miles, "via": row.get("via") or "", "note": row.get("note") or ""}


def read_observations(path, now, since=None):
    """Observations at or after `since` (an ISO date). Older rows are not parsed, except that each
    direction's newest row is always kept so `latest` survives a long sampler outage."""
    observations = []
    skipped = 0
    newest_old = {}
    try:
        with open(path, newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            required = {"date", "time", "direction", "minutes"}
            if not required.issubset(reader.fieldnames or []):
                return [], 0, "CSV is missing required columns"
            for row in reader:
                if since and row["date"] and row["date"] < since:
                    newest_old[row["direction"]] = row
                    continue
                try:
                    observations.append(parse_row(row, now))
                except (ValueError, TypeError, KeyError):
                    # Also ignore a partially appended final row while the sampler writes.
                    skipped += 1
    except FileNotFoundError:
        return [], 0, "No samples yet"
    except (OSError, UnicodeError, csv.Error) as exc:
        return sorted(observations, key=lambda item: item["stamp"]), skipped, f"Cannot read samples: {exc}"
    present = {item["direction"] for item in observations}
    for direction, row in newest_old.items():
        if direction not in present:
            try:
                observations.append(parse_row(row, now))
            except (ValueError, TypeError, KeyError):
                pass
    observations.sort(key=lambda item: item["stamp"])
    return observations, skipped, None


def public_observation(item):
    return {key: value for key, value in item.items() if key not in ("direction", "stamp")}


def grid_minutes(start, end):
    """Window edges plus every multiple of the grid spacing between them."""
    points = {start, end}
    points.update(range((start // GRID_MINUTES + 1) * GRID_MINUTES, end, GRID_MINUTES))
    return sorted(points)


def historical_point(minute, samples):
    """Typical drive near `minute`. `samples` are (minute_of_day, drive_minutes, date) within the kernel.

    The typical value is a triangle-weighted mean (smooth, unlike an integer-valued median that flips
    at thresholds); the spread is the empirical percentiles of the same samples, interpolated.
    """
    ordered = sorted(value for _, value, _ in samples)

    def percentile(fraction):
        position = (len(ordered) - 1) * fraction
        lower = math.floor(position)
        upper = math.ceil(position)
        return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)

    weights = [1 - abs(m - minute) / KERNEL_MINUTES for m, _, _ in samples]
    typical = sum(w * value for w, (_, value, _) in zip(weights, samples)) / sum(weights)
    # One freak day should not drag the line outside the band that describes the same samples.
    typical = min(max(typical, percentile(0.1)), percentile(0.9))
    return {"minute_of_day": minute, "typical_minutes": round(typical, 3),
            "sample_count": len(ordered), "day_count": len({day for _, _, day in samples}),
            "p10_minutes": percentile(0.1), "p90_minutes": percentile(0.9),
            "min_minutes": ordered[0], "max_minutes": ordered[-1]}


def typical_curve(samples, grid):
    points = []
    for minute in grid:
        near = [sample for sample in samples if abs(sample[0] - minute) < KERNEL_MINUTES]
        if near:
            points.append(historical_point(minute, near))
    return points


def curve_at(history, minute):
    """Typical drive at an arbitrary minute, interpolated between grid points (None outside the curve)."""
    for point in history:
        if point["minute_of_day"] == minute:
            return point["typical_minutes"]
    for before, after in zip(history, history[1:]):
        if before["minute_of_day"] <= minute <= after["minute_of_day"]:
            span = after["minute_of_day"] - before["minute_of_day"]
            if span > 2 * GRID_MINUTES:
                return None  # a data gap, not a trend to draw across
            fraction = (minute - before["minute_of_day"]) / span
            return before["typical_minutes"] + (after["typical_minutes"] - before["typical_minutes"]) * fraction
    return None


def forecast_weight(horizon):
    if horizon <= FORECAST_WEIGHTS[0][0]:
        return FORECAST_WEIGHTS[0][1]
    for (h0, w0), (h1, w1) in zip(FORECAST_WEIGHTS, FORECAST_WEIGHTS[1:]):
        if horizon <= h1:
            return w0 + (w1 - w0) * (horizon - h0) / (h1 - h0)
    return FORECAST_WEIGHTS[-1][1]


def live_offset(history, today, current):
    """Mean deviation of today's latest samples from the typical curve; None without recent samples."""
    deviations = []
    for minute, value in reversed(today):
        if current - minute > RESIDUAL_MAX_AGE:
            break
        typical = curve_at(history, minute)
        if typical is not None:
            deviations.append(value - typical)
            if len(deviations) == RESIDUAL_SAMPLES:
                break
    return mean(deviations) if deviations else None


def adjusted(typical, offset, horizon):
    if offset is None:
        return None
    return round(max(1.0, typical + forecast_weight(max(0.0, horizon)) * offset), 2)


def _rounded(value):
    return None if value is None else round(value, 2)


def forecast_points(history, current, offset):
    if offset is None:
        return []
    return [{"minute_of_day": point["minute_of_day"],
             "minutes": adjusted(point["typical_minutes"], offset, point["minute_of_day"] - current)}
            for point in history if point["minute_of_day"] > current - GRID_MINUTES]


def unavailable_recommendation(status, strategy, extra_drive_minutes):
    """A recommendation with no departure: `status` says why (no_data, outside_window, weekend, day_off)."""
    return {"status": status, "strategy": strategy, "depart_at": None, "minutes": None, "expected_minutes": None,
            "sample_count": 0, "fastest_minutes": None, "fastest_depart_at": None,
            "extra_drive_minutes": extra_drive_minutes}


def departure_recommendation(history, now, start, end, extra_drive_minutes=5, strategy="earliest",
                             offset=None, expected_at=None):
    """Leave earliest or latest within the allowed extra drive time of the fastest remaining estimate, or (going to
    work) near the middle of the window, trading distance from the middle against extra driving.

    Decided on the stable curve; today's live information only adjusts the expected drive time (via
    `expected_at(minute)` when given, else the legacy `offset`). The caller supplies the curve for the kind of day
    `now` is (normal day or day off); this function only looks at the time of day.
    """
    current = now.hour * 60 + now.minute + now.second / 60
    if current > end:
        return unavailable_recommendation("outside_window", strategy, extra_drive_minutes)
    # (minute, typical drive, days behind it): leaving right now, then each later grid point.
    candidates = []
    if current >= start:
        typical = curve_at(history, current)
        if typical is not None:
            nearest = min(history, key=lambda point: abs(point["minute_of_day"] - current))
            candidates.append((current, typical, nearest["day_count"]))
    candidates += [(point["minute_of_day"], point["typical_minutes"], point["day_count"])
                   for point in history if current < point["minute_of_day"] <= end]
    if not candidates:
        return unavailable_recommendation("no_data", strategy, extra_drive_minutes)

    def when(minute):
        if minute == current:
            return now
        return now.replace(hour=int(minute) // 60, minute=int(minute) % 60, second=0, microsecond=0)

    fastest = min(candidates, key=lambda c: (c[1], c[0]))
    reasonable = [c for c in candidates if c[1] <= fastest[1] + extra_drive_minutes + 1e-9]
    if strategy == "latest":
        best = max(reasonable, key=lambda c: c[0])
    elif strategy == "middle":
        # No hard cutoff: the cost of a time is how far it is from the middle of the window plus a price for every
        # extra minute of driving over the fastest option. The more extra driving you accept, the cheaper it is, so
        # the more the middle wins; with a zero allowance only the fastest times count. This sits just past the
        # rush instead of being pushed to the far end of the window by the very early, rush-free morning minutes.
        middle = (start + end) / 2
        price = 50.0 / max(extra_drive_minutes, 0.05)    # minutes off-center that one extra driving minute costs
        # The core of the rush (within a tenth of the hump's height of its peak) is never chosen, even when the
        # middle of the window happens to fall on it.
        peak = max(c[1] for c in candidates)
        pool = candidates
        if peak - fastest[1] >= 1.5:
            outside = [c for c in candidates if c[1] < peak - 0.1 * (peak - fastest[1])]
            pool = outside or candidates
        best = min(pool, key=lambda c: (abs(c[0] - middle) + price * (c[1] - fastest[1]), -c[0]))
    else:
        best = min(reasonable, key=lambda c: c[0])
    return {"status": "available", "strategy": strategy, "depart_at": when(best[0]).isoformat(),
            "minutes": round(best[1], 2),
            "expected_minutes": (None if expected_at is None else _rounded(expected_at(best[0])))
            if expected_at is not None else adjusted(best[1], offset, best[0] - current),
            "sample_count": best[2], "fastest_minutes": round(fastest[1], 2),
            "fastest_depart_at": when(fastest[0]).isoformat(),
            "extra_drive_minutes": extra_drive_minutes}


def next_normal_day(settings, day):
    """The first day after `day` that is not a weekend or day off."""
    for _ in range(14):
        day += timedelta(days=1)
        if not is_off_day(day, settings):
            return day
    return day


def target_day(settings, now, current, end):
    """The day this direction's curve and suggestion are for: today while its window is still ahead or open (a weekend
    or day off included, which has its own forecast); once it is over, the next day that is not a day off."""
    return now.date() if current <= end else next_normal_day(settings, now.date())


def _simple_status(days):
    """The label when the learning code itself is unavailable."""
    collecting = days < 3
    return {"state": "collecting" if collecting else "learning", "days": days, "ledger_days": 0, "readout": None,
            "label": "Collecting data" if collecting else "Typical (learning)"}


def _finite_quantile(q):
    """A usable model estimate: present and finite where the display curve reads it. A NaN/inf leaking out of
    the model drops the point instead of poisoning the snapshot's JSON (which forbids non-finite floats)."""
    return q is not None and all(math.isfinite(q[k]) for k in ("mean", "q10", "q90"))


def model_curves(pm, direction, now, current, start, end, today_all, target_date, usable_today,
                 observed_sibling=None):
    """Predicted display curve (live where it can be), the stable base curve for decisions, and today's pace.

    Past times show the forecast made before the window opened (what you would have been told); times still
    ahead use everything seen so far today. Returns (display, base, offset, expected_at), where expected_at(minute)
    is today's expected drive time at any departure minute (smooth, unlike the grid curve, whose past/live seam
    would jump at every 10-minute mark). `observed_sibling` is today's samples in the sibling direction, which
    inform to_home's day effect.
    """
    grid = grid_minutes(start, end)
    weekday = target_date.weekday()
    day = target_date.isoformat()
    base_q = pm.predict(direction, start - commute_model.MORNING_OF_LEAD, [], grid, weekday,
                        day=day, observed_other=observed_sibling)
    display_q = list(base_q)
    offset = None
    live_origin = int(current) if (usable_today and start - 60 <= current <= end) else None
    expected_at = None
    if live_origin is not None:
        observed = [(m, v) for m, v in today_all if start - 60 <= m <= live_origin]

        def expected_at(minute):
            q = pm.predict(direction, live_origin, observed, [minute], weekday,
                           day=day, observed_other=observed_sibling)[0]
            # A non-finite estimate would turn the whole snapshot into a json.dumps failure: answer None instead.
            return q["mean"] if q is not None and math.isfinite(q["mean"]) else None
        ahead = [i for i, t in enumerate(grid) if t > live_origin]
        if ahead:
            live_q = pm.predict(direction, live_origin, observed, [grid[i] for i in ahead], weekday,
                                day=day, observed_other=observed_sibling)
            for i, q in zip(ahead, live_q):
                if _finite_quantile(q):
                    display_q[i] = q
            first = ahead[0]
            if (_finite_quantile(live_q[0]) and base_q[first] is not None
                    and math.isfinite(base_q[first]["mean"])):
                offset = live_q[0]["mean"] - base_q[first]["mean"]
    days = pm.n_days(direction)
    display, base = [], []
    for t, q, b in zip(grid, display_q, base_q):
        if not _finite_quantile(q) or not _finite_quantile(b):
            continue
        low, high = math.ceil(q["q10"] - 1e-9), math.floor(q["q90"] + 1e-9)
        if low > high:
            low = high = round(q["mean"])
        display.append({"minute_of_day": t, "predicted_minutes": round(q["mean"], 3),
                        "typical_minutes": round(q["mean"], 3), "p10_minutes": round(q["q10"], 3),
                        "p90_minutes": round(q["q90"], 3), "range_low": low, "range_high": high,
                        "day_count": days})
        # The morning-of forecast with its own band: what the day was expected to look like before
        # any of today's traffic was seen, for drawing behind today's actuals.
        base.append({"minute_of_day": t, "typical_minutes": round(b["mean"], 3),
                     "p10_minutes": round(b["q10"], 3), "p90_minutes": round(b["q90"], 3),
                     "day_count": days})
    if expected_at is None:
        expected_at = lambda minute: curve_at(display, minute)
    return display, base, offset, expected_at


def snapshot(path=None, now=None, settings=None):
    settings = settings if settings is not None else load_settings()
    path = path if path is not None else sample_path(settings, CSV)
    now = (now or datetime.now(TZ)).astimezone(TZ)
    since = (now.date() - timedelta(days=HISTORY_DAYS + 1)).isoformat()
    observations, skipped, error = read_observations(path, now, since)
    reason = day_off_reason(now.date(), settings)
    weekend = now.weekday() >= 5
    try:
        pm, pm_why = commute_model.load_for_snapshot(path, settings)
    except Exception as exc:  # the simple curve must always be available, whatever happens to the learned state
        pm, pm_why = None, f"error: {exc}"
    result = {"generated_at": now.isoformat(), "timezone": str(TZ), "source_error": error,
              "skipped_rows": skipped, "day_status": "weekend" if weekend else "day_off" if reason else "weekday",
              "day_off_reason": None if weekend else reason, "routes": {}}
    cutoff = now.date() - timedelta(days=HISTORY_DAYS)
    current = now.hour * 60 + now.minute + now.second / 60
    # today's samples per raw direction: to_home's day-effect prior learns from to_work's day so far
    today_pairs_by_dir = {
        d: [(item["stamp"].hour * 60 + item["stamp"].minute, item["minutes"]) for item in observations
            if item["direction"] == d and item["stamp"].date() == now.date()]
        for d in ("to_work", "to_home")}
    for direction in ("to_work", "to_home"):
        points = route_points(settings, direction)
        origin, destination = points[0], points[-1]
        matching = [item for item in observations if item["direction"] == direction]
        start, end = (hour * 60 + minute for hour, minute in window(settings, direction))
        def plan(day):
            """(is a day off, is today, model stream, history samples) for forecasting `day`. The samples are only from
            days of the same kind: a Saturday says little about a Monday and vice versa."""
            off = is_off_day(day, settings)
            found = []
            for item in matching:
                stamp = item["stamp"]
                minute = stamp.hour * 60 + stamp.minute
                if (cutoff <= stamp.date() < now.date() and start <= minute <= end
                        and is_off_day(stamp.date(), settings) == off):
                    found.append((minute, item["minutes"], stamp.date()))
            key = commute_model.stream_key(direction, off) if commute_model is not None else direction
            return off, day == now.date(), key, found

        target_date = target_day(settings, now, current, end)   # today, or tomorrow / the next normal day once the window is over
        target_off, is_today, stream, samples = plan(target_date)
        if target_off and is_today and not samples:
            # No history yet for this kind of day (a new route, or the first weekend): point at the next normal day
            # instead of forecasting from nothing.
            target_date = next_normal_day(settings, now.date())
            target_off, is_today, stream, samples = plan(target_date)
        today_all = [item for item in matching if item["stamp"].date() == now.date()]
        today = [item for item in today_all if start <= item["stamp"].hour * 60 + item["stamp"].minute <= end]
        usable_today = is_today   # today's samples describe the day being forecast
        extra = settings.get("extra_drive_minutes", 5)
        strategy = STRATEGY[direction]
        today_pairs = [(item["stamp"].hour * 60 + item["stamp"].minute, item["minutes"]) for item in today_all]
        status, learned = None, None
        if pm is not None:
            try:
                status = pm.status(stream)
                if status["state"] in ("calibrating", "ready"):
                    sib_pairs = today_pairs_by_dir["to_work"] if (direction == "to_home" and is_today) else None
                    learned = model_curves(pm, stream, now, current, start, end, today_pairs,
                                           target_date, usable_today, sib_pairs)
            except Exception:  # a bad model never breaks the app: show the simple curve instead
                status, learned = None, None
        if status is None:
            days = len({day for _, _, day in samples})
            status = (commute_model.compute_status(commute_model.DEFAULT, [], {}, direction, days)
                      if commute_model is not None else _simple_status(days))
        if learned is not None:
            history, base, offset, expected_at = learned
            forecast = []
            recommendation = departure_recommendation(
                base, now, start, end, extra, strategy, None, expected_at=expected_at)
        else:
            history = typical_curve(samples, grid_minutes(start, end))  # with <3 days it is labeled "Collecting data"
            offset = None
            if usable_today and start <= current:
                offset = live_offset(history, [(m, v) for m, v in today_pairs if start <= m <= end], current)
            forecast = forecast_points(history, current, offset) if current <= end else []
            recommendation = departure_recommendation(history, now, start, end, extra, strategy, offset)
        if not is_today:   # today's window is over: say why there is no suggestion for today
            recommendation = unavailable_recommendation(
                "weekend" if weekend else "day_off" if reason else "outside_window", strategy, extra)
        # The model's report card for the Model tab: how each scored day went, how far it has got through the
        # label ladder, and which extra signals it has learned to use.
        model_block = {"label": status["label"], "state": status["state"], "days": status["days"],
                       "interval_percent": 80, "readout": status["readout"], "note": None,
                       "ledger_days": 0, "series": [], "gates": None, "phi": None, "cov": None}
        if pm is not None:
            finite = lambda v: v is not None and math.isfinite(v)
            ledger_rows = [r for r in pm.ledger
                           if r.get("direction") == stream and finite(r.get("mae")) and finite(r.get("cov80"))]
            model_block["ledger_days"] = len(ledger_rows)
            model_block["series"] = [
                {"date": r["date"], "n": r.get("n", 0), "mae": round(r["mae"], 3), "cov80": round(r["cov80"], 3),
                 "is80": round(r["is80"], 3) if finite(r["is80"]) else None,
                 "cmp_mae": round(r["cmp_mae"], 3) if finite(r.get("cmp_mae")) else None,
                 "cmp_is80": round(r["cmp_is80"], 3) if finite(r.get("cmp_is80")) else None,
                 "learned": ({k: v for k, v in r["learned"].items() if finite(v)} or None)
                            if isinstance(r.get("learned"), dict) else None}
                for r in ledger_rows[-45:]]
            model_block["gates"] = {"cal": pm.opt["gate_calibrating_days"],
                                    "ready": pm.opt["gate_predicted_days"], "cov": 0.80}
            f = pm.dirs.get(stream)
            if f is not None:
                th = getattr(f, "theta", None)
                if th is not None:
                    if finite(th.phi):
                        model_block["phi"] = round(th.phi, 3)
                    if finite(th.tau):
                        model_block["tau"] = round(th.tau, 3)
                cov = getattr(f, "cov", None) or {}
                if cov:
                    model_block["cov"] = {k: round(float(v), 4) for k, v in cov.items()
                                          if isinstance(v, (int, float)) and finite(v)}
        # The forecast day's precipitation from the model's weather feed, surfaced so the UI can say when
        # rain is part of the prediction — only on the learned path that actually factors it in.
        rain = None
        if learned is not None and pm is not None:
            w = pm.weather.get(target_date.isoformat())
            if w is not None and w.get("mm", 0) >= 1.0:
                rain = {"mm": round(float(w["mm"]), 1), "src": w.get("src"), "date": target_date.isoformat()}
        # With today's window over (or a day off), say when to leave next time: tomorrow morning, or Monday.
        next_departure = None
        if recommendation["status"] in ("outside_window", "weekend", "day_off"):
            future = now.replace(year=target_date.year, month=target_date.month, day=target_date.day,
                                 hour=0, minute=0, second=0, microsecond=0)
            if learned is not None:
                upcoming = departure_recommendation(base, future, start, end, extra, strategy, None,
                                                  expected_at=expected_at)
            else:
                upcoming = departure_recommendation(history, future, start, end, extra, strategy, None)
            if upcoming["status"] == "available":
                next_departure = dict(upcoming, for_date=target_date.isoformat())
        result["routes"][direction] = {
            "window_start": start, "window_end": end, "curve_date": target_date.isoformat(),
            "origin": origin, "destination": destination, "url": gmaps_url(points),
            "latest": public_observation(matching[-1]) if matching else None,
            "today": [public_observation(item) for item in today],  # inside the window: what the forecast uses
            "today_all": [public_observation(item) for item in today_all],  # every sample, for the all-day graph
            "history": history,
            "base": base if learned is not None else [],    # the morning-of forecast: "what we thought it would be"
            "forecast": forecast,
            "forecast_offset_minutes": None if offset is None else round(offset, 2),
            "rain": rain,
            "model": model_block,
            "recommendation": recommendation,
            "next_departure": next_departure}
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default=None, help="CSV path (defaults to sampler output)")
    args = parser.parse_args()
    try:
        print(json.dumps(snapshot(args.csv), allow_nan=False))
    except SettingsError as exc:
        parser.exit(1, f"Settings error: {exc}\n")

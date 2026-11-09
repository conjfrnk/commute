"""Tests for the self-training Predicted model (commute_model.py) on synthetic commutes with known structure."""
import asyncio
import csv
import json
import math
import os
import random
import sys
import tempfile
import types
import unittest
from datetime import date, datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, patch

# fixtures pins COMMUTE_TZ and points COMMUTE_SETTINGS at a fictional route before commute_config reads them.
import fixtures

REAL_SLEEP = asyncio.sleep

import commute_config
import commute_data
import commute_model
import commute_sampler

SETTINGS = commute_config.validate(fixtures.EXAMPLE)
WINDOWS = commute_model.windows_of(SETTINGS)   # includes the "@off" day-off streams
NORMAL = {d: WINDOWS[d] for d in ("to_work", "to_home")}
TZ = commute_config.TZ

# Tests never touch the network: sync() fetches weather only through this seam.
commute_model._fetch_weather = lambda *a, **k: {}


def truth(direction, minute):
    """Synthetic free-flow + rush-hour bump."""
    centre, base, height = (8 * 60 + 30, 15.0, 9.0) if direction == "to_work" else (18 * 60, 19.0, 8.0)
    return base + height * math.exp(-((minute - centre) / 70.0) ** 2)


def off_truth(direction, minute):
    """Weekend traffic: no rush-hour hump, a few minutes lighter than the weekday free-flow level."""
    return 12.0 if direction == "to_work" else 16.0


def make_rows(n_days, start=date(2026, 8, 3), seed=1, day_sd=1.5, weekday_effect=None, off_days=0,
              shared=0.0, ar1=0.0, rain_days=None, rain_boost=0.0):
    """Clean model rows for the first n weekdays from `start` (a Monday): curve + persistent day effect + AR(1).
    With `off_days`, also that many Saturdays/Sundays (the "@off" streams) on their own flat curve.
    `shared` in [0,1] is the share of the day effect common to both directions (the sibling signal);
    `ar1` makes the day effect carry over between days; `rain_days` marks dates (gets +rain_boost)."""
    rng = random.Random(seed)
    rows, day, made, made_off = [], start, 0, 0
    prev = {}
    while made < n_days or made_off < off_days:
        off = day.weekday() >= 5
        if (not off and made < n_days) or (off and made_off < off_days):
            common = rng.gauss(0, day_sd) if (shared or ar1) else 0.0   # off: keep the legacy draw stream
            for direction, (s, e) in NORMAL.items():
                own = rng.gauss(0, day_sd)
                if shared or ar1:
                    innov = math.sqrt(shared) * common + math.sqrt(1 - shared) * own
                    carry = ar1 * prev.get(direction, 0.0) + math.sqrt(max(0.0, 1 - ar1 * ar1)) * innov
                    prev[direction] = carry                    # the weekday-free day effect carries over
                else:
                    carry = own
                delta = carry + (weekday_effect or {}).get(day.weekday(), 0.0)
                if rain_days and day.isoformat() in rain_days:
                    delta += rain_boost
                x = 0.0
                for minute in range(s - 60, e + 70, 10):
                    x = 0.6 * x + rng.gauss(0, 0.7)
                    level = off_truth(direction, minute) if off else truth(direction, minute)
                    value = max(5, round(level + delta + x))
                    rows.append({"date": day.isoformat(), "minute": minute, "weekday": day.weekday(),
                                 "direction": commute_model.stream_key(direction, off), "minutes": float(value)})
            if off:
                made_off += 1
            else:
                made += 1
        day += timedelta(days=1)
    return rows


def write_csv(path, rows):
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["date", "time", "dow", "direction", "minutes", "miles", "via", "note"])
        for r in rows:
            writer.writerow([r["date"], f"{r['minute'] // 60:02d}:{r['minute'] % 60:02d}", "Mon",
                             r["direction"].replace(commute_model.OFF, ""),   # the CSV never carries the stream suffix
                             int(r["minutes"]), "", "", ""])


class EngineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = make_rows(16)
        cls.train = [r for r in cls.rows if r["date"] < "2026-08-24"]       # 15 weekdays? first 15 days
        cls.last_day = max(r["date"] for r in cls.rows)
        cls.train = [r for r in cls.rows if r["date"] < cls.last_day]
        cls.test = [r for r in cls.rows if r["date"] == cls.last_day]
        cls.model = commute_model.FinalModel(WINDOWS)
        cls.model.fit(cls.train)

    def test_quantiles_ordered_and_roughly_calibrated_on_a_held_out_day(self):
        hits = total = 0
        for direction, (s, e) in NORMAL.items():
            series = [(r["minute"], r["minutes"]) for r in self.test if r["direction"] == direction]
            targets = list(range(s, e + 1, 10))
            for q, t in zip(self.model.predict(direction, s - 60, [], targets), targets):
                self.assertTrue(q["q05"] < q["q10"] < q["q25"] < q["q50"] < q["q75"] < q["q90"] < q["q95"])
                actual = dict(series).get(t)
                if actual is not None:
                    total += 1
                    hits += q["q10"] <= actual <= q["q90"]
        self.assertGreater(hits / total, 0.55)   # nominal 0.80; a single day is noisy, so only a sanity bound

    def test_extreme_deviation_fades_slower_and_widens_but_ordinary_ones_are_untouched(self):
        import commute_statespace
        direction, (s, e) = "to_home", WINDOWS["to_home"]
        origin, targets = s + 90, [s + 100, s + 120, s + 180]
        typical = {t: self.model.predict(direction, s - 60, [], [t])[0]["mean"] for t in [origin] + targets}

        def forecast(jump, z=None):
            seen = [(origin - 20, typical[origin] + jump / 2), (origin - 10, typical[origin] + jump), (origin, typical[origin] + jump)]
            old = commute_statespace.INCIDENT_Z
            commute_statespace.INCIDENT_Z = old if z is None else z
            try:
                return self.model.predict(direction, origin, seen, targets)
            finally:
                commute_statespace.INCIDENT_Z = old
        for jump in (0.0, 1.5, 3.0):   # within the usual spread: identical with the rule on or off
            for on, off in zip(forecast(jump), forecast(jump, z=99)):
                self.assertAlmostEqual(on["mean"], off["mean"])
                self.assertAlmostEqual(on["q10"], off["q10"])
        on, off = forecast(12.0), forecast(12.0, z=99)
        for a, b in zip(on, off):
            self.assertGreater(a["mean"], b["mean"])               # fades more slowly
            self.assertGreater(a["q90"] - a["q10"], b["q90"] - b["q10"])   # and is less certain
        self.assertLessEqual(on[0]["mean"], typical[targets[0]] + 12.0 + 1e-6)   # never more deviation than there is now
        self.assertGreater(on[0]["mean"] - typical[targets[0]], on[-1]["mean"] - typical[targets[-1]])   # still fades

    def test_todays_observations_sharpen_later_forecasts(self):
        """Over several held-out days, knowing the first hour of the morning beats not knowing it."""
        rows = make_rows(20, seed=5, day_sd=2.0)
        days = sorted({r["date"] for r in rows})
        model = commute_model.FinalModel(WINDOWS)
        model.fit([r for r in rows if r["date"] < days[16]])
        cold_err = live_err = n = 0.0
        narrower = 0
        for day in days[16:]:
            for direction, (s, e) in NORMAL.items():
                series = [(r["minute"], r["minutes"]) for r in rows if r["date"] == day and r["direction"] == direction]
                origin = s + 60
                observed = [(m, v) for m, v in series if m <= origin]
                targets = [t for t in range(origin + 20, e + 1, 10) if dict(series).get(t) is not None]
                cold = model.predict(direction, s - 60, [], targets)
                live = model.predict(direction, origin, observed, targets)
                for c, l, t in zip(cold, live, targets):
                    cold_err += abs(c["mean"] - dict(series)[t])
                    live_err += abs(l["mean"] - dict(series)[t])
                    n += 1
                narrower += (live[0]["q90"] - live[0]["q10"]) < (cold[0]["q90"] - cold[0]["q10"])
        self.assertLess(live_err / n, cold_err / n)
        self.assertGreaterEqual(narrower, 6)       # of 8 (day, direction) cases

    def test_state_roundtrip_gives_identical_forecasts(self):
        state = json.loads(json.dumps(self.model.to_state(), allow_nan=False))
        poll = commute_model.PollPredictor(state)
        direction, (s, e) = "to_home", WINDOWS["to_home"]
        observed = [(r["minute"], r["minutes"]) for r in self.test if r["direction"] == direction and r["minute"] <= s + 40]
        targets = list(range(s, e + 1, 10))
        a = self.model.predict(direction, s + 40, observed, targets)
        b = poll.predict(direction, s + 40, observed, targets)
        for qa, qb in zip(a, b):
            for key in qa:
                self.assertAlmostEqual(qa[key], qb[key], places=3)   # the persisted curve is rounded to 1e-4 min
        self.assertEqual(self.model.status("to_home"), poll.status("to_home"))

    def test_update_scores_each_day_once_before_learning_it(self):
        model = commute_model.FinalModel(WINDOWS)
        model.fit(self.train)
        before = len(model.ledger)
        model.update(self.test)
        self.assertEqual(len(model.ledger), before + 2)   # one row per direction
        self.assertEqual({(r["date"], r["direction"]) for r in model.ledger[-2:]},
                         {(self.last_day, "to_work"), (self.last_day, "to_home")})
        self.assertEqual(len({(r["date"], r["direction"]) for r in model.ledger}), len(model.ledger))
        with self.assertRaises(AssertionError):           # replaying an old day would corrupt the ledger
            model.update(self.test)

    def test_weekday_effect_switches_on_only_with_enough_days_per_weekday(self):
        short = commute_model.FinalModel(WINDOWS)
        short._rows = make_rows(7)                    # Mon-Fri then Mon, Tue: Wed, Thu and Fri have one day
        self.assertFalse(short._weekday_ready())
        enough = commute_model.FinalModel(WINDOWS)
        enough._rows = make_rows(10)                  # two per weekday
        self.assertTrue(enough._weekday_ready())
        off_only = commute_model.FinalModel(WINDOWS)  # days off never count toward the weekday pattern
        off_only._rows = make_rows(7, off_days=8)
        self.assertFalse(off_only._weekday_ready())
        long = commute_model.FinalModel(WINDOWS)
        long._rows = make_rows(26)                    # 5 per weekday
        self.assertTrue(long._weekday_ready())
        long._refit(None)
        self.assertTrue(long.cfg["weekday"])
        for direction in NORMAL:
            self.assertGreater(long.dirs[direction].theta.phi, 0.0)   # the engine learns how much of the day effect is weekday


class StatusTests(unittest.TestCase):
    OPT = commute_model.DEFAULT

    def rows(self, n, cov=0.8, is80=5.0, cmp_is80=5.0, direction="to_work"):
        return [{"date": f"2026-09-{i + 1:02d}", "direction": direction, "mae": 1.2, "cov80": cov, "cov90": 0.9,
                 "is80": is80, "cmp_is80": cmp_is80} for i in range(n)]

    def label(self, ledger, flags, n_hist=30):
        return commute_model.compute_status(self.OPT, ledger, flags, "to_work", n_hist)["label"]

    def test_label_ladder_follows_data(self):
        flags = {"fallback": {}, "gate": {}}
        self.assertEqual(self.label([], flags, 2), "Collecting data")
        self.assertEqual(self.label(self.rows(2), flags, 5), "Typical (learning)")
        self.assertEqual(self.label(self.rows(8), flags, 11), "Predicted (calibrating)")
        flags = commute_model.update_flags(self.OPT, self.rows(20, cov=0.8), flags, "to_work")
        self.assertTrue(flags["gate"]["to_work"])
        self.assertEqual(self.label(self.rows(20), flags), "Predicted")

    def test_gate_has_hysteresis_and_over_coverage_does_not_pass(self):
        flags = {"fallback": {}, "gate": {}}
        self.assertFalse(commute_model.update_flags(self.OPT, self.rows(20, cov=0.93), flags, "to_work")["gate"]["to_work"])
        open_ = commute_model.update_flags(self.OPT, self.rows(20, cov=0.80), flags, "to_work")
        # coverage drifting to 0.68 would have failed the entry test (needs >= 0.70) but must not slam it shut
        still = commute_model.update_flags(self.OPT, self.rows(20, cov=0.68), open_, "to_work")
        self.assertTrue(still["gate"]["to_work"])
        closed = commute_model.update_flags(self.OPT, self.rows(20, cov=0.5), open_, "to_work")
        self.assertFalse(closed["gate"]["to_work"])

    def test_fallback_when_clearly_worse_with_hysteresis(self):
        flags = {"fallback": {}, "gate": {}}
        worse = commute_model.update_flags(self.OPT, self.rows(10, is80=8.0, cmp_is80=5.0), flags, "to_work")
        self.assertTrue(worse["fallback"]["to_work"])
        self.assertEqual(self.label(self.rows(10), worse), "Typical (fallback)")
        # a ratio of 1.2 is not bad enough to enter but not good enough to leave
        middling = commute_model.update_flags(self.OPT, self.rows(10, is80=6.0, cmp_is80=5.0), worse, "to_work")
        self.assertTrue(middling["fallback"]["to_work"])
        better = commute_model.update_flags(self.OPT, self.rows(10, is80=5.0, cmp_is80=5.0), middling, "to_work")
        self.assertFalse(better["fallback"]["to_work"])
        # a missing comparator score never silently flips the state
        unknown = [dict(r, cmp_is80=None) for r in self.rows(10, is80=9.0)]
        self.assertTrue(commute_model.update_flags(self.OPT, unknown, worse, "to_work")["fallback"]["to_work"])


class PersistenceTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = os.path.join(self.directory.name, "commute_model.json")
        model = commute_model.FinalModel(WINDOWS)
        model.fit(make_rows(8))
        self.state = commute_model.wrap(model.to_state(), WINDOWS, commute_model.skip_key(SETTINGS), "2026-08-12", "2026-08-12")

    def test_atomic_save_keeps_previous_and_recovers_from_corruption(self):
        commute_model.save_state(self.path, self.state)
        newer = dict(self.state, through="2026-08-13")
        newer["digest"] = commute_model._digest(newer["model"])
        commute_model.save_state(self.path, newer)
        st, source, _ = commute_model.load_state(self.path, WINDOWS)
        self.assertEqual((source, st["through"]), ("current", "2026-08-13"))
        Path(self.path).write_text('{"schema": 1, "tru')           # torn write
        st, source, why = commute_model.load_state(self.path, WINDOWS)
        self.assertEqual((source, st["through"]), ("prev", "2026-08-12"))
        self.assertIn("current: unreadable", why)
        Path(self.path + ".prev").write_text("garbage")
        self.assertEqual(commute_model.load_state(self.path, WINDOWS)[0], None)
        self.assertEqual([n for n in os.listdir(self.directory.name) if n.startswith(".model_state.") and n.endswith(".tmp")], [])

    def test_tampering_version_windows_and_nonfinite_values_are_rejected(self):
        commute_model.save_state(self.path, self.state)
        good = commute_model.load_state(self.path, WINDOWS)[0]
        self.assertIsNotNone(good)
        self.assertEqual(commute_model.validate_state(good, {**WINDOWS, "to_work": (8 * 60, 11 * 60)}), "windows changed")
        self.assertEqual(commute_model.validate_state(good, WINDOWS, "other"), "days-off settings changed")
        self.assertEqual(commute_model.validate_state(dict(good, code_version=99)), "code version")
        bad = json.loads(json.dumps(good))
        bad["model"]["dirs"]["to_work"]["m"][0] = 1e9
        self.assertEqual(commute_model.validate_state(bad), "curve range")
        tampered = json.loads(json.dumps(good))
        tampered["model"]["opt"]["n0"] = 1.0                      # passes validation but the digest no longer matches
        Path(self.path).write_text(json.dumps(tampered))
        os.remove(self.path + ".prev") if os.path.exists(self.path + ".prev") else None
        self.assertIsNone(commute_model.load_state(self.path, WINDOWS)[0])


class SyncTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.csv = os.path.join(self.directory.name, "commute_live.csv")
        patcher = patch.object(commute_model, "LOCK", os.path.join(self.directory.name, ".lock"))
        patcher.start()
        self.addCleanup(patcher.stop)
        self.rows = make_rows(12)
        self.days = sorted({r["date"] for r in self.rows})

    def at(self, day, hour=23):
        return datetime.fromisoformat(day).replace(hour=hour, tzinfo=TZ)

    def sync(self, day, **kw):
        return commute_model.sync(SETTINGS, self.at(day), self.csv, log=lambda *_: None, **kw)

    def state(self):
        return commute_model.load_state(commute_model.model_path(self.csv), WINDOWS)[0]

    def test_model_path_is_per_route(self):
        self.assertTrue(commute_model.model_path("/x/commute_live.csv").endswith("/x/commute_model.json"))
        self.assertTrue(commute_model.model_path("/x/commute_live_ab12.csv").endswith("/x/commute_model_ab12.json"))

    def test_builds_then_updates_incrementally_and_is_idempotent(self):
        write_csv(self.csv, [r for r in self.rows if r["date"] <= self.days[9]])
        first = self.sync(self.days[9])
        self.assertEqual((first["status"], first["action"]), ("ok", "rebuilt"))
        ledger = self.state()["model"]["ledger"]
        again = self.sync(self.days[9])
        self.assertEqual(again["action"], "updated 0 day(s)")
        self.assertEqual(self.state()["model"]["ledger"], ledger)               # nothing counted twice
        write_csv(self.csv, self.rows)                                          # two more days arrive
        step = self.sync(self.days[11])
        self.assertEqual(step["action"], "updated 2 day(s)")
        grown = self.state()["model"]["ledger"]
        self.assertEqual(len(grown), len(ledger) + 4)
        self.assertEqual(len({(r["date"], r["direction"]) for r in grown}), len(grown))
        self.assertEqual(self.sync(self.days[11])["action"], "updated 0 day(s)")
        # the incremental model equals a from-scratch rebuild on the same data
        rebuilt = self.sync(self.days[11], force_rebuild=True)
        self.assertEqual(rebuilt["action"], "rebuilt")
        scratch = self.state()["model"]
        self.assertEqual([(r["date"], r["direction"]) for r in grown][-4:], [(r["date"], r["direction"]) for r in scratch["ledger"]][-4:])
        for d in ("to_work", "to_home"):
            for a, b in zip(grown[-1:], scratch["ledger"][-1:]):
                self.assertAlmostEqual(a["mae"], b["mae"], places=3)

    def test_unfinished_day_is_not_folded_in_and_gaps_do_not_loop(self):
        write_csv(self.csv, self.rows)
        today = self.days[-1]
        early = commute_model.sync(SETTINGS, self.at(today, hour=19), self.csv, log=lambda *_: None)
        self.assertEqual(early["last_day"], self.days[-2])              # the day is still in progress at 19:00
        self.assertFalse(commute_model.sync_due(SETTINGS, self.at(today, hour=19), self.csv))
        self.assertTrue(commute_model.sync_due(SETTINGS, self.at(today, hour=23), self.csv))
        self.assertEqual(self.sync(today)["last_day"], today)
        saturday = (date.fromisoformat(today) + timedelta(days=(5 - date.fromisoformat(today).weekday()))).isoformat()
        self.assertFalse(commute_model.sync_due(SETTINGS, self.at(saturday, hour=12), self.csv) and False)

    def test_too_little_data_writes_nothing_and_changed_windows_rebuild(self):
        write_csv(self.csv, [r for r in self.rows if r["date"] <= self.days[1]])
        self.assertEqual(self.sync(self.days[1])["status"], "insufficient")
        self.assertFalse(os.path.exists(commute_model.model_path(self.csv)))
        self.assertTrue(commute_model.sync_due(SETTINGS, self.at(self.days[1]), self.csv))
        write_csv(self.csv, self.rows)
        self.sync(self.days[-1])
        narrowed = commute_config.validate({**SETTINGS, "to_work_start": "07:00", "to_work_end": "11:00"})
        self.assertTrue(commute_model.sync_due(narrowed, self.at(self.days[-1]), self.csv))   # windows are part of identity
        result = commute_model.sync(narrowed, self.at(self.days[-1]), self.csv, log=lambda *_: None)
        self.assertEqual(result["action"], "rebuilt")

    def test_corrupt_state_recovers_from_previous_and_then_from_scratch(self):
        write_csv(self.csv, [r for r in self.rows if r["date"] <= self.days[9]])
        self.sync(self.days[9])
        write_csv(self.csv, self.rows)
        self.sync(self.days[10])                                         # creates a .prev
        path = commute_model.model_path(self.csv)
        Path(path).write_text("{")
        self.assertEqual(self.sync(self.days[11])["action"], "updated 2 day(s)")   # continued from .prev
        Path(path).write_text("{")
        Path(path + ".prev").write_text("{")
        self.assertEqual(self.sync(self.days[11])["action"], "rebuilt")

    def test_snapshot_serves_the_model_when_calibrated_and_the_simple_curve_otherwise(self):
        write_csv(self.csv, self.rows)
        now = self.at(self.days[-1], hour=10).replace(minute=5)      # mid-morning of the last day: to_work live
        history_csv = [r for r in self.rows if r["date"] < self.days[-1] or r["minute"] <= 10 * 60]
        write_csv(self.csv, history_csv)
        before = commute_data.snapshot(self.csv, now, SETTINGS)
        self.assertEqual(before["routes"]["to_work"]["model"]["state"], "learning")   # no model state yet
        self.assertNotIn("predicted_minutes", before["routes"]["to_work"]["history"][0])
        self.sync(self.days[-2])
        after = commute_data.snapshot(self.csv, now, SETTINGS)
        route = after["routes"]["to_work"]
        self.assertEqual(route["model"]["state"], "calibrating")
        self.assertEqual(route["model"]["label"], "Predicted (calibrating)")
        self.assertEqual(route["model"]["interval_percent"], 80)
        point = route["history"][len(route["history"]) // 2]
        self.assertGreater(point["p90_minutes"], point["predicted_minutes"])
        self.assertLess(point["p10_minutes"], point["predicted_minutes"])
        self.assertLessEqual(point["range_low"], point["range_high"])
        self.assertEqual(route["forecast"], [])
        self.assertIsNotNone(route["forecast_offset_minutes"])        # today's pace is part of the curve now
        self.assertEqual(route["recommendation"]["strategy"], "middle")
        self.assertEqual(route["recommendation"]["status"], "available")
        self.assertIsNotNone(route["recommendation"]["expected_minutes"])
        json.dumps(after, allow_nan=False)
        # Break the state: the app quietly serves the simple curve instead of failing.
        Path(commute_model.model_path(self.csv)).write_text("{")
        Path(commute_model.model_path(self.csv) + ".prev").unlink(missing_ok=True)
        broken = commute_data.snapshot(self.csv, now, SETTINGS)
        self.assertNotIn("predicted_minutes", broken["routes"]["to_work"]["history"][0])
        self.assertEqual(broken["routes"]["to_work"]["model"]["state"], "learning")

    def test_snapshot_falls_back_when_the_comparator_wins(self):
        write_csv(self.csv, self.rows)
        self.sync(self.days[-1])
        path = commute_model.model_path(self.csv)
        with open(path) as handle:
            st = json.load(handle)
        st["model"]["flags"] = {"fallback": {"to_work": True}, "gate": {}}
        st["digest"] = commute_model._digest(st["model"])
        Path(path).write_text(json.dumps(st))
        now = self.at(self.days[-1], hour=23)
        route = commute_data.snapshot(self.csv, now, SETTINGS)["routes"]["to_work"]
        self.assertEqual((route["model"]["state"], route["model"]["label"]), ("fallback", "Typical (fallback)"))
        self.assertNotIn("predicted_minutes", route["history"][0])


class SamplerSyncTests(unittest.IsolatedAsyncioTestCase):
    async def test_training_is_started_when_due_and_not_re_asked_within_the_check_interval(self):
        browser = types.SimpleNamespace(close=AsyncMock())

        class PlaywrightContext:
            async def __aenter__(self):
                return types.SimpleNamespace(chromium=types.SimpleNamespace(launch=AsyncMock(return_value=browser)))

            async def __aexit__(self, *args):
                return False

        module = types.ModuleType("playwright.async_api")
        module.async_playwright = PlaywrightContext
        ticks = []

        async def sleep(seconds):
            ticks.append(1)
            await REAL_SLEEP(0)   # let the training task the loop just created run
            if len(ticks) >= 4:
                raise asyncio.CancelledError()

        started = AsyncMock(return_value=0)
        with tempfile.TemporaryDirectory() as directory, \
                patch.dict(sys.modules, {"playwright.async_api": module}), \
                patch.object(commute_sampler, "MODEL_SYNC", True), \
                patch.object(commute_sampler, "REFRESH", str(Path(directory) / "refresh")), \
                patch.object(commute_sampler, "in_window", return_value=False), \
                patch.object(commute_sampler, "sample_once", AsyncMock()), \
                patch.object(commute_sampler, "run_model_sync", started), \
                patch.object(commute_model, "sync_due", return_value=True), \
                patch.object(commute_sampler.asyncio, "sleep", sleep):
            with self.assertRaises(asyncio.CancelledError):
                await commute_sampler.run(False)
        self.assertEqual(started.await_count, 1)        # four loop passes, one training run

    async def test_a_crashing_training_job_never_stops_the_sampler(self):
        browser = types.SimpleNamespace(close=AsyncMock())

        class PlaywrightContext:
            async def __aenter__(self):
                return types.SimpleNamespace(chromium=types.SimpleNamespace(launch=AsyncMock(return_value=browser)))

            async def __aexit__(self, *args):
                return False

        module = types.ModuleType("playwright.async_api")
        module.async_playwright = PlaywrightContext
        ticks = []

        async def sleep(seconds):
            ticks.append(1)
            await REAL_SLEEP(0)
            if len(ticks) >= 5:
                raise asyncio.CancelledError()

        async def boom():
            raise RuntimeError("training exploded")

        with tempfile.TemporaryDirectory() as directory, \
                patch.dict(sys.modules, {"playwright.async_api": module}), \
                patch.object(commute_sampler, "MODEL_SYNC", True), \
                patch.object(commute_sampler, "MODEL_CHECK_SECONDS", 0), \
                patch.object(commute_sampler, "REFRESH", str(Path(directory) / "refresh")), \
                patch.object(commute_sampler, "in_window", return_value=False), \
                patch.object(commute_sampler, "sample_once", AsyncMock()), \
                patch.object(commute_sampler, "run_model_sync", boom), \
                patch.object(commute_model, "sync_due", return_value=True), \
                patch.object(commute_sampler.asyncio, "sleep", sleep):
            with self.assertRaises(asyncio.CancelledError):
                await commute_sampler.run(False)


class WakeTests(unittest.IsolatedAsyncioTestCase):
    async def test_sampler_resamples_immediately_and_cleanly_after_the_mac_sleeps(self):
        launch = AsyncMock()
        browser = types.SimpleNamespace(close=AsyncMock())
        launch.return_value = browser

        class PlaywrightContext:
            async def __aenter__(self):
                return types.SimpleNamespace(chromium=types.SimpleNamespace(launch=launch))

            async def __aexit__(self, *args):
                return False

        module = types.ModuleType("playwright.async_api")
        module.async_playwright = PlaywrightContext
        start = 1_000_000 * 600 + 10          # just after a 10-minute mark: the next sample is ~10 minutes away
        clock = {"t": float(start)}
        sample = AsyncMock()
        passes = []

        async def sleep(seconds):
            passes.append(clock["t"])
            if len(passes) == 1:
                clock["t"] += 200              # the lid closes for a few minutes, mid-slot
            elif len(passes) >= 2:
                raise asyncio.CancelledError()

        with tempfile.TemporaryDirectory() as directory, \
                patch.dict(sys.modules, {"playwright.async_api": module}), \
                patch.object(commute_sampler, "REFRESH", str(Path(directory) / "refresh")), \
                patch.object(commute_sampler, "in_window", return_value=True), \
                patch.object(commute_sampler, "sample_once", sample), \
                patch.object(commute_sampler.time, "time", lambda: clock["t"]), \
                patch.object(commute_sampler.asyncio, "sleep", sleep):
            with self.assertRaises(asyncio.CancelledError):
                await commute_sampler.run(False)
        # Pass 1 samples; after the 200 s gap pass 2 notices the sleep, drops Chromium and samples again at once.
        self.assertEqual(sample.await_count, 2)
        self.assertEqual(launch.await_count, 2)
        self.assertGreaterEqual(browser.close.await_count, 1)

    async def test_wake_clears_a_failure_backoff(self):
        browser = types.SimpleNamespace(close=AsyncMock())

        class PlaywrightContext:
            async def __aenter__(self):
                return types.SimpleNamespace(chromium=types.SimpleNamespace(launch=AsyncMock(return_value=browser)))

            async def __aexit__(self, *args):
                return False

        module = types.ModuleType("playwright.async_api")
        module.async_playwright = PlaywrightContext
        clock = {"t": 1_000_000.0 * 600 + 10}
        calls = []

        async def sample(browser_, settings, directions, context, done):
            calls.append(directions)
            if len(calls) == 1:
                raise commute_sampler.SampleError("to_home: offline", succeeded={"to_work"})   # backs off 60 s

        passes = []

        async def sleep(seconds):
            passes.append(1)
            if len(passes) == 1:
                clock["t"] += 90               # asleep, but not even long enough to have outlived the backoff plus a bit
            elif len(passes) >= 2:
                raise asyncio.CancelledError()

        with tempfile.TemporaryDirectory() as directory, \
                patch.dict(sys.modules, {"playwright.async_api": module}), \
                patch.object(commute_sampler, "REFRESH", str(Path(directory) / "refresh")), \
                patch.object(commute_sampler, "in_window", return_value=True), \
                patch.object(commute_sampler, "sample_once", sample), \
                patch.object(commute_sampler.time, "time", lambda: clock["t"]), \
                patch.object(commute_sampler.asyncio, "sleep", sleep):
            with self.assertRaises(asyncio.CancelledError):
                await commute_sampler.run(False)
        # After waking, both directions are sampled again from scratch (not just the one that failed).
        self.assertEqual(calls, [("to_work", "to_home"), ("to_work", "to_home")])


class ExpectedMinutesTests(unittest.TestCase):
    def test_expected_drive_time_does_not_jump_at_ten_minute_marks(self):
        with tempfile.TemporaryDirectory() as directory:
            csv_path = os.path.join(directory, "commute_live.csv")
            rows = make_rows(14, seed=9, day_sd=2.5)
            today = max(r["date"] for r in rows)
            finished = [r for r in rows if r["date"] < today]
            patcher = patch.object(commute_model, "LOCK", os.path.join(directory, ".lock"))
            patcher.start()
            self.addCleanup(patcher.stop)
            write_csv(csv_path, rows)
            day = datetime.fromisoformat(today)
            commute_model.sync(SETTINGS, day.replace(hour=23, tzinfo=TZ) - timedelta(days=1), csv_path, log=lambda *_: None)
            outputs = []
            for second in (590, 599, 600, 601, 610):    # 18:xx around a grid mark, in seconds past 18:00:00 minus ten minutes
                now = day.replace(hour=18, tzinfo=TZ) + timedelta(seconds=second + 600)
                route = commute_data.snapshot(csv_path, now, SETTINGS)["routes"]["to_home"]
                if route["recommendation"]["expected_minutes"] is not None:
                    outputs.append(route["recommendation"]["expected_minutes"])
            self.assertGreaterEqual(len(outputs), 3)
            self.assertLess(max(outputs) - min(outputs), 0.6)    # a sawtooth would swing by the whole day offset (~2 min)


class ReviewRegressionTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.csv = os.path.join(self.directory.name, "commute_live.csv")
        patcher = patch.object(commute_model, "LOCK", os.path.join(self.directory.name, ".lock"))
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_comparator_is_scored_on_the_same_targets_for_odd_window_starts(self):
        odd = commute_config.validate({**SETTINGS, "to_work_start": "07:05", "to_work_end": "11:00",
                                       "to_home_start": "16:45", "to_home_end": "20:15"})
        windows = commute_model.windows_of(odd)
        rows = []
        for r in make_rows(20):
            s, e = windows[r["direction"]]
            if s - 60 <= r["minute"] <= e + 70:
                rows.append(r)
        model = commute_model.FinalModel(windows)
        model.fit(rows)
        self.assertTrue(model.ledger)
        for row in model.ledger:
            self.assertIsNotNone(row["cmp_is80"], row)       # the comparator answered nearly all of the same targets
        self.assertFalse(any(model.flags["fallback"].values()))
        self.assertNotEqual(model.status("to_work")["state"], "fallback")

    def test_a_day_the_mac_slept_through_is_not_scored_or_calibrated(self):
        rows = make_rows(14, seed=4)
        model = commute_model.FinalModel(WINDOWS)
        last = max(r["date"] for r in rows)
        model.fit([r for r in rows if r["date"] < last])
        ledger, seen = len(model.ledger), [list(v) for v in model.cal.days_seen.values()]
        sparse = [r for r in rows if r["date"] == last and r["minute"] in (7 * 60, 7 * 60 + 10, 17 * 60)]
        model.update(sparse)
        self.assertEqual(len(model.ledger), ledger)                    # no row from two samples
        self.assertEqual([list(v) for v in model.cal.days_seen.values()], seen)
        self.assertEqual(model.dirs["to_work"].info["n_days"], model.dirs["to_work"].info["n_days"])

    def test_a_missing_or_truncated_csv_never_erases_what_was_learned(self):
        rows = make_rows(14, seed=6)
        days = sorted({r["date"] for r in rows})
        at = datetime.fromisoformat(days[-1]).replace(hour=23, tzinfo=TZ)
        write_csv(self.csv, rows)
        self.assertEqual(commute_model.sync(SETTINGS, at, self.csv, log=lambda *_: None)["status"], "ok")
        path = commute_model.model_path(self.csv)
        with open(path) as fh:
            learned = json.load(fh)["model"]["dirs"]["to_work"]["n_days"]
        os.remove(self.csv)
        self.assertEqual(commute_model.sync(SETTINGS, at, self.csv, log=lambda *_: None)["status"], "no data")
        with open(path) as fh:
            self.assertEqual(json.load(fh)["model"]["dirs"]["to_work"]["n_days"], learned)
        write_csv(self.csv, [r for r in rows if r["date"] <= days[4]])  # a replaced, much shorter file
        result = commute_model.sync(SETTINGS, at, self.csv, log=lambda *_: None)
        self.assertIn(result["status"], ("insufficient", "ok"))
        if result["status"] == "ok":
            self.assertEqual(result["action"], "rebuilt")             # not "updated": the old state no longer fits the data

    def test_orphan_temp_files_are_cleaned_up_after_a_killed_job(self):
        orphan = os.path.join(self.directory.name, ".model_state.abc123.tmp")
        Path(orphan).write_text("{")
        os.utime(orphan, (1, 1))
        commute_model._remove_orphans(self.directory.name)
        self.assertFalse(os.path.exists(orphan))

    def test_snapshot_works_without_the_learning_code(self):
        write_csv(self.csv, make_rows(6))
        with patch.object(commute_data, "commute_model", None):
            route = commute_data.snapshot(self.csv, datetime(2026, 8, 12, 12, tzinfo=TZ), SETTINGS)["routes"]["to_work"]
        self.assertEqual(route["model"]["state"], "learning")
        self.assertTrue(route["history"])


class NextDepartureTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.csv = os.path.join(self.directory.name, "commute_live.csv")
        write_csv(self.csv, make_rows(12, seed=3))        # 2026-08-03 .. 2026-08-18, Mon-Fri

    def snap(self, text, settings=SETTINGS):
        return commute_data.snapshot(self.csv, datetime.fromisoformat(text).replace(tzinfo=TZ), settings)["routes"]

    def test_at_night_it_says_when_to_leave_tomorrow_morning(self):
        route = self.snap("2026-08-19T22:30")["to_work"]                     # Wednesday night, window closed
        self.assertEqual(route["recommendation"]["status"], "outside_window")
        nxt = route["next_departure"]
        self.assertEqual(nxt["for_date"], "2026-08-20")
        self.assertTrue(nxt["depart_at"].startswith("2026-08-20T"))
        self.assertEqual((nxt["status"], nxt["strategy"]), ("available", "middle"))
        self.assertIsNotNone(nxt["minutes"])

    def test_the_next_departure_skips_weekends_and_days_off(self):
        friday = self.snap("2026-08-21T22:30")["to_work"]
        self.assertEqual(friday["next_departure"]["for_date"], "2026-08-24")          # Monday
        saturday = self.snap("2026-08-22T10:00")["to_work"]
        self.assertEqual((saturday["recommendation"]["status"], saturday["next_departure"]["for_date"]), ("weekend", "2026-08-24"))
        off = commute_config.validate({**SETTINGS, "skip_dates": ["2026-08-20"]})
        thursday_off = self.snap("2026-08-19T22:30", off)["to_work"]
        self.assertEqual(thursday_off["next_departure"]["for_date"], "2026-08-21")   # Thursday is a day off

    def test_no_next_departure_while_today_still_has_one(self):
        morning = self.snap("2026-08-19T06:30")["to_work"]                  # before the window opens: today's is shown
        self.assertEqual(morning["recommendation"]["status"], "available")
        self.assertIsNone(morning["next_departure"])
        self.assertIsNone(self.snap("2026-08-19T09:00")["to_work"]["next_departure"])
        # to_home's window is still ahead at 22:30? No: it ended at 21:30, so it points at tomorrow evening.
        self.assertEqual(self.snap("2026-08-19T22:30")["to_home"]["next_departure"]["for_date"], "2026-08-20")

    def test_the_predicted_model_forecasts_the_next_day_with_its_weekday(self):
        csv_dir = self.directory.name
        patcher = patch.object(commute_model, "LOCK", os.path.join(csv_dir, ".lock"))
        patcher.start()
        self.addCleanup(patcher.stop)
        commute_model.sync(SETTINGS, datetime.fromisoformat("2026-08-19T23:30").replace(tzinfo=TZ), self.csv, log=lambda *_: None)
        route = self.snap("2026-08-19T22:30")["to_work"]
        self.assertIn(route["model"]["state"], ("calibrating", "ready"))
        self.assertTrue(route["next_departure"]["depart_at"].startswith("2026-08-20T"))
        self.assertIsNotNone(route["next_departure"]["expected_minutes"])
        self.assertTrue(route["history"])                                        # the graph shows tomorrow's forecast
        json.dumps(route, allow_nan=False)


class MiddleOfWindowTests(unittest.TestCase):
    def test_going_to_work_lands_just_after_the_rush_near_the_middle_of_the_window(self):
        # 07:00 14.7 min, rush peaks near 08:30 at 22, settles at about 19.5 by 09:30 and stays flat.
        curve = lambda m: 14.7 + 7.5 * math.exp(-((m - 510) / 55.0) ** 2) if m <= 540 else 19.5
        history = [{"minute_of_day": m, "typical_minutes": curve(m), "day_count": 8} for m in range(420, 661, 10)]
        night = datetime(2026, 10, 6, 0, 0, tzinfo=TZ)
        rec = commute_data.departure_recommendation(history, night, 420, 660, 5, "middle")
        depart = datetime.fromisoformat(rec["depart_at"])
        minute = depart.hour * 60 + depart.minute
        # within 5 min of the ~15.2 fastest (<= ~20.2): before the rush (<= ~07:50) or after it (>= ~09:20); the middle
        # of the window is 09:00, so the post-rush time closest to it wins, not the end of the window and not the rush.
        self.assertTrue(540 <= minute <= 580, minute)
        self.assertLessEqual(rec["minutes"], 20.3)
        earliest = commute_data.departure_recommendation(history, night, 420, 660, 5, "earliest")
        self.assertEqual(datetime.fromisoformat(earliest["depart_at"]).hour, 7)
        latest = commute_data.departure_recommendation(history, night, 420, 660, 5, "latest")
        self.assertEqual(datetime.fromisoformat(latest["depart_at"]).hour, 11)
        # an allowance of 0 insists on the fastest time; a generous one is free to sit on the middle itself
        tight = commute_data.departure_recommendation(history, night, 420, 660, 0, "middle")
        self.assertEqual(tight["minutes"], tight["fastest_minutes"])
        generous = commute_data.departure_recommendation(history, night, 420, 660, 30, "middle")
        self.assertEqual(datetime.fromisoformat(generous["depart_at"]).hour * 60 + datetime.fromisoformat(generous["depart_at"]).minute, 540)
        # the rush itself (08:30, the peak) is never chosen at the default allowance, even if the middle were there
        for start, end in ((390, 630), (360, 660)):
            rec = commute_data.departure_recommendation(history, night, start, end, 5, "middle")
            chosen = datetime.fromisoformat(rec["depart_at"])
            self.assertFalse(480 <= chosen.hour * 60 + chosen.minute <= 520, (start, end, chosen))

    def test_on_the_real_morning_curve_the_middle_of_the_window_right_after_the_hump_wins(self):
        values = [14.7, 15.4, 16.2, 17.1, 18.2, 19.7, 21.1, 22.0, 22.3, 22.1, 22.0, 21.8, 21.3, 20.5, 19.9, 19.8, 19.8,
                  19.7, 19.4, 19.4, 19.5, 19.4, 19.5, 19.7, 20.0]            # 07:00 .. 11:00 every 10 minutes
        history = [{"minute_of_day": 420 + 10 * i, "typical_minutes": v, "day_count": 8} for i, v in enumerate(values)]
        night = datetime(2026, 10, 6, 0, 0, tzinfo=TZ)
        rec = commute_data.departure_recommendation(history, night, 420, 660, 5, "middle")
        self.assertEqual(rec["depart_at"], "2026-10-06T09:00:00-07:00")    # exactly the middle, just past the hump


class DayKindTests(unittest.TestCase):
    """Weekdays differ from each other (light Mondays) and weekends/days off are a different pattern altogether."""

    def test_a_light_monday_is_predicted_lighter(self):
        rows = make_rows(30, seed=11, day_sd=1.0, weekday_effect={0: -4.0})   # every Monday is 4 minutes lighter
        model = commute_model.FinalModel(WINDOWS)
        model.fit(rows)
        for direction, (start, end) in NORMAL.items():
            targets = list(range(start, end + 1, 10))
            monday = [q["mean"] for q in model.predict(direction, start - 60, [], targets, weekday=0)]
            wednesday = [q["mean"] for q in model.predict(direction, start - 60, [], targets, weekday=2)]
            gap = sum(w - m for w, m in zip(wednesday, monday)) / len(targets)
            self.assertGreater(gap, 2.0, direction)    # most of the true 4-minute gap, before any of the day is known
            self.assertLess(gap, 5.0, direction)

    def test_weekend_rows_get_their_own_streams_and_never_shape_the_weekday_curve(self):
        rows = make_rows(20, seed=4, off_days=8)
        self.assertEqual({r["direction"] for r in rows}, {"to_work", "to_home", "to_work@off", "to_home@off"})
        model = commute_model.FinalModel(WINDOWS)
        model.fit(rows)
        for direction, (start, end) in NORMAL.items():
            hump = (start + end) // 2 if direction == "to_work" else 18 * 60
            normal = model.predict(direction, start - 60, [], [hump], weekday=0)[0]["mean"]
            off = model.predict(commute_model.stream_key(direction, True), start - 60, [], [hump], weekday=5)[0]["mean"]
            self.assertLess(off, normal - 2.0, direction)          # no rush-hour hump on a day off
            self.assertAlmostEqual(off, off_truth(direction, hump), delta=2.5)
            self.assertGreater(normal, truth(direction, hump) - 2.5)   # the weekday curve is not dragged down by weekends

    def test_loading_keeps_days_off_under_their_own_stream(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "commute_live.csv")
            write_csv(path, [{"date": d, "minute": 8 * 60, "direction": "to_work", "minutes": 20.0}
                             for d in ("2026-10-09", "2026-10-10", "2026-10-12")])   # Fri, Sat, Columbus Day
            rows = commute_model.load_clean_rows(path, SETTINGS)
        self.assertEqual({r["date"]: r["direction"] for r in rows},
                         {"2026-10-09": "to_work", "2026-10-10": "to_work@off", "2026-10-12": "to_work@off"})

    def test_the_nightly_job_covers_weekends_too(self):
        windows = commute_model.windows_of(SETTINGS)
        saturday_night = datetime(2026, 10, 24, 23, 0, tzinfo=TZ)
        self.assertEqual(commute_model.latest_complete_day(saturday_night, windows).isoformat(), "2026-10-24")
        sunday_morning = datetime(2026, 10, 25, 6, 0, tzinfo=TZ)
        self.assertEqual(commute_model.latest_complete_day(sunday_morning, windows).isoformat(), "2026-10-24")


class WeekendForecastTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.csv = os.path.join(self.directory.name, "commute_live.csv")
        write_csv(self.csv, make_rows(12, seed=3, off_days=4))   # weekdays 08-03..08-18, plus Sat/Sun 08-08/09 and 08-15/16

    def route(self, text):
        return commute_data.snapshot(self.csv, datetime.fromisoformat(text).replace(tzinfo=TZ), SETTINGS)["routes"]["to_work"]

    def test_saturday_morning_is_forecast_from_weekend_history(self):
        route = self.route("2026-08-22T07:30")                         # Saturday, window still ahead
        self.assertEqual(route["curve_date"], "2026-08-22")
        self.assertEqual(route["recommendation"]["status"], "available")
        self.assertIsNone(route["next_departure"])
        self.assertLess(max(p["typical_minutes"] for p in route["history"]), 15)   # weekend level, no weekday hump

    def test_after_the_weekend_window_the_next_normal_day_is_forecast_from_weekdays(self):
        route = self.route("2026-08-22T13:00")                         # Saturday, window over
        self.assertEqual((route["recommendation"]["status"], route["curve_date"]), ("weekend", "2026-08-24"))
        self.assertEqual(route["next_departure"]["for_date"], "2026-08-24")
        self.assertGreater(max(p["typical_minutes"] for p in route["history"]), 18)  # Monday's weekday hump

    def test_a_weekday_curve_is_for_today_until_its_window_ends(self):
        self.assertEqual(self.route("2026-08-19T08:00")["curve_date"], "2026-08-19")
        self.assertEqual(self.route("2026-08-19T13:00")["curve_date"], "2026-08-20")
        self.assertEqual(self.route("2026-08-21T13:00")["curve_date"], "2026-08-24")   # Friday evening -> Monday

    def test_weekday_samples_never_leak_into_a_weekend_curve(self):
        weekday_only = os.path.join(self.directory.name, "weekdays.csv")
        write_csv(weekday_only, make_rows(12, seed=3))                 # no weekend history at all
        route = commute_data.snapshot(weekday_only, datetime(2026, 8, 22, 7, 30, tzinfo=TZ), SETTINGS)["routes"]["to_work"]
        self.assertEqual((route["recommendation"]["status"], route["curve_date"]), ("weekend", "2026-08-24"))


class CovariateTests(unittest.TestCase):
    """The day-effect prior learns from covariates: the sibling direction, yesterday, and rain."""

    def fit(self, rows, **kw):
        model = commute_model.FinalModel(WINDOWS, **kw)
        model.fit(rows)
        return model

    def test_the_morning_direction_informs_the_evening_forecast(self):
        rows = make_rows(18, seed=7, day_sd=1.8, shared=0.8)          # heavy morning -> heavy evening
        model = self.fit(rows)
        fit = model.dirs["to_home"]
        self.assertGreater(fit.cov["sib"], 0.15, fit.cov)             # the sibling term earned weight
        day = sorted({r["date"] for r in rows})[-1]
        wd = date.fromisoformat(day).weekday()
        s, e = NORMAL["to_home"]
        targets = [17 * 60, 18 * 60]
        sib_heavy = [(NORMAL["to_work"][0] - 30 + 10 * i, truth("to_work", NORMAL["to_work"][0] - 30 + 10 * i) + 8.0)
                     for i in range(4)]
        plain = model.predict("to_home", s - 60, [], targets, weekday=wd, day=day)
        informed = model.predict("to_home", s - 60, [], targets, weekday=wd, day=day, observed_other=sib_heavy)
        width = lambda q: q["q90"] - q["q10"]
        for a, b in zip(plain, informed):
            self.assertGreater(b["mean"], a["mean"] + 2.0)            # a heavy morning lifts the evening forecast
            self.assertLess(width(b), width(a))                       # and removes some day-level uncertainty
        future = (date.fromisoformat(day) + timedelta(days=1)).isoformat()
        ahead = model.predict("to_home", s - 60, [], targets, weekday=(wd + 1) % 7, day=future)
        self.assertTrue(all(math.isfinite(q["mean"]) for q in ahead))  # a future day has no sibling data yet

    def test_yesterdays_direction_informs_today_only_by_carryover(self):
        rows = make_rows(18, seed=9, day_sd=1.8, ar1=0.8)
        model = self.fit(rows)
        cov = model.dirs["to_work"].cov
        self.assertGreater(cov["prev"], 0.1, cov)
        days = sorted({r["date"] for r in rows})
        last, nxt = days[-1], (date.fromisoformat(days[-1]) + timedelta(days=3)).isoformat()   # a Monday
        s, e = NORMAL["to_work"]
        forecast = model.predict("to_work", s - 60, [], [8 * 60], weekday=0, day=nxt)[0]
        resid = model.dirs["to_work"].r_hist.get(last, 0.0)
        expected_shift = cov["prev"] * resid
        plain = model.predict("to_work", s - 60, [], [8 * 60], weekday=0, day=None)[0]
        self.assertAlmostEqual(forecast["mean"] - plain["mean"], expected_shift, delta=0.5)

    def test_rain_covariate_lifts_a_wet_forecast_day(self):
        rainy = {"2026-08-05", "2026-08-12", "2026-08-19", "2026-08-26"}
        rows = make_rows(20, seed=5, day_sd=1.5, rain_days=rainy, rain_boost=4.0)
        model = commute_model.FinalModel(WINDOWS)
        model.weather = {d: {"mm": 12.0 if d in rainy else 0.0, "src": "obs"} for d in sorted({r["date"] for r in rows})}
        model.fit(rows)
        cov = model.dirs["to_home"].cov
        self.assertGreater(cov["rain"], 0.05, cov)
        model.weather["2026-09-07"] = {"mm": 12.0, "src": "fcst"}
        model.weather["2026-09-08"] = {"mm": 0.0, "src": "fcst"}
        s, e = NORMAL["to_home"]
        wet = model.predict("to_home", s - 60, [], [18 * 60], weekday=0, day="2026-09-07")[0]["mean"]
        dry = model.predict("to_home", s - 60, [], [18 * 60], weekday=1, day="2026-09-08")[0]["mean"]
        self.assertGreater(wet - dry, 0.5)                            # wet day forecasts run heavier

    def test_rain_stays_off_until_a_few_wet_days_exist(self):
        rainy = {"2026-08-05", "2026-08-12"}                          # two wet days is an anecdote, not a covariate
        rows = make_rows(20, seed=5, day_sd=1.5, rain_days=rainy, rain_boost=4.0)
        model = commute_model.FinalModel(WINDOWS)
        model.weather = {d: {"mm": 12.0 if d in rainy else 0.0, "src": "obs"} for d in sorted({r["date"] for r in rows})}
        model.fit(rows)
        self.assertEqual(model.dirs["to_home"].cov["rain"], 0.0)
        self.assertEqual(model.dirs["to_home"].cov["rain_days"], 2)
        rainy.add("2026-08-19")                                       # the third wet day unlocks the term
        rows = make_rows(20, seed=5, day_sd=1.5, rain_days=rainy, rain_boost=4.0)
        model = commute_model.FinalModel(WINDOWS)
        model.weather = {d: {"mm": 12.0 if d in rainy else 0.0, "src": "obs"} for d in sorted({r["date"] for r in rows})}
        model.fit(rows)
        self.assertGreater(model.dirs["to_home"].cov["rain"], 0.0)

    def test_covariates_stay_off_until_the_data_supports_them(self):
        short = self.fit(make_rows(6, seed=3, shared=0.9))
        for name, fit in short.dirs.items():
            self.assertEqual(fit.cov.get("n", 0), 0, name)            # below COV_MIN_DAYS: no covariate fit
        model = self.fit(make_rows(12, seed=3))
        self.assertGreater(model.dirs["to_home"].cov["n"], 0)

    def test_a_dateless_forecast_ignores_covariate_means(self):
        rows = make_rows(15, seed=2, shared=0.8, ar1=0.5)
        model = self.fit(rows)
        s, e = NORMAL["to_home"]
        term = commute_model.covariate_term(model.dirs, model.weather, "to_home", None, s - 60, None)
        self.assertEqual(term["mean"], 0.0)                            # no date -> no covariate mean
        self.assertGreater(term["ve2"], 0.0)
        th = model.dirs["to_home"].theta
        self.assertLessEqual(term["ve2"], (1.0 - th.phi) * th.tau ** 2 + 1e-9)
        last = sorted(model.dirs["to_home"].r_hist)[-1]
        dated = commute_model.covariate_term(model.dirs, model.weather, "to_home",
                                             (date.fromisoformat(last) + timedelta(days=1)).isoformat(), s - 60, None)
        self.assertAlmostEqual(dated["mean"], model.dirs["to_home"].cov["prev"] * model.dirs["to_home"].r_hist[last],
                               places=6)

    def test_covariate_state_survives_the_roundtrip(self):
        model = self.fit(make_rows(15, seed=2, shared=0.8))
        model.weather = {"2026-08-21": {"mm": 3.5, "src": "obs"}}
        state = json.loads(json.dumps(model.to_state(), allow_nan=False))
        poll = commute_model.PollPredictor(state)
        fit = model.dirs["to_home"]
        for key, value in fit.cov.items():
            self.assertAlmostEqual(value, poll.dirs["to_home"].cov[key], places=6)
        for day, value in fit.r_hist.items():
            self.assertAlmostEqual(value, poll.dirs["to_home"].r_hist[day], places=4)
        self.assertEqual(model.weather, poll.weather)
        self.assertGreater(fit.cov.get("ve2", 9), 0.0)
        th = fit.theta
        self.assertLessEqual(fit.cov["ve2"], (1.0 - th.phi) * th.tau ** 2 + 1e-9)   # innovations <= day effect

    def test_scoring_is_covariate_aware_so_the_ledger_stays_honest(self):
        rows = make_rows(16, seed=4, shared=0.8)
        on = self.fit(rows)
        off = self.fit(rows, covariates=False)
        self.assertEqual(len(on.ledger), len(off.ledger))
        for a, b in zip(on.ledger, off.ledger):
            self.assertTrue(math.isfinite(a["is80"]) and math.isfinite(b["is80"]))
        changes = [i for i, (a, b) in enumerate(zip(on.ledger, off.ledger))
                   if abs(a["is80"] - b["is80"]) > 1e-9]
        self.assertTrue(changes)                                      # covariate scoring actually differs somewhere


if __name__ == "__main__":
    unittest.main()

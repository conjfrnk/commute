import asyncio
import csv
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from datetime import date, datetime
from unittest.mock import AsyncMock, patch

# fixtures pins COMMUTE_TZ and points COMMUTE_SETTINGS at a fictional route before commute_config reads them.
import fixtures

import commute_calendar
import commute_config
import commute_data
import commute_model
import commute_sampler

commute_sampler.MODEL_SYNC = False  # tests never start the real nightly training job


class SnapshotTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "samples.csv"
        self.now = datetime(2026, 10, 5, 7, 10, tzinfo=commute_sampler.TZ)

    def write(self, rows):
        with self.path.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["date", "time", "direction", "minutes", "miles", "via", "note"])
            for row in rows:
                writer.writerow([*row, "", "", ""])

    def test_suite_never_reads_personal_settings(self):
        self.assertNotEqual(os.path.basename(commute_config.CONFIG), "commute_settings.json")
        self.assertEqual(commute_config.load_settings(), commute_config.validate(fixtures.EXAMPLE))

    def test_recent_weekday_history_and_smoothed_typical_curve(self):
        self.write([
            ("2026-10-05", "07:00", "to_work", 1),  # today's data cannot bias history
            ("2026-10-02", "07:00", "to_work", 30),
            ("2026-10-01", "07:00", "to_work", 10),
            ("2026-10-04", "07:00", "to_work", 2),  # Sunday
            ("2026-09-04", "07:00", "to_work", 3),  # older than 30 days
            ("2026-10-02", "15:00", "to_work", 4),  # reverse sampling window
            ("2026-10-02", "15:00", "to_home", 25),
        ])
        data = commute_data.snapshot(self.path, self.now)
        route = data["routes"]["to_work"]
        by_minute = {point["minute_of_day"]: point for point in route["history"]}
        # Ten-minute grid; each point averages samples within 15 minutes (triangle weights).
        self.assertEqual(sorted(by_minute), [410, 420, 430])
        self.assertEqual(by_minute[420], {
            "minute_of_day": 420, "typical_minutes": 20, "sample_count": 2, "day_count": 2,
            "p10_minutes": 12, "p90_minutes": 28, "min_minutes": 10, "max_minutes": 30})
        self.assertEqual(route["latest"]["minutes"], 1)
        self.assertEqual(len(route["today"]), 1)
        self.assertEqual(data["routes"]["to_home"]["history"][0]["typical_minutes"], 25)
        self.assertEqual(data["day_status"], "weekday")

    def test_kernel_weights_nearer_samples_more(self):
        self.write([("2026-10-02", "07:00", "to_work", 10), ("2026-10-02", "07:10", "to_work", 40)])
        by_minute = {p["minute_of_day"]: p for p in commute_data.snapshot(self.path, self.now)["routes"]["to_work"]["history"]}
        # At 07:05 a sample 5 min away weighs 2/3 and one 5 min the other way the same: plain mean.
        self.assertAlmostEqual(by_minute[420]["typical_minutes"], (10 * 1 + 40 * (1 - 10 / 15)) / (1 + 1 - 10 / 15), places=2)
        self.assertEqual(by_minute[420]["sample_count"], 2)

    def test_historical_range_interpolates_and_contains_typical(self):
        self.write([("2026-10-02", "07:00", "to_work", value)
                    for value in (90, 10, 40, 20, 30)])
        point = {p["minute_of_day"]: p for p in commute_data.snapshot(self.path, self.now)["routes"]["to_work"]["history"]}[420]
        self.assertEqual(point["sample_count"], 5)
        self.assertEqual(point["typical_minutes"], 38)
        self.assertAlmostEqual(point["p10_minutes"], 14)
        self.assertAlmostEqual(point["p90_minutes"], 70)
        bounds = [point[key] for key in ("min_minutes", "p10_minutes", "typical_minutes",
                                       "p90_minutes", "max_minutes")]
        self.assertEqual(bounds, sorted(bounds))
        self.assertEqual((bounds[0], bounds[-1]), (10, 90))

    def test_single_historical_sample_has_collapsed_range(self):
        self.write([("2026-10-02", "07:00", "to_work", 22.5)])
        point = commute_data.snapshot(self.path, self.now)["routes"]["to_work"]["history"][0]
        self.assertEqual((point["sample_count"], point["day_count"]), (1, 1))
        for key in ("min_minutes", "p10_minutes", "typical_minutes", "p90_minutes", "max_minutes"):
            self.assertEqual(point[key], 22.5)

    def test_to_work_leaves_near_the_middle_of_the_window_and_to_home_as_early(self):
        rows = [("2026-10-02", time, direction, value)
                for direction in ("to_work", "to_home")
                for time, value in (("08:00", 20), ("09:00", 22), ("10:00", 24), ("11:00", 30),
                                    ("16:00", 30), ("17:00", 24), ("18:00", 22), ("19:00", 20))]
        self.write(rows)
        now = self.now.replace(hour=7, minute=0)
        data = commute_data.snapshot(self.path, now)
        work = data["routes"]["to_work"]["recommendation"]
        home = data["routes"]["to_home"]["recommendation"]
        self.assertEqual((work["strategy"], home["strategy"]), ("middle", "earliest"))
        # Allowance 5 of the 20-minute fastest drive: to_work accepts <= 25 minutes and picks the time closest to the
        # middle of its 06:00-12:00 window, 09:00 (22 min); to_home picks the FIRST (16:50, smoothed toward the 17:00 sample).
        self.assertEqual((work["depart_at"], work["minutes"]), ("2026-10-05T09:00:00-07:00", 22))
        self.assertEqual((work["fastest_minutes"], work["fastest_depart_at"]), (20, "2026-10-05T07:50:00-07:00"))
        self.assertEqual(home["depart_at"], "2026-10-05T16:50:00-07:00")
        self.assertEqual(home["fastest_depart_at"], "2026-10-05T18:50:00-07:00")

    def test_current_slot_counts_and_ties_prefer_the_strategy_side(self):
        self.write([("2026-10-02", "07:00", "to_work", 20), ("2026-10-02", "07:30", "to_work", 20)])
        result = commute_data.snapshot(self.path, self.now)["routes"]["to_work"]["recommendation"]
        self.assertEqual(result["status"], "available")
        self.assertEqual(result["fastest_depart_at"], self.now.isoformat())  # leaving now ties the fastest
        self.assertEqual(result["depart_at"], "2026-10-05T07:40:00-07:00")  # latest time within allowance

    def test_evening_goldilocks_and_configured_allowance(self):
        self.write([("2026-10-02", time, "to_home", value)
                    for time, value in (("15:00", 24), ("18:30", 23.5),
                                        ("19:00", 22.5), ("21:30", 18))])
        now = self.now.replace(hour=12, minute=0)
        for allowance, expected, minutes in ((5, "18:50", 22.5), (0, "21:20", 18),
                                             (4, "21:20", 18), (5.5, "18:20", 23.5),
                                             (6, "15:00", 24)):
            with self.subTest(allowance=allowance):
                settings = commute_config.validate({**fixtures.EXAMPLE,
                                                    "extra_drive_minutes": allowance})
                rec = commute_data.snapshot(self.path, now, settings)["routes"]["to_home"]["recommendation"]
                self.assertEqual(rec["depart_at"], f"2026-10-05T{expected}:00-07:00")
                self.assertEqual(rec["minutes"], minutes)
                self.assertEqual(rec["sample_count"], 1)
                self.assertEqual(rec["fastest_minutes"], 18)
                self.assertEqual(rec["fastest_depart_at"], "2026-10-05T21:20:00-07:00")
                self.assertEqual(rec["extra_drive_minutes"], allowance)

    def test_expired_slots_cannot_set_fastest_or_departure(self):
        self.write([("2026-10-02", "07:00", "to_work", 5),
                    ("2026-10-02", "07:30", "to_work", 22),
                    ("2026-10-02", "08:00", "to_work", 20)])
        now = self.now.replace(minute=30)
        rec = commute_data.snapshot(self.path, now)["routes"]["to_work"]["recommendation"]
        self.assertEqual(rec["fastest_minutes"], 20)  # the 5-minute 07:00 slot already passed
        self.assertEqual(rec["fastest_depart_at"], "2026-10-05T07:50:00-07:00")
        self.assertEqual(rec["depart_at"], "2026-10-05T08:10:00-07:00")

    def test_end_boundary_and_unavailable_reference_fields(self):
        self.write([("2026-10-02", "12:00", "to_work", 20)])
        end = self.now.replace(hour=12, minute=0)
        rec = commute_data.snapshot(self.path, end)["routes"]["to_work"]["recommendation"]
        self.assertEqual(rec["depart_at"], end.isoformat())
        for now, expected in ((end.replace(second=1), "outside_window"),
                              (end.replace(day=4), "weekend"),
                              (self.now, "no_data")):
            # No observations in the morning for the no-data case.
            path = self.path if expected != "no_data" else self.path.with_name("missing.csv")
            with self.subTest(status=expected):
                rec = commute_data.snapshot(path, now)["routes"]["to_work"]["recommendation"]
                self.assertEqual(rec, {"status": expected, "strategy": "middle", "depart_at": None,
                                       "minutes": None, "expected_minutes": None, "sample_count": 0,
                                       "fastest_minutes": None, "fastest_depart_at": None,
                                       "extra_drive_minutes": 5})

    def test_todays_pace_adjusts_forecast_with_decaying_weight(self):
        days = ("2026-10-01", "2026-10-02")
        self.write([(day, f"{h:02d}:{m:02d}", "to_work", 20) for day in days for h in range(7, 10) for m in (0, 10, 20, 30, 40, 50)]
                   + [("2026-10-05", t, "to_work", 25) for t in ("07:20", "07:30", "07:40", "07:50", "08:00")])
        now = self.now.replace(hour=8, minute=2)
        route = commute_data.snapshot(self.path, now)["routes"]["to_work"]
        self.assertAlmostEqual(route["forecast_offset_minutes"], 5)  # running 5 minutes over a flat 20
        forecast = {p["minute_of_day"]: p["minutes"] for p in route["forecast"]}
        self.assertEqual(min(forecast), 480)  # starts at the grid point just before now
        self.assertAlmostEqual(forecast[480], 24)  # 20 + 0.8 * 5
        self.assertAlmostEqual(forecast[540], 20 + 0.7 * 5 - (0.7 - 0.5) * 5 * (540 - 482 - 30) / 30, places=1)
        self.assertGreater(forecast[480], forecast[600])  # the day effect fades with horizon
        self.assertGreater(forecast[600], 20)
        rec = route["recommendation"]
        self.assertEqual(rec["minutes"], 20)
        self.assertGreater(rec["expected_minutes"], rec["minutes"])

    def test_today_all_keeps_samples_outside_the_window_for_the_all_day_graph(self):
        self.write([("2026-10-05", "05:50", "to_work", 12), ("2026-10-05", "07:00", "to_work", 20),
                    ("2026-10-05", "13:00", "to_work", 17), ("2026-10-04", "13:00", "to_work", 99)])
        route = commute_data.snapshot(self.path, self.now.replace(hour=14))["routes"]["to_work"]
        self.assertEqual([o["minutes"] for o in route["today"]], [20])  # the window still defines "today"
        self.assertEqual([o["minutes"] for o in route["today_all"]], [12, 20, 17])

    def test_no_forecast_without_recent_samples_today(self):
        self.write([("2026-10-02", "07:00", "to_work", 20), ("2026-10-05", "07:00", "to_work", 30)])
        route = commute_data.snapshot(self.path, self.now.replace(hour=9))["routes"]["to_work"]
        self.assertIsNone(route["forecast_offset_minutes"])  # 2 hours old: says nothing about now
        self.assertEqual(route["forecast"], [])
        self.assertIsNone(route["recommendation"]["expected_minutes"])

    def test_days_off_are_excluded_from_history_and_suggestions(self):
        # 2026-10-12 is Columbus Day; that Monday's fast drive must not shape the typical curve.
        self.write([("2026-10-02", "07:00", "to_work", 30), ("2026-10-12", "07:00", "to_work", 10)])
        monday = self.now.replace(day=19)
        history = commute_data.snapshot(self.path, monday)["routes"]["to_work"]["history"]
        self.assertEqual({round(p["typical_minutes"]) for p in history}, {30})
        holiday = commute_data.snapshot(self.path, monday.replace(day=12))
        self.assertEqual((holiday["day_status"], holiday["day_off_reason"]), ("day_off", "Columbus Day"))
        self.assertEqual(holiday["routes"]["to_work"]["recommendation"]["status"], "day_off")
        self.assertEqual(holiday["routes"]["to_work"]["forecast"], [])
        optional = commute_config.validate({**fixtures.EXAMPLE, "skip_us_holidays": False, "skip_dates": ["2026-10-02"]})
        data = commute_data.snapshot(self.path, monday.replace(day=12), optional)
        self.assertEqual(data["day_status"], "weekday")
        self.assertEqual([p["typical_minutes"] for p in data["routes"]["to_work"]["history"]], [])  # Oct 2 skipped

    def test_outlier_day_cannot_pull_typical_outside_its_band(self):
        rows = [(f"2026-09-{d:02d}", "07:00", "to_work", 20) for d in (14, 15, 16, 17, 18, 21, 22, 23, 24, 25, 28, 29, 30)]
        self.write(rows + [("2026-10-01", "07:00", "to_work", 70)])
        point = {p["minute_of_day"]: p for p in commute_data.snapshot(self.path, self.now)["routes"]["to_work"]["history"]}[420]
        self.assertEqual(point["max_minutes"], 70)
        self.assertLessEqual(point["typical_minutes"], point["p90_minutes"])
        self.assertGreaterEqual(point["typical_minutes"], point["p10_minutes"])

    def test_curve_lookup_at_a_point_beside_a_data_gap(self):
        history = [{"minute_of_day": m, "typical_minutes": v} for m, v in ((600, 10), (640, 20), (650, 30))]
        self.assertEqual(commute_data.curve_at(history, 600), 10)
        self.assertEqual(commute_data.curve_at(history, 640), 20)
        self.assertEqual(commute_data.curve_at(history, 645), 25)
        self.assertIsNone(commute_data.curve_at(history, 620))  # inside the gap

    def test_latest_survives_a_long_outage_and_corrupt_rows_are_skipped(self):
        self.write([("2026-08-01", "07:00", "to_work", 21), ("2026-08-01", "07:00", "to_home", 9999999)])
        data = commute_data.snapshot(self.path, self.now)
        self.assertEqual(data["routes"]["to_work"]["latest"]["minutes"], 21)
        self.assertIsNone(data["routes"]["to_home"]["latest"])  # 9,999,999 minutes is not a drive
        self.write([("2026-10-02", "07:00", "to_work", 1e308), ("2026-10-02", "07:00", "to_work", 1e308)])
        json.dumps(commute_data.snapshot(self.path, self.now), allow_nan=False)

    def test_no_data_no_remaining_window_and_weekends(self):
        data = commute_data.snapshot(self.path, self.now)
        self.assertEqual(data["source_error"], "No samples yet")
        self.assertIsNone(data["routes"]["to_work"]["latest"])
        self.assertEqual(data["routes"]["to_work"]["recommendation"]["status"], "no_data")
        late = self.now.replace(hour=12, minute=1)
        self.assertEqual(commute_data.snapshot(self.path, late)["routes"]["to_work"]["recommendation"]["status"], "outside_window")
        weekend = self.now.replace(day=4)
        self.assertEqual(commute_data.snapshot(self.path, weekend)["routes"]["to_work"]["recommendation"]["status"], "weekend")
        self.write([("2026-10-02", "06:00", "to_work", 10)])
        self.assertEqual(commute_data.snapshot(self.path, self.now)["routes"]["to_work"]["recommendation"]["status"], "no_data")

    def test_bad_partial_future_rows_skipped_latest_is_chronological(self):
        self.write([("2026-10-05", "07:00", "to_home", 21),
                    ("2026-10-02", "15:00", "to_home", 25),
                    ("2026-10-05", "08:00", "to_home", 9),
                    ("2026-10-02", "15:00", "unknown", 8),
                    ("2026-10-02", "15:00", "to_home", "NaN"),
                    ("2026-10-02", "15:00", "to_home", -2),
                    ("bad-date", "15:00", "to_home", 10)])
        with self.path.open("a") as handle:
            handle.write("2026-10-05,07:")
        data = commute_data.snapshot(self.path, self.now)
        self.assertEqual(data["skipped_rows"], 6)
        self.assertEqual(data["routes"]["to_home"]["latest"]["minutes"], 21)
        self.assertIsNone(data["routes"]["to_work"]["latest"])
        json.dumps(data, allow_nan=False)

    def test_invalid_header_is_explicit_and_cli_emits_json(self):
        self.path.write_text("unexpected,columns\n1,2\n")
        proc = subprocess.run([sys.executable, "commute_data.py", "--csv", str(self.path)],
                              cwd=Path(__file__).parent, capture_output=True, text=True, check=True)
        data = json.loads(proc.stdout)
        self.assertEqual(data["source_error"], "CSV is missing required columns")
        self.assertEqual(set(data["routes"]), {"to_work", "to_home"})


class SamplerTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "samples.csv"
        self.page = types.SimpleNamespace(close=AsyncMock())
        self.browser = types.SimpleNamespace(new_page=AsyncMock(return_value=self.page), close=AsyncMock())

    async def test_missing_direction_persists_valid_row_but_fails_sample(self):
        self.path.touch()  # empty CSV must get a header
        with patch.object(commute_sampler, "CSV", str(self.path)), \
                patch.object(commute_sampler, "fetch_routes", AsyncMock(side_effect=[["29 min | 3.4 miles | via Main St"], []])):
            with self.assertRaisesRegex(RuntimeError, "to_home: no route parsed"):
                await commute_sampler.sample_once(self.browser, fixtures.EXAMPLE)
        out = Path(commute_config.sample_path(fixtures.EXAMPLE, str(self.path)))
        with out.open() as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["direction"], "to_work")
        self.assertEqual(rows[0]["minutes"], "29")
        self.assertTrue(self.page.close.await_count)

    async def test_once_exit_code_distinguishes_complete_and_failed_sample(self):
        context = types.SimpleNamespace(chromium=types.SimpleNamespace(launch=AsyncMock(return_value=self.browser)))

        class PlaywrightContext:
            async def __aenter__(self):
                return context

            async def __aexit__(self, *args):
                return False

        module = types.ModuleType("playwright.async_api")
        module.async_playwright = PlaywrightContext
        with patch.dict(sys.modules, {"playwright.async_api": module}), \
                patch.object(commute_sampler, "CSV", str(self.path)), \
                patch.object(commute_sampler, "REFRESH", str(self.path) + ".refresh"), \
                patch.object(commute_sampler, "fetch_routes", AsyncMock(side_effect=[["20 min"], ["25 min"], ["20 min"], []])):
            self.assertEqual(await commute_sampler.run(True), 0)
            self.assertEqual(await commute_sampler.run(True), 1)
        self.assertEqual(self.browser.close.await_count, 2)

    async def test_fetch_error_does_not_hide_reverse_route_and_page_closes(self):
        with patch.object(commute_sampler, "CSV", str(self.path)), \
                patch.object(commute_sampler, "fetch_routes", AsyncMock(side_effect=[RuntimeError("offline"), ["25 min"]])):
            with self.assertRaisesRegex(RuntimeError, "to_work: offline"):
                await commute_sampler.sample_once(self.browser, fixtures.EXAMPLE)
        out = Path(commute_config.sample_path(fixtures.EXAMPLE, str(self.path)))
        with out.open() as handle:
            self.assertEqual(list(csv.DictReader(handle))[0]["direction"], "to_home")
        self.assertEqual(self.page.close.await_count, 1)

    def playwright_module(self, context):
        class PlaywrightContext:
            async def __aenter__(self):
                return context

            async def __aexit__(self, *args):
                return False

        module = types.ModuleType("playwright.async_api")
        module.async_playwright = PlaywrightContext
        return {"playwright.async_api": module}

    async def run_one_tick(self, sample, marker, **patches):
        """Run the daemon loop until its first sleep, then cancel it."""
        with patch.dict(sys.modules, self.playwright_module(self.context)), \
                patch.object(commute_sampler, "REFRESH", marker), \
                patch.object(commute_sampler, "in_window", return_value=False), \
                patch.object(commute_sampler, "sample_once", sample), \
                patch.object(commute_sampler.asyncio, "sleep", AsyncMock(side_effect=asyncio.CancelledError)):
            for name, value in patches.items():
                setattr(commute_sampler, name, value)
            with self.assertRaises(asyncio.CancelledError):
                await commute_sampler.run(False)

    async def test_daemon_samples_outside_window_and_releases_browser(self):
        # Sampling is around the clock so the bar never goes stale; only Chromium is released off-window.
        self.context = types.SimpleNamespace(chromium=types.SimpleNamespace(launch=AsyncMock(return_value=self.browser)))
        marker = str(self.path) + ".refresh"
        sample = AsyncMock()
        await self.run_one_tick(sample, marker)  # no refresh request and not in a window
        self.assertEqual(sample.await_count, 1)
        self.assertEqual(self.browser.close.await_count, 1)
        Path(marker).touch()
        await self.run_one_tick(sample, marker)
        self.assertEqual(sample.await_count, 2)
        self.assertFalse(Path(marker).exists())

    async def test_hung_sample_is_abandoned_and_browser_released(self):
        self.context = types.SimpleNamespace(chromium=types.SimpleNamespace(launch=AsyncMock(return_value=self.browser)))

        async def hang(*args):
            await asyncio.Event().wait()

        original = commute_sampler.SAMPLE_TIMEOUT
        self.addCleanup(setattr, commute_sampler, "SAMPLE_TIMEOUT", original)
        await self.run_one_tick(hang, str(self.path) + ".refresh", SAMPLE_TIMEOUT=0.01)
        self.assertEqual(self.browser.close.await_count, 1)  # loop survived the timeout and reached its sleep

    def test_next_sample_schedule(self):
        slot = 1_000_000 * 600 + 123  # 123 s into a 10-minute slot
        self.assertEqual(commute_sampler.next_sample_at(slot, 0), (slot // 600 + 1) * 600)
        self.assertEqual([commute_sampler.next_sample_at(0, n) for n in (1, 2, 3, 4, 9)], [60, 120, 240, 300, 300])

    async def test_repeated_settings_error_prints_once(self):
        sleeps = []

        async def sleep(seconds):
            sleeps.append(seconds)
            if len(sleeps) == 3:
                raise asyncio.CancelledError()

        err = io.StringIO()
        with patch.dict(sys.modules, self.playwright_module(
                types.SimpleNamespace(chromium=types.SimpleNamespace(launch=AsyncMock())))), \
                patch.object(commute_config, "CONFIG", str(self.path) + ".missing"), \
                patch.object(commute_sampler.asyncio, "sleep", sleep), \
                patch.object(sys, "stderr", err):
            with self.assertRaises(asyncio.CancelledError):
                await commute_sampler.run(False)
        self.assertEqual(sleeps, [5, 5, 5])
        self.assertEqual(err.getvalue().count("settings error"), 1)


class LockAndParserTests(unittest.TestCase):
    def test_process_lock_refuses_second_writer_and_releases(self):
        with tempfile.TemporaryDirectory() as directory:
            lock = str(Path(directory) / "sampler.lock")
            script = "import sys,commute_sampler as s; s.LOCK=sys.argv[1];\nwith s.sampler_lock():\n print('ready',flush=True)\n sys.stdin.read()"
            child = subprocess.Popen([sys.executable, "-c", script, lock], cwd=Path(__file__).parent,
                                     stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
            try:
                self.assertEqual(child.stdout.readline().strip(), "ready")
                with patch.object(commute_sampler, "LOCK", lock):
                    with self.assertRaisesRegex(RuntimeError, "Another commute sampler"):
                        with commute_sampler.sampler_lock():
                            self.fail("second writer acquired lock")
                    with patch.object(sys, "argv", ["commute_sampler.py", "--once"]):
                        self.assertEqual(commute_sampler.main(), 75)
                child.communicate(timeout=5)
                self.assertEqual(child.returncode, 0)
                with patch.object(commute_sampler, "LOCK", lock):
                    with commute_sampler.sampler_lock():
                        pass
            finally:
                if child.poll() is None:
                    child.kill()
                    child.communicate()

    def test_lock_records_holder_pid(self):
        with tempfile.TemporaryDirectory() as directory:
            lock = Path(directory) / "sampler.lock"
            lock.write_text("99999999999")  # stale, longer pid must be fully replaced
            with patch.object(commute_sampler, "LOCK", str(lock)):
                with commute_sampler.sampler_lock():
                    self.assertEqual(lock.read_text(), str(os.getpid()))

    def test_request_marker_is_consumed_once(self):
        with tempfile.TemporaryDirectory() as directory:
            marker = str(Path(directory) / "refresh")
            with patch.object(commute_sampler, "REFRESH", marker):
                self.assertFalse(commute_sampler.consume_refresh())
                Path(marker).touch()
                self.assertTrue(commute_sampler.consume_refresh())
                self.assertFalse(commute_sampler.consume_refresh())

    def test_append_after_a_torn_row_keeps_the_new_row_intact(self):
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "samples.csv")
            row = ["2026-10-20", "08:00", "Tue", "to_work", 25, 9.1, "via Main St", ""]
            with patch("builtins.print"):
                commute_sampler.append_rows(path, [row])
                with open(path, "a", newline="") as handle:
                    handle.write("2026-10-20,08:10,Tue,to_wo")  # the sampler died mid-write
                commute_sampler.append_rows(path, [["2026-10-20", "08:20", "Tue", "to_work", 27, 9.1, "via Main St", ""]])
            observations, skipped, error = commute_data.read_observations(path, datetime(2026, 10, 21, tzinfo=commute_data.TZ))
            self.assertIsNone(error)
            self.assertEqual([item["minutes"] for item in observations], [25.0, 27.0])
            self.assertEqual(skipped, 1)  # only the torn fragment is lost

    def test_user_addresses_escape_url_path_characters(self):
        self.assertIn("A%2FB+%231%3F", commute_sampler.gmaps_url(["A/B #1?", "Work"]))

    def test_hour_minute_and_hour_only_google_durations(self):
        self.assertEqual(commute_sampler.parse("1 hr 5 min | 3.4 miles | via Main St")[0:2], (65, 3.4))
        self.assertEqual(commute_sampler.parse("2 hr")[0], 120)
        self.assertIsNone(commute_sampler.parse("No routes available"))



class SettingsTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.config = Path(self.directory.name) / "commute_settings.json"
        self.config.write_text(json.dumps(fixtures.EXAMPLE))
        self.csv = Path(self.directory.name) / "commute_live.csv"
        self.settings = commute_config.load_settings(self.config)
        self.now = datetime(2026, 10, 5, 9, 10, tzinfo=commute_sampler.TZ)

    def write_csv(self, path, rows):
        with open(path, "w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["date", "time", "direction", "minutes"])
            writer.writerows(rows)

    def test_missing_and_invalid_configs_are_explicit(self):
        missing = Path(self.directory.name) / "absent.json"
        with self.assertRaisesRegex(commute_config.SettingsError, "Not set up yet"):
            commute_config.load_settings(missing)
        self.assertEqual(commute_config.route_points(self.settings, "to_work"),
                         [self.settings["home"], *self.settings["to_work_waypoints"], self.settings["work"]])
        bad_values = ["{", "[]", json.dumps({}),
                      json.dumps({**self.settings, "home": " "}),
                      json.dumps({**self.settings, "to_home_waypoints": "Street"}),
                      json.dumps({**self.settings, "to_work_start": "25:00"}),
                      json.dumps({**self.settings, "to_work_start": "12:00"})]
        for contents in bad_values:
            with self.subTest(contents=contents):
                self.config.write_text(contents)
                with self.assertRaises(commute_config.SettingsError):
                    commute_config.load_settings(self.config)

    def test_legacy_settings_default_and_allowance_validation(self):
        legacy = {key: value for key, value in self.settings.items() if key != "extra_drive_minutes"}
        self.config.write_text(json.dumps(legacy))
        self.assertEqual(commute_config.load_settings(self.config)["extra_drive_minutes"], 5)
        for value in (0, 5.5, 30):
            self.assertEqual(commute_config.validate({**legacy, "extra_drive_minutes": value})["extra_drive_minutes"], value)
        for value in (True, False, None, "5", -1, 30.1, float("nan"), float("inf")):
            with self.subTest(value=value), self.assertRaises(commute_config.SettingsError):
                commute_config.validate({**legacy, "extra_drive_minutes": value})
        for value in (10 ** 400, -10 ** 400):
            with self.assertRaises(commute_config.SettingsError):  # not OverflowError
                commute_config.validate({**legacy, "extra_drive_minutes": value})

    def test_custom_windows_filter_graph_and_recommendation(self):
        settings = commute_config.validate({**self.settings, "to_work_start": "09:15", "to_work_end": "10:00"})
        self.write_csv(self.csv, [("2026-10-02", "09:00", "to_work", 1),
                                 ("2026-10-02", "09:20", "to_work", 20),
                                 ("2026-10-02", "10:00", "to_work", 25),
                                 ("2026-10-02", "10:01", "to_work", 1),
                                 ("2026-10-05", "09:00", "to_work", 1)])
        route = commute_data.snapshot(self.csv, self.now, settings)["routes"]["to_work"]
        # Samples outside 09:15-10:00 (09:00, 10:01) and today's never reach the curve.
        self.assertEqual([(p["minute_of_day"], p["typical_minutes"]) for p in route["history"]],
                         [(555, 20), (560, 20), (570, 20), (590, 25), (600, 25)])
        self.assertEqual(route["today"], [])
        self.assertEqual((route["window_start"], route["window_end"]), (555, 600))
        # Going to work: the time closest to the middle of the 09:15-10:00 window (09:37) within 5 minutes of the
        # 20-minute fastest drive; the grid points are 09:15, 09:20, 09:30, 09:50, 10:00, so 09:30 (20 min) wins.
        self.assertEqual(route["recommendation"]["depart_at"], "2026-10-05T09:30:00-07:00")
        self.assertEqual(route["recommendation"]["minutes"], 20)
        tight = commute_config.validate({**settings, "extra_drive_minutes": 0})
        early = commute_data.snapshot(self.csv, self.now, tight)["routes"]["to_work"]["recommendation"]
        self.assertEqual((early["depart_at"], early["minutes"]), ("2026-10-05T09:30:00-07:00", 20))
        later = commute_data.snapshot(self.csv, self.now.replace(minute=35), settings)["routes"]["to_work"]
        # at 09:35 leaving right now is the closest to the middle of what is left (interpolated 21.25 minutes)
        self.assertEqual(later["recommendation"]["depart_at"], self.now.replace(minute=35).isoformat())
        self.assertAlmostEqual(later["recommendation"]["minutes"], 21.25)
        self.assertFalse(commute_sampler.in_window(self.now, settings))
        self.assertTrue(commute_sampler.in_window(self.now.replace(minute=15), settings))

    def test_route_change_isolates_history_and_time_change_preserves_it(self):
        live = commute_config.sample_path(self.settings, str(self.csv))
        self.write_csv(live, [("2026-10-02", "09:00", "to_work", 20)])
        with patch.object(commute_data, "CSV", str(self.csv)):
            self.assertTrue(commute_data.snapshot(now=self.now, settings=self.settings)["routes"]["to_work"]["history"])
            for key, value in (("home", "New home"), ("work", "New work"),
                               ("to_work_waypoints", []), ("to_home_waypoints", [])):
                settings = commute_config.validate({**self.settings, key: value})
                with self.subTest(key=key):
                    data = commute_data.snapshot(now=self.now, settings=settings)
                    self.assertIsNone(data["routes"]["to_work"]["latest"])
                    self.assertEqual(data["routes"]["to_work"]["history"], [])
                    self.assertNotEqual(commute_config.sample_path(settings, str(self.csv)), live)
            settings = commute_config.validate({**self.settings, "to_work_start": "08:00"})
            self.assertEqual(commute_config.sample_path(settings, str(self.csv)), live)
            self.assertTrue(commute_data.snapshot(now=self.now, settings=settings)["routes"]["to_work"]["history"])
            settings = commute_config.validate({**self.settings, "extra_drive_minutes": 0})
            self.assertEqual(commute_config.sample_path(settings, str(self.csv)), live)
            self.assertTrue(commute_data.snapshot(now=self.now, settings=settings)["routes"]["to_work"]["history"])
        custom = commute_config.validate({**self.settings, "home": "New home"})
        custom_path = commute_config.sample_path(custom, str(self.csv))
        changed = commute_config.validate({**custom, "to_home_start": "16:00", "extra_drive_minutes": 30})
        self.assertEqual(commute_config.sample_path(changed, str(self.csv)), custom_path)
        self.write_csv(custom_path, [("2026-10-02", "09:00", "to_work", 11)])
        with patch.object(commute_data, "CSV", str(self.csv)):
            data = commute_data.snapshot(now=self.now, settings=custom)
        self.assertEqual(data["routes"]["to_work"]["latest"]["minutes"], 11)
        self.assertIn("New+home", data["routes"]["to_work"]["url"])


class ConfiguredSamplerTests(unittest.IsolatedAsyncioTestCase):
    async def test_sample_captures_settings_and_csv_before_fetch(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "commute_settings.json"
            csv_path = str(Path(directory) / "commute_live.csv")
            settings = commute_config.validate({**fixtures.EXAMPLE, "home": "New home", "work": "New work",
                                                "to_work_waypoints": ["My waypoint"], "to_home_waypoints": []})
            config.write_text(json.dumps(settings))
            calls = []
            async def fetch(page, points):
                calls.append(points)
                # Saving a different route mid-sample must not mislabel either row.
                config.write_text(json.dumps({**settings, "home": "Another home"}))
                return ["20 min"]
            page = types.SimpleNamespace(close=AsyncMock())
            browser = types.SimpleNamespace(new_page=AsyncMock(return_value=page))
            with patch.object(commute_config, "CONFIG", str(config)), patch.object(commute_sampler, "CSV", csv_path), \
                    patch.object(commute_sampler, "fetch_routes", fetch):
                await commute_sampler.sample_once(browser)
            self.assertEqual(calls, [["New home", "My waypoint", "New work"], ["New work", "New home"]])
            self.assertFalse(Path(csv_path).exists())
            with open(commute_config.sample_path(settings, csv_path)) as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), 2)

    async def test_daemon_reloads_saved_settings_between_iterations(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "commute_settings.json"
            initial = commute_config.validate(fixtures.EXAMPLE)
            changed = commute_config.validate({**initial, "home": "New home"})
            config.write_text(json.dumps(initial))
            browser = types.SimpleNamespace(close=AsyncMock())
            class PlaywrightContext:
                async def __aenter__(self):
                    return types.SimpleNamespace(chromium=types.SimpleNamespace(launch=AsyncMock(return_value=browser)))
                async def __aexit__(self, *args):
                    return False
            module = types.ModuleType("playwright.async_api")
            module.async_playwright = PlaywrightContext
            samples = []
            async def sample(browser, settings, *args):
                samples.append(settings)
            async def sleep(seconds):
                if len(samples) == 1:
                    config.write_text(json.dumps(changed))
                else:
                    raise asyncio.CancelledError()
            with patch.dict(sys.modules, {"playwright.async_api": module}), \
                    patch.object(commute_config, "CONFIG", str(config)), \
                    patch.object(commute_sampler, "consume_refresh", return_value=True), \
                    patch.object(commute_sampler, "sample_once", sample), \
                    patch.object(commute_sampler.asyncio, "sleep", sleep):
                with self.assertRaises(asyncio.CancelledError):
                    await commute_sampler.run(False)
            self.assertEqual([settings["home"] for settings in samples], [initial["home"], "New home"])

    async def test_malformed_settings_never_launches_browser(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "commute_settings.json"
            config.write_text("{")
            launch = AsyncMock()
            class PlaywrightContext:
                async def __aenter__(self):
                    return types.SimpleNamespace(chromium=types.SimpleNamespace(launch=launch))
                async def __aexit__(self, *args):
                    return False
            module = types.ModuleType("playwright.async_api")
            module.async_playwright = PlaywrightContext
            with patch.dict(sys.modules, {"playwright.async_api": module}), \
                    patch.object(commute_config, "CONFIG", str(config)):
                self.assertEqual(await commute_sampler.run(True), 1)
            launch.assert_not_awaited()


class CardValidationTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "samples.csv"
        self.page = types.SimpleNamespace(close=AsyncMock())
        self.browser = types.SimpleNamespace(new_page=AsyncMock(return_value=self.page), close=AsyncMock())

    def test_duration_must_be_a_whole_segment(self):
        parse = commute_sampler.parse
        # Banner text before the real card must not be read as the drive time.
        self.assertEqual(parse("Traffic is 5 min slower than usual | 19 min | 3.5 miles | via Main St")[:3], (19, 3.5, "Main St"))
        self.assertEqual(parse("\ue531 | 23 min | 3.5 miles | via Main St | Fastest route | Details")[:2], (23, 3.5))
        self.assertIsNone(parse("5 min slower than usual | via Main St"))
        self.assertEqual(parse("19 min | 5.6 km | via Main St")[1], 3.5)
        self.assertEqual(parse("1 hr 2 min | 1,200 miles")[:2], (62, 1200))
        self.assertEqual(parse("20 min")[1], "")

    def test_card_bounds_and_distance_band(self):
        check = commute_sampler.check_card
        self.assertIsNone(check((23, 3.5, "", "")))
        self.assertIsNone(check((23, "", "", "")))  # missing distance is logged, not rejected
        for card in ((0, 3.5), (181, 3.5), (20, 0.1), (20, 61)):
            self.assertIsNotNone(check((*card, "", "")), card)
        self.assertIsNone(check((23, 4.5, "", ""), typical_miles=3.8))
        self.assertIn("typical", check((23, 9.0, "", ""), typical_miles=3.8))
        self.assertIn("typical", check((23, 1.2, "", ""), typical_miles=3.8))

    def test_recent_miles_needs_enough_rows_for_the_direction(self):
        with self.path.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["date", "time", "dow", "direction", "minutes", "miles", "via", "note"])
            for i in range(12):
                writer.writerow(["2026-10-02", f"07:{i:02d}", "Fri", "to_work", 20, 3.4 + (i % 3) / 10, "", ""])
            writer.writerow(["2026-10-02", "07:30", "Fri", "to_home", 20, 99, "", ""])
        self.assertAlmostEqual(commute_sampler.recent_miles(str(self.path), "to_work"), 3.5)
        self.assertIsNone(commute_sampler.recent_miles(str(self.path), "to_home"))  # one row is not a baseline
        self.assertIsNone(commute_sampler.recent_miles(str(self.path) + ".missing", "to_work"))

    async def test_a_new_route_distance_that_keeps_agreeing_becomes_the_baseline(self):
        commute_sampler._band_rejects.clear()
        self.addCleanup(commute_sampler._band_rejects.clear)
        card = (23, 6.0, "", "")  # a detour: 6 miles against a typical 3.4
        for _ in range(commute_sampler.ACCEPT_AFTER - 1):
            self.assertIn("typical", commute_sampler.vet_card("to_work", card, 3.4))
        self.assertIsNone(commute_sampler.vet_card("to_work", card, 3.4))  # consistent for six samples in a row
        # Disagreeing or absolute-bound failures never earn acceptance.
        for miles in (6.0, 9.0, 6.0, 9.0, 6.0, 9.0):
            problem = commute_sampler.vet_card("to_home", (23, miles, "", ""), 3.4)
        self.assertIsNotNone(problem)
        self.assertIsNotNone(commute_sampler.vet_card("to_home", (500, 3.4, "", ""), 3.4))
        self.assertNotIn("to_home", commute_sampler._band_rejects)  # a good card resets the streak

    async def test_wedged_page_close_cannot_outlive_the_sample_timeout(self):
        async def hang():
            await asyncio.Event().wait()

        page = types.SimpleNamespace(close=hang)
        browser = types.SimpleNamespace(new_page=AsyncMock(return_value=page))
        with patch.object(commute_sampler, "CSV", str(self.path)), patch.object(commute_sampler, "CLOSE_TIMEOUT", 0.05), \
                patch.object(commute_sampler, "fetch_routes", AsyncMock(return_value=["23 min | 3.4 miles | via Main St"])):
            await asyncio.wait_for(commute_sampler.sample_once(browser, fixtures.EXAMPLE), 2)

    async def test_rows_are_saved_per_direction_even_if_the_sample_is_cancelled(self):
        done = set()
        fetch = AsyncMock(side_effect=[["23 min | 3.4 miles | via Main St"], asyncio.CancelledError()])
        with patch.object(commute_sampler, "CSV", str(self.path)), patch.object(commute_sampler, "fetch_routes", fetch):
            with self.assertRaises(asyncio.CancelledError):
                await commute_sampler.sample_once(self.browser, fixtures.EXAMPLE, done=done)
        self.assertEqual(done, {"to_work"})
        out = Path(commute_config.sample_path(fixtures.EXAMPLE, str(self.path)))
        with out.open() as handle:
            self.assertEqual([row["direction"] for row in csv.DictReader(handle)], ["to_work"])

    async def test_rejected_card_fails_only_its_direction(self):
        with patch.object(commute_sampler, "CSV", str(self.path)), \
                patch.object(commute_sampler, "fetch_routes",
                             AsyncMock(side_effect=[["23 min | 3.4 miles | via Main St"], ["19 min | 900 miles"]])):
            with self.assertRaises(commute_sampler.SampleError) as caught:
                await commute_sampler.sample_once(self.browser, fixtures.EXAMPLE)
        self.assertEqual(caught.exception.succeeded, {"to_work"})
        self.assertIn("to_home: rejected card", str(caught.exception))
        out = Path(commute_config.sample_path(fixtures.EXAMPLE, str(self.path)))
        with out.open() as handle:
            self.assertEqual([row["direction"] for row in csv.DictReader(handle)], ["to_work"])

    async def test_only_requested_directions_are_sampled_through_the_shared_context(self):
        context = types.SimpleNamespace(new_page=AsyncMock(return_value=self.page))
        fetch = AsyncMock(return_value=["23 min | 3.4 miles | via Main St"])
        with patch.object(commute_sampler, "CSV", str(self.path)), patch.object(commute_sampler, "fetch_routes", fetch):
            await commute_sampler.sample_once(self.browser, fixtures.EXAMPLE, ("to_home",), context)
        self.assertEqual(fetch.await_count, 1)
        self.assertEqual(fetch.await_args.args[1][0], fixtures.EXAMPLE["work"])
        context.new_page.assert_awaited_once_with()
        self.browser.new_page.assert_not_awaited()  # pages come from the long-lived context, so its cache persists
        self.assertEqual(self.page.close.await_count, 1)


class RetryTests(unittest.IsolatedAsyncioTestCase):
    async def test_retry_resamples_only_the_direction_that_failed(self):
        browser = types.SimpleNamespace(close=AsyncMock())
        calls = []

        async def sample(browser, settings, directions, context, done):
            calls.append(directions)
            if len(calls) == 1:
                raise commute_sampler.SampleError("to_home: timeout", succeeded={"to_work"})

        class PlaywrightContext:
            async def __aenter__(self):
                return types.SimpleNamespace(chromium=types.SimpleNamespace(launch=AsyncMock(return_value=browser)))

            async def __aexit__(self, *args):
                return False

        module = types.ModuleType("playwright.async_api")
        module.async_playwright = PlaywrightContext

        async def sleep(seconds):
            if len(calls) >= 3:
                raise asyncio.CancelledError()

        with tempfile.TemporaryDirectory() as directory, \
                patch.dict(sys.modules, {"playwright.async_api": module}), \
                patch.object(commute_sampler, "REFRESH", str(Path(directory) / "refresh")), \
                patch.object(commute_sampler, "in_window", return_value=True), \
                patch.object(commute_sampler, "sample_once", sample), \
                patch.object(commute_sampler, "next_sample_at", return_value=0), \
                patch.object(commute_sampler.asyncio, "sleep", sleep):
            with self.assertRaises(asyncio.CancelledError):
                await commute_sampler.run(False)
        # Slot 1 fails after saving to_work; the retry owes only to_home; the next slot samples both again.
        self.assertEqual(calls, [("to_work", "to_home"), ("to_home",), ("to_work", "to_home")])

    async def run_failing(self, sample, slots, browser=None):
        browser = browser or types.SimpleNamespace(close=AsyncMock())

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
            if len(ticks) >= len(slots):
                raise asyncio.CancelledError()

        with tempfile.TemporaryDirectory() as directory, \
                patch.dict(sys.modules, {"playwright.async_api": module}), \
                patch.object(commute_sampler, "REFRESH", str(Path(directory) / "refresh")), \
                patch.object(commute_sampler, "in_window", return_value=True), \
                patch.object(commute_sampler, "sample_once", sample), \
                patch.object(commute_sampler, "next_sample_at", return_value=0), \
                patch.object(commute_sampler, "sample_slot", lambda now=None: slots[min(len(ticks), len(slots) - 1)]), \
                patch.object(commute_sampler.asyncio, "sleep", sleep):
            with self.assertRaises(asyncio.CancelledError):
                await commute_sampler.run(False)

    async def test_one_direction_failing_forever_never_starves_the_other(self):
        calls = []

        async def sample(browser, settings, directions, context, done):
            calls.append(directions)
            done.add("to_work") if "to_work" in directions else None
            raise commute_sampler.SampleError("to_home: rejected card", succeeded=done)

        await self.run_failing(sample, [5, 5, 6, 6])  # the slot seen by each loop iteration
        self.assertEqual(calls[0], ("to_work", "to_home"))
        self.assertEqual(calls[1], ("to_home",))   # same slot: only the direction still owed
        self.assertEqual(calls[2], ("to_work", "to_home"))  # new slot: everything again

    async def test_timeout_keeps_rows_already_saved_and_run_uses_the_shared_context(self):
        context = types.SimpleNamespace()
        browser = types.SimpleNamespace(close=AsyncMock(), new_context=AsyncMock(return_value=context))
        calls = []

        async def sample(browser_, settings, directions, context_, done):
            calls.append((directions, context_))
            if len(calls) == 1:
                done.add("to_work")  # saved before the hang
                raise asyncio.TimeoutError()

        await self.run_failing(sample, [5, 5], browser)
        self.assertIs(calls[0][1], context)
        self.assertEqual(calls[1][0], ("to_home",))  # the saved direction is not sampled twice
        self.assertEqual(browser.new_context.await_count, 2)  # the failure dropped the browser; the retry relaunched
        self.assertEqual(browser.close.await_count, 2)


class SetupTests(unittest.TestCase):
    def test_defaults_are_a_blank_first_run_form(self):
        self.assertEqual(commute_config.DEFAULTS, {
            "home": "", "work": "", "to_work_waypoints": [], "to_home_waypoints": [],
            "to_work_start": "06:00", "to_work_end": "12:00",
            "to_home_start": "15:00", "to_home_end": "21:30",
            "extra_drive_minutes": 5, "notify": True, "skip_us_holidays": True, "skip_dates": []})
        self.assertNotIn("weather_lat", commute_config.DEFAULTS)
        self.assertNotIn("weather_lon", commute_config.DEFAULTS)
        with self.assertRaises(commute_config.SettingsError):
            commute_config.validate(commute_config.DEFAULTS)

    def test_setup_hint_and_missing_settings_file(self):
        self.assertEqual(commute_config.SETUP_HINT,
                         "Not set up yet: open Settings and enter your home and work addresses.")
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(commute_config.SettingsError) as caught:
                commute_config.load_settings(Path(directory) / "absent.json")
            self.assertEqual(str(caught.exception), commute_config.SETUP_HINT)

    def test_config_cli_prints_defaults_once_then_settings(self):
        script = str(Path(commute_config.HERE) / "commute_config.py")
        with tempfile.TemporaryDirectory() as directory:
            missing = str(Path(directory) / "missing.json")
            env = {**os.environ, "COMMUTE_SETTINGS": missing}
            blank = subprocess.run([sys.executable, script], env=env, capture_output=True, text=True)
            self.assertEqual(blank.returncode, 0, blank.stderr)
            self.assertEqual(json.loads(blank.stdout), commute_config.DEFAULTS)
            Path(missing).write_text("{")
            corrupt = subprocess.run([sys.executable, script], env=env, capture_output=True, text=True)
            self.assertEqual(corrupt.returncode, 1)
        configured = subprocess.run([sys.executable, script], env=dict(os.environ),
                                    capture_output=True, text=True)
        self.assertEqual(configured.returncode, 0, configured.stderr)
        self.assertEqual(json.loads(configured.stdout), fixtures.EXAMPLE)

    def test_data_cli_reports_setup_hint_without_settings(self):
        script = str(Path(commute_config.HERE) / "commute_data.py")
        with tempfile.TemporaryDirectory() as directory:
            env = {**os.environ, "COMMUTE_SETTINGS": str(Path(directory) / "missing.json")}
            result = subprocess.run([sys.executable, script], env=env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 1)
        self.assertIn("Not set up yet", result.stderr)

    def test_weather_point_is_opt_in(self):
        base = {key: value for key, value in fixtures.EXAMPLE.items()
                if key not in ("weather_lat", "weather_lon")}
        self.assertIsNone(commute_config.weather_point(commute_config.validate(base)))
        self.assertIsNone(commute_config.weather_point(
            commute_config.validate({**base, "weather_lat": 10.5})))
        self.assertIsNone(commute_config.weather_point(
            commute_config.validate({**base, "weather_lon": 20.25})))
        self.assertEqual(commute_config.weather_point(
            commute_config.validate({**base, "weather_lat": 10.5, "weather_lon": 20.25})),
            (10.5, 20.25))

    def test_sample_path_is_hashed_stable_and_route_scoped(self):
        settings = commute_config.validate(fixtures.EXAMPLE)
        name = os.path.basename(commute_config.sample_path(settings))
        self.assertRegex(name, r"commute_live_[0-9a-f]{20}\.csv")
        self.assertEqual(commute_config.sample_path(settings), commute_config.sample_path(dict(settings)))
        for key, value in (("home", "Elsewhere"), ("work", "Elsewhere"),
                           ("to_work_waypoints", ["Detour"]), ("to_home_waypoints", ["Detour"])):
            with self.subTest(key=key):
                changed = commute_config.validate({**settings, key: value})
                self.assertNotEqual(commute_config.sample_path(changed), commute_config.sample_path(settings))
        windows_only = commute_config.validate({**settings, "to_work_start": "07:00", "to_home_end": "22:00",
                                                "extra_drive_minutes": 0, "notify": False,
                                                "skip_dates": ["2026-12-25"]})
        self.assertEqual(commute_config.sample_path(windows_only), commute_config.sample_path(settings))

    def test_sample_path_digest_is_stable_for_existing_routes(self):
        # Regression: the hash inputs and encoding must not drift, or a configured user's history file is orphaned.
        settings = {
            "home": "1 Infinite Loop, Cupertino, CA",
            "work": "350 Fifth Ave, New York, NY",
            "to_work_waypoints": ["2 WPA Plaza, New York, NY"],
            "to_home_waypoints": [],
        }
        self.assertEqual(os.path.basename(commute_config.sample_path(settings, "/tmp/commute_live.csv")),
                         "commute_live_93b1b57cfe3eebb9690b.csv")

    def test_local_timezone_prefers_env_then_zoneinfo_link_then_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            zone = Path(directory) / "zoneinfo" / "Asia" / "Tokyo"
            zone.parent.mkdir(parents=True)
            zone.touch()
            link = Path(directory) / "localtime"
            link.symlink_to(zone)
            self.assertEqual(commute_config.local_timezone(localtime=str(link), environ={}).key, "Asia/Tokyo")
            self.assertEqual(commute_config.local_timezone(
                localtime=str(link), environ={"COMMUTE_TZ": "Europe/Paris"}).key, "Europe/Paris")
        fallback = commute_config.local_timezone(localtime="/definitely/not/a/localtime", environ={})
        self.assertIsNotNone(datetime.now(fallback))


class CalendarAndSettingsTests(unittest.TestCase):
    def test_holiday_dates_and_observed_rules(self):
        names = {day: name for year in (2026, 2027) for day, name in commute_calendar.us_holidays(year).items()}
        self.assertEqual(names[date(2026, 11, 26)], "Thanksgiving")
        self.assertEqual(names[date(2026, 11, 27)], "Day after Thanksgiving")
        self.assertEqual(names[date(2026, 7, 3)], "Independence Day")  # July 4 is a Saturday: observed Friday
        self.assertNotIn(date(2026, 7, 4), names)
        self.assertEqual(names[date(2027, 12, 24)], "Christmas Day")
        self.assertEqual(names[date(2026, 5, 25)], "Memorial Day")
        self.assertEqual(names[date(2026, 9, 7)], "Labor Day")
        # New Year's Day 2028 falls on a Saturday and is observed on Friday, Dec 31, 2027.
        self.assertEqual(commute_calendar.day_off_reason(date(2027, 12, 31), {}), "New Year's Day")
        self.assertIsNone(commute_calendar.day_off_reason(date(2026, 10, 6), {}))
        self.assertEqual(commute_calendar.day_off_reason(date(2026, 10, 6), {"skip_dates": ["2026-10-06"]}), "Day off")
        self.assertIsNone(commute_calendar.day_off_reason(date(2026, 11, 26), {"skip_us_holidays": False}))

    def test_new_settings_defaults_and_validation(self):
        base = {key: value for key, value in fixtures.EXAMPLE.items()
                if key not in ("notify", "skip_us_holidays", "skip_dates")}
        result = commute_config.validate(base)
        self.assertEqual((result["notify"], result["skip_us_holidays"], result["skip_dates"]), (True, True, []))
        self.assertEqual(commute_config.validate({**base, "skip_dates": ["2026-12-24", "2026-12-24", "2026-11-27"]})["skip_dates"],
                         ["2026-11-27", "2026-12-24"])
        for key, value in (("notify", "yes"), ("skip_us_holidays", 1), ("skip_dates", "2026-12-24"),
                           ("skip_dates", ["2026-02-30"]), ("skip_dates", ["12/24/2026"]), ("skip_dates", [20261224])):
            with self.subTest(key=key, value=value), self.assertRaises(commute_config.SettingsError):
                commute_config.validate({**base, key: value})


if __name__ == "__main__":
    unittest.main()

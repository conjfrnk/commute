"""Shared test fixture: a fictional Springfield route, pinned before commute_config is imported.

Import this before any commute_* module: it sets COMMUTE_TZ and COMMUTE_SETTINGS while they are
still unread, so the suite never depends on the host timezone or a real settings file.
"""
import json
import os
import tempfile

os.environ["COMMUTE_TZ"] = "America/Los_Angeles"

EXAMPLE = {
    "home": "10 Main St, Springfield, IL 62701",
    "work": "200 Capitol Ave, Springfield, IL 62701",
    "weather_lat": 39.7817,
    "weather_lon": -89.6501,
    "to_work_waypoints": ["5 Elm St, Springfield, IL", "1 Oak St, Springfield, IL"],
    "to_home_waypoints": ["3 Birch St, Springfield, IL", "7 Maple St, Springfield, IL",
                          "9 Cedar St, Springfield, IL"],
    "to_work_start": "06:00",
    "to_work_end": "12:00",
    "to_home_start": "15:00",
    "to_home_end": "21:30",
    "extra_drive_minutes": 5,
    "notify": True,
    "skip_us_holidays": True,
    "skip_dates": [],
}

_SETTINGS_PATH = os.path.join(tempfile.mkdtemp(), "settings.json")
with open(_SETTINGS_PATH, "w", encoding="utf-8") as handle:
    json.dump(EXAMPLE, handle)
os.environ["COMMUTE_SETTINGS"] = _SETTINGS_PATH

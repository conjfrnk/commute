"""Shared, validated settings and route-specific sample storage for Commute Bar."""
import hashlib
import json
import math
import os
import re
from datetime import date, datetime
from urllib.parse import quote_plus
from zoneinfo import ZoneInfo

HERE = os.path.dirname(os.path.abspath(__file__))
# COMMUTE_SETTINGS lets tests (and other checkouts) use a settings file other than the app's commute_settings.json.
CONFIG = os.environ.get("COMMUTE_SETTINGS") or os.path.join(HERE, "commute_settings.json")
CSV = os.path.join(HERE, "commute_live.csv")
# First-run defaults: a blank form for the Settings window; validate() rejects them until addresses are entered.
DEFAULTS = {
    "home": "",
    "work": "",
    "to_work_waypoints": [],
    "to_home_waypoints": [],
    "to_work_start": "06:00",
    "to_work_end": "12:00",
    "to_home_start": "15:00",
    "to_home_end": "21:30",
    "extra_drive_minutes": 5,
    "notify": True,
    "skip_us_holidays": True,
    "skip_dates": [],
}
SETUP_HINT = "Not set up yet: open Settings and enter your home and work addresses."


def local_timezone(localtime="/etc/localtime", environ=os.environ):
    """The zone 'local' means here: $COMMUTE_TZ, else the zone the localtime link points into, else the host offset."""
    name = environ.get("COMMUTE_TZ") or ""
    if not name:
        real = os.path.realpath(localtime)
        if "zoneinfo/" in real:
            name = real.split("zoneinfo/", 1)[1]
    try:
        if name:
            return ZoneInfo(name)
    except (ValueError, OSError, KeyError):  # ZoneInfoNotFoundError is a KeyError; a zone that is a directory raises IsADirectoryError
        pass
    return datetime.now().astimezone().tzinfo


TZ = local_timezone()


class SettingsError(ValueError):
    pass


def validate(settings):
    if not isinstance(settings, dict):
        raise SettingsError("Settings must be a JSON object")
    result = {}
    for key in ("home", "work"):
        value = settings.get(key)
        if not isinstance(value, str) or not value.strip():
            raise SettingsError(f"{key} must be a nonempty address")
        result[key] = value.strip()
    for key in ("to_work_waypoints", "to_home_waypoints"):
        value = settings.get(key)
        if not isinstance(value, list) or any(not isinstance(p, str) or not p.strip() for p in value):
            raise SettingsError(f"{key} must be a list of nonempty addresses")
        result[key] = [p.strip() for p in value]
    for direction in ("to_work", "to_home"):
        for suffix in ("start", "end"):
            key = f"{direction}_{suffix}"
            value = settings.get(key)
            if not isinstance(value, str) or not re.fullmatch(r"(?:[01]\d|2[0-3]):[0-5]\d", value):
                raise SettingsError(f"{key} must be HH:mm")
            result[key] = value
        if result[f"{direction}_start"] >= result[f"{direction}_end"]:
            raise SettingsError(f"{direction} start must be before end on the same day")
    extra = settings.get("extra_drive_minutes", 5)
    if (isinstance(extra, bool) or not isinstance(extra, (int, float)) or
            not 0 <= extra <= 30):  # also rejects nan, inf, and ints too big for float
        raise SettingsError("extra_drive_minutes must be a number from 0 to 30")
    result["extra_drive_minutes"] = extra
    for key in ("notify", "skip_us_holidays"):
        value = settings.get(key, True)
        if not isinstance(value, bool):
            raise SettingsError(f"{key} must be true or false")
        result[key] = value
    skip = settings.get("skip_dates", [])
    if not isinstance(skip, list):
        raise SettingsError("skip_dates must be a list of YYYY-MM-DD dates")
    dates = set()
    for value in skip:
        try:
            if not isinstance(value, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
                raise ValueError
            date.fromisoformat(value)
        except ValueError:
            raise SettingsError(f"skip_dates entry {value!r} must be a real YYYY-MM-DD date") from None
        dates.add(value)
    result["skip_dates"] = sorted(dates)
    for key in ("weather_lat", "weather_lon"):
        value = settings.get(key)
        if value is None:
            continue
        if (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
                or (key == "weather_lat" and not -90 <= value <= 90)
                or (key == "weather_lon" and not -180 <= value <= 180)):
            raise SettingsError(f"{key} must be a valid coordinate")
        result[key] = float(value)
    return result


def load_settings(path=None):
    try:
        with open(path or CONFIG, encoding="utf-8") as handle:
            raw = json.load(handle)
    except FileNotFoundError:
        raise SettingsError(SETUP_HINT) from None
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SettingsError(f"Cannot read settings: {exc}") from exc
    return validate(raw)


def weather_point(settings):
    """Opt-in rain covariate: (lat, lon) only when the settings carry both coordinates; None disables the fetch."""
    lat = settings.get("weather_lat")
    lon = settings.get("weather_lon")
    if isinstance(lat, (int, float)) and isinstance(lon, (int, float)):
        return float(lat), float(lon)
    return None


def gmaps_url(points):
    enc = "/".join(quote_plus(p, safe=",") for p in points)
    return f"https://www.google.com/maps/dir/{enc}/data=!4m2!4m1!3e0"  # 3e0 = driving, no depart-at = leave now


def route_points(settings, direction):
    origin, destination = ("home", "work") if direction == "to_work" else ("work", "home")
    return [settings[origin], *settings[f"{direction}_waypoints"], settings[destination]]


def window(settings, direction):
    return tuple(tuple(map(int, settings[f"{direction}_{suffix}"].split(":"))) for suffix in ("start", "end"))


def sample_path(settings, default_path=None):
    default_path = default_path or CSV
    keys = ("home", "work", "to_work_waypoints", "to_home_waypoints")
    identity = {key: settings[key] for key in keys}
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()[:20]
    stem, extension = os.path.splitext(default_path)
    return f"{stem}_{digest}{extension}"


if __name__ == "__main__":
    import sys
    try:
        if os.path.exists(CONFIG):
            print(json.dumps(load_settings(), ensure_ascii=False))
        else:
            print(json.dumps(DEFAULTS))  # first run: the Settings window starts from a blank form
    except SettingsError as exc:
        print(f"Settings error: {exc}", file=sys.stderr)
        sys.exit(1)

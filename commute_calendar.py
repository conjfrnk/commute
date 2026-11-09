"""Days off: US holidays and user-listed dates, excluded from history and suggestions."""
from datetime import date, timedelta


def _nth_weekday(year, month, weekday, n):
    first = date(year, month, 1)
    return first + timedelta(days=(weekday - first.weekday()) % 7 + 7 * (n - 1))


def _last_weekday(year, month, weekday):
    last = date(year + (month == 12), month % 12 + 1, 1) - timedelta(days=1)
    return last - timedelta(days=(last.weekday() - weekday) % 7)


def _observed(day):
    # Federal rule: a Saturday holiday is observed Friday, a Sunday holiday Monday.
    return day - timedelta(days=1) if day.weekday() == 5 else day + timedelta(days=1) if day.weekday() == 6 else day


def us_holidays(year):
    """{date: name} of federal holidays as observed, plus the day after Thanksgiving."""
    thanksgiving = _nth_weekday(year, 11, 3, 4)
    fixed = {"New Year's Day": date(year, 1, 1), "Juneteenth": date(year, 6, 19),
             "Independence Day": date(year, 7, 4), "Veterans Day": date(year, 11, 11),
             "Christmas Day": date(year, 12, 25)}
    floating = {"Martin Luther King Jr. Day": _nth_weekday(year, 1, 0, 3),
                "Presidents' Day": _nth_weekday(year, 2, 0, 3),
                "Memorial Day": _last_weekday(year, 5, 0),
                "Labor Day": _nth_weekday(year, 9, 0, 1),
                "Columbus Day": _nth_weekday(year, 10, 0, 2),
                "Thanksgiving": thanksgiving,
                "Day after Thanksgiving": thanksgiving + timedelta(days=1)}
    result = {_observed(day): name for name, day in fixed.items()}
    result.update({day: name for name, day in floating.items()})
    return result


def day_off_reason(day, settings):
    """Why `day` is a day off under `settings`, or None. Weekends are handled separately."""
    if day.isoformat() in settings.get("skip_dates", ()):
        return "Day off"
    if settings.get("skip_us_holidays", True):
        # A new year's holiday can be observed on Dec 31 of the previous year.
        for year in (day.year, day.year + 1):
            name = us_holidays(year).get(day)
            if name:
                return name
    return None


def is_off_day(day, settings):
    """A weekend or a day off under `settings`: traffic on these days follows its own pattern."""
    return day.weekday() >= 5 or bool(day_off_reason(day, settings))

#!/usr/bin/env python3
"""The self-training "Predicted" drive-time model: a probabilistic forecast with a calibrated 80% interval.

How it works (full argument and evidence: docs/MODEL.md):

    drive time = smooth time-of-day curve  +  persistent day effect  +  within-day mean-reverting wiggle  +  noise

  * commute_statespace.py learns the curve and its noise from your own history and tracks today's day effect
    with a Kalman filter. That filter replaces the hand-set "how much of today's deviation carries forward"
    weights: the weights are now learned and fade with horizon automatically.
  * Its hyper-parameters are re-fitted nightly from PRIOR days only (leave-one-day-out likelihood).
  * An online recalibration layer watches how often real drive times fell inside the intervals it issued
    (the probability-integral transform of each realized value) and bends future intervals to match.
  * A weekday effect switches itself on once every weekday has enough days to estimate it.
  * Every night the finished day is first SCORED by the model that did not know it, then folded in. Those
    scores form the ledger, which drives the honest label ("Collecting data" -> "Typical (learning)" ->
    "Predicted (calibrating)" -> "Predicted"), the accuracy readout, and an automatic fallback to the simple
    typical curve if this model ever scores clearly worse than it.

Training runs in a subprocess started by the sampler after each day's last window (`python commute_model.py
--sync`); polling (commute_data.snapshot) only reads the small state file.
"""
import argparse
import csv
import fcntl
import hashlib
import json
import math
import os
import sys
import tempfile
import urllib.request
from collections import defaultdict
from contextlib import contextmanager
from datetime import date, datetime, timedelta

import commute_statespace as S
from commute_calendar import is_off_day
from commute_config import CSV, HERE, TZ, SettingsError, load_settings, sample_path, weather_point, window

CODE_VERSION = 4   # 4: learned day-effect covariates (sibling direction, own yesterday, rain)
SCHEMA = 1
LOCK = os.path.join(HERE, ".commute-model.lock")
GRID = S.GRID
QKEYS = ("q05", "q10", "q25", "q50", "q75", "q90", "q95")
QP = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)
MATCH_TOL = 5            # minutes between a target time and the sample scored against it
ORIGIN_STEP = 30         # forecast origins (minutes) used when scoring a finished day
MORNING_OF_LEAD = 60     # the "no information from today" origin sits this long before the window opens
MIN_REPLAY_HIST = 3      # days of history before the first day can be scored
MIN_SCORED_FRACTION = 0.3  # a day needs this share of its window's samples to be scored (the Mac may have slept through it)
SYNC_DELAY = 70          # minutes after the last window closes before the day is folded in (the engine reads ~66 past it)
WEEKDAY_MIN_DAYS = 2     # every weekday needs this many days before the weekday effect is switched on
OFF = "@off"             # stream-key suffix for days off: weekends, holidays and listed dates have their own curve
MAX_MINUTES = 24 * 60
LEDGER_KEEP = 400

DEFAULT = {
    "scale": "add",           # engine scale law
    "student": False,         # Gaussian head (the recalibration layer owns the tails)
    "half_life_days": 60.0,   # engine forgetting, in weekdays
    "n_bins": 40,
    "n0": 8.0,                # pseudo-days of "the model is right" mass in the recalibration histogram
    "decay": 0.97,            # per scored day (half-life ~23 days)
    "cal_min_hist": 6,        # only learn from forecasts issued by an engine that had this many history days
    "groups": 3,              # 0 morning-of, 1 <= 60 min ahead, 2 > 60 min ahead
    "replay_days": 60,        # a rebuild replays at most this many most-recent days to rebuild the ledger
    "covariates": True,       # learned day-effect covariates (sibling direction, own yesterday, rain)
    "cov_min_days": 10,       # history days needed before any covariate may carry weight
    "cov_prior_sd": 0.35,     # ridge shrinkage on the sibling / previous-day coefficients
    "cov_rain_sd": 0.15,      # ridge shrinkage on the rain coefficient (per mm, standardised units)
    "cov_shrink_days": 10.0,  # pseudo-days pulling the post-covariate innovation variance back to plain
    "gate_calibrating_days": 5, "gate_predicted_days": 15, "gate_cov_lo": 0.70, "gate_cov_hi": 0.90,
    "gate_last_k": 5, "gate_last_lo": 0.60, "gate_last_hi": 0.97,
    "gate_close_lo": 0.65, "gate_close_hi": 0.93, "gate_close_last_lo": 0.55, "gate_close_last_hi": 0.98,
    "fb_window": 7, "fb_ratio": 1.25, "fb_release": 1.10, "fb_min_days": 5,
}


def stream_key(direction, off):
    """Name of the model stream for a direction on a normal day or on a day off."""
    return direction + OFF if off else direction


def sibling_stream(direction):
    """The other direction's stream of the same day kind: to_home learns from to_work's same-day traffic
    and to_work from to_home's previous day; the day-off streams pair up the same way."""
    base = direction[:-len(OFF)] if direction.endswith(OFF) else direction
    suffix = OFF if direction.endswith(OFF) else ""
    other = "to_work" if base == "to_home" else "to_home"
    return other + suffix


def windows_of(settings):
    """{stream: (start_minute, end_minute)} from validated settings. Days off use the same windows as other days."""
    base = {d: tuple(h * 60 + m for h, m in window(settings, d)) for d in ("to_work", "to_home")}
    return {**base, **{stream_key(d, True): w for d, w in base.items()}}


def skip_key(settings):
    """Days-off settings; changing them changes which days are history, so the model is rebuilt."""
    raw = json.dumps([settings.get("skip_us_holidays", True), sorted(settings.get("skip_dates", []))])
    return hashlib.sha256(raw.encode()).hexdigest()[:12]


def group_of(windows, direction, origin, target, n_groups=3):
    if n_groups == 1:
        return 0
    if origin is None or origin < windows[direction][0]:
        return 0
    if n_groups == 2:
        return 1
    return 1 if target - origin <= 60 else 2


def _match(series, target):
    best = None
    for m, v in series:
        d = abs(m - target)
        if d <= MATCH_TOL and (best is None or (d, m) < (best[0], best[1][0])):
            best = (d, (m, v))
    return None if best is None else best[1]


# --------------------------------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------------------------------
def load_clean_rows(path, settings):
    """Valid rows from the sample CSV; a repeated (date, time, direction) keeps the newest. Rows on days off
    (weekends, holidays) are kept under their own stream key (direction + OFF) so they never shape the weekday curve."""
    newest = {}
    try:
        with open(path, newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            if not {"date", "time", "direction", "minutes"}.issubset(reader.fieldnames or []):
                return []
            for raw in reader:
                try:
                    day = date.fromisoformat(raw["date"].strip())
                    hh, mm = raw["time"].strip().split(":")[:2]
                    minute = int(hh) * 60 + int(mm)
                    direction = raw["direction"].strip()
                    value = float(raw["minutes"])
                    if (direction not in ("to_work", "to_home") or not 0 <= minute < 1440
                            or not math.isfinite(value) or not 0 < value <= MAX_MINUTES):
                        raise ValueError("invalid")
                except (ValueError, TypeError, AttributeError, KeyError):
                    continue  # also a partially appended final line
                direction = stream_key(direction, is_off_day(day, settings))
                newest[(day.isoformat(), minute, direction)] = {
                    "date": day.isoformat(), "minute": minute, "weekday": day.weekday(),
                    "direction": direction, "minutes": value}
    except (FileNotFoundError, OSError, UnicodeError, csv.Error):
        return []
    return [newest[k] for k in sorted(newest)]


def latest_complete_day(now, windows):
    """The most recent day (weekends and days off included) whose last window has closed, plus the engine's look-past margin."""
    last_end = max(end for _, end in windows.values())
    day = now.date()
    if now.hour * 60 + now.minute < last_end + SYNC_DELAY:
        day -= timedelta(days=1)
    return day


# --------------------------------------------------------------------------------------------------
# Recalibration layer
# --------------------------------------------------------------------------------------------------
class PITCalibrator:
    """Per (direction, group): decayed day-unit histogram of jittered PIT values plus n0 uniform pseudo-days.

    The PIT value u = F_model(y) of every realized drive time is histogrammed. If the model's intervals are right
    the histogram is flat and nothing changes; if reality lands in the tails too often (or never) the final
    quantile at level p is the model's quantile at level Q_PIT(p), where Q_PIT is the histogram's quantile
    function. A day counts once however many forecasts it contributes; old days fade; with no data the map is the
    identity (Kuleshov et al. 2018, recalibration of regression forecasts).
    """

    def __init__(self, n_bins=40, n0=8.0, decay=0.97, n_groups=3, directions=("to_work", "to_home")):
        self.nb, self.n0, self.decay, self.ng = n_bins, n0, decay, n_groups
        self.h = {d: [[0.0] * n_bins for _ in range(n_groups)] for d in directions}
        self.m = {d: [0.0] * n_groups for d in directions}
        self.days_seen = {d: [0] * n_groups for d in directions}
        self._levels = {}

    def add_day(self, direction, cells):
        for g in range(self.ng):
            self.h[direction][g] = [v * self.decay for v in self.h[direction][g]]
            self.m[direction][g] *= self.decay
        for g, ivs in cells.items():
            if not ivs:
                continue
            w = 1.0 / len(ivs)
            for lo, hi in ivs:
                self._spread(self.h[direction][g], lo, hi, w)
            self.m[direction][g] += 1.0
            self.days_seen[direction][g] += 1
        self._levels = {k: v for k, v in self._levels.items() if k[0] != direction}

    def _spread(self, hist, lo, hi, w):
        nb = self.nb
        lo, hi = min(max(lo, 0.0), 1.0), min(max(hi, 0.0), 1.0)
        if hi - lo < 1e-9:
            hist[min(nb - 1, int(lo * nb))] += w
            return
        a, b = lo * nb, hi * nb
        for i in range(int(a), min(nb - 1, int(b)) + 1):
            ov = min(b, i + 1) - max(a, i)
            if ov > 0:
                hist[i] += w * ov / (b - a)

    def levels(self, direction, g):
        """Mapped probability levels for QP."""
        key = (direction, g)
        if key in self._levels:
            return self._levels[key]
        hist, m, nb = self.h[direction][g], self.m[direction][g], self.nb
        if m <= 1e-9:
            lv = list(QP)
        else:
            tot, pr = m + self.n0, self.n0 / nb
            cdf, c = [0.0], 0.0
            for v in hist:
                c += (v + pr) / tot
                cdf.append(c)
            lv = []
            for p in QP:
                j = 0
                while j < nb - 1 and cdf[j + 1] < p:
                    j += 1
                span = cdf[j + 1] - cdf[j]
                lv.append((j + ((p - cdf[j]) / span if span > 1e-12 else 0.5)) / nb)
            lv = [min(0.9995, max(0.0005, x)) for x in lv]
            for i in range(1, len(lv)):
                lv[i] = max(lv[i], lv[i - 1] + 1e-6)
        self._levels[key] = lv
        return lv

    def to_state(self):
        r = lambda xs: [round(v, 5) for v in xs]
        return {"n_bins": self.nb, "n0": self.n0, "decay": self.decay, "n_groups": self.ng,
                "h": {d: [r(x) for x in self.h[d]] for d in self.h}, "m": {d: r(self.m[d]) for d in self.m},
                "days_seen": self.days_seen}

    @classmethod
    def from_state(cls, st):
        c = cls(st["n_bins"], st["n0"], st["decay"], st["n_groups"], tuple(st["h"]))
        c.h = {d: [list(x) for x in v] for d, v in st["h"].items()}
        c.m = {d: list(v) for d, v in st["m"].items()}
        c.days_seen = {d: list(v) for d, v in st["days_seen"].items()}
        return c


# --------------------------------------------------------------------------------------------------
# The simple model the champion must keep beating
# --------------------------------------------------------------------------------------------------
class Comparator:
    """Typical curve (30-day kernel mean) + the old live adjustment, with its empirical 80% band.

    Used only for scoring: if the champion's recent interval scores are clearly worse than this, the app falls
    back to showing this model's output. It is always rebuilt from code, never from persisted state.
    """

    def __init__(self, windows):
        self.windows = windows

    def fit(self, rows, day):
        import commute_data as cd
        self.cd = cd
        cutoff = date.fromisoformat(day) - timedelta(days=cd.HISTORY_DAYS)
        self.points = {}
        for direction, (start, end) in self.windows.items():
            samples = [(r["minute"], r["minutes"], date.fromisoformat(r["date"])) for r in rows
                       if r["direction"] == direction and cutoff <= date.fromisoformat(r["date"]) < date.fromisoformat(day)
                       and start <= r["minute"] <= end]
            # the model is scored at every GRID step from the window start, which is not the typical curve's
            # clock-aligned grid when the window starts off a multiple of 10: the comparator must answer the same targets
            grid = sorted(set(cd.grid_minutes(start, end)) | set(range(start, end + 1, GRID)))
            self.points[direction] = cd.typical_curve(samples, grid)
        return self

    def predict(self, direction, origin, observed, targets):
        cd = self.cd
        start, end = self.windows[direction]
        points = {p["minute_of_day"]: p for p in self.points[direction]}
        today = [(m, v) for m, v in observed if start <= m <= end]
        offset = cd.live_offset(self.points[direction], today, origin) if today and origin >= start else None
        out = []
        for t in targets:
            p = points.get(t)
            if p is None:
                out.append(None)
                continue
            mean = p["typical_minutes"] if offset is None else cd.adjusted(p["typical_minutes"], offset, t - origin)
            out.append({"mean": mean, "q10": p["p10_minutes"], "q90": p["p90_minutes"]})
        return out


# --------------------------------------------------------------------------------------------------
# Raw (mean, sd) from an engine direction fit
# --------------------------------------------------------------------------------------------------
def raw_predict(f, origin, observed, targets, weekday=None, cov=None):
    th = f.theta
    if f.m is None or th is None:
        return [None] * len(targets)
    same = f.wd_deltas.get(weekday) if (th.phi > 0.0 and weekday is not None) else None
    prior = f.day_prior(th, f.n_eff_days, same, cov)
    series = []
    for t, y in observed:
        if f.lo <= t <= min(f.hi, origin):
            s = f._scale_at(f.m, t, f.gamma)
            series.append((t, (y - f._m_at(f.m, t)) / s, s))
    states = f.filter_states(series, th, prior)
    st, last = (states[-1][1], states[-1][0]) if states else (None, None)
    return [f.predict_target(f.m, f.neff_slot, f.gamma, th, prior, st, last, t) for t in targets]


def _ks(cal, kcache, direction, g):
    key = (direction, g)
    ks = kcache.get(key)
    if ks is None:
        ks = [S._ND.inv_cdf(p) for p in cal.levels(direction, g)]
        kcache[key] = ks
    return ks


def _quantiles(windows, cal, kcache, n_groups, direction, origin, observed, targets, raw):
    o_eff = None if (not observed and origin < windows[direction][0]) else origin
    out = []
    for t, mv in zip(targets, raw):
        if mv is None:
            out.append(None)
            continue
        mu, sd = mv
        ks = _ks(cal, kcache, direction, group_of(windows, direction, o_eff, t, n_groups))
        q = {k: mu + sd * kk for k, kk in zip(QKEYS, ks)}
        q["mean"] = mu
        out.append(q)
    return out


# --------------------------------------------------------------------------------------------------
# Learned day-effect covariates
#
# The weekday-pooled prior on the day effect knows only the weekday. Three covariates add what the
# pooled mean cannot: the sibling direction's deviation (to_home <- today's to_work; to_work <-
# yesterday's to_home), the direction's own previous-day residual (day effects run in streaks), and
# the day's precipitation. Coefficients are ridge-shrunk and refit nightly; an empty cov dict makes
# every term a no-op, so a thin or newly built state degrades to the plain model.
# --------------------------------------------------------------------------------------------------
def _wd_pooled_mean(f, target_wd, upto, wts):
    """The weekday pooled day-effect mean for `target_wd` from same-weekday days strictly before `upto`
    (the convention a forecast issued on `upto` would use)."""
    th = f.theta
    same = [(f.delta_std[d], wts.get(d, 0.0)) for d in f.days
            if d < upto and f.wd[d] == target_wd and d in f.delta_std]
    if th.phi <= 0.0 or not same:
        return 0.0
    return f.day_prior(th, f.n_eff_days, same)[0]


def covariate_term(dirs, weather, direction, day, origin, observed_sibling):
    """{"mean","var","ve2"} for `direction`'s day prior on `day` (ISO date) from the learned covariates.

    to_home-family streams use the sibling direction's same-day deviation, estimated from its own
    observations up to `origin` (or its prior when it has none: zero mean shift, its full prior variance).
    to_work-family streams use the sibling's most recent earlier day's residual. Both add the direction's
    own previous-day residual (gaps over 3 days are too stale to use) and the day's rain, if known."""
    out = {"mean": 0.0, "var": 0.0, "ve2": None}
    f = dirs.get(direction)
    cov = getattr(f, "cov", None) or {}
    if f is None or f.m is None or not cov or not day:
        out["ve2"] = cov.get("ve2")
        return out
    out["ve2"] = cov.get("ve2")
    base = direction[:-len(OFF)] if direction.endswith(OFF) else direction
    d0 = date.fromisoformat(day)
    wd = d0.weekday()
    fs = dirs.get(sibling_stream(direction))
    b = cov.get("sib", 0.0)
    if b and fs is not None and fs.m is not None and fs.theta is not None:
        if base == "to_home":
            scov = covariate_term(dirs, weather, sibling_stream(direction), day, origin, None)
            same = fs.wd_deltas.get(wd) if fs.theta.phi > 0.0 else None
            prior_mu, prior_var = fs.day_prior(fs.theta, fs.n_eff_days, same, scov)
            if observed_sibling is None:
                out["var"] += b * b * prior_var   # sibling not observed: no mean shift, honest extra width
            else:
                mu, pv = fs.delta_posterior(observed_sibling, origin, wd, scov)
                out["mean"] += b * (mu - prior_mu)
                out["var"] += b * b * pv
        else:
            past = [d2 for d2 in fs.r_hist if d2 < day and (d0 - date.fromisoformat(d2)).days <= 3]
            if past:
                out["mean"] += b * fs.r_hist[max(past)]
    b = cov.get("prev", 0.0)
    if b:
        past = [d2 for d2 in f.r_hist if d2 < day and (d0 - date.fromisoformat(d2)).days <= 3]
        if past:
            out["mean"] += b * f.r_hist[max(past)]
    b = cov.get("rain", 0.0)
    if b:
        w = weather.get(day)
        if w is not None:
            out["mean"] += b * (w["mm"] - cov.get("rain_mu", 0.0))
    return out


# --------------------------------------------------------------------------------------------------
# Status, label and readout (pure functions of the ledger so poll time needs no model object)
# --------------------------------------------------------------------------------------------------
def _mean(xs):
    return sum(xs) / len(xs)


def update_flags(opt, ledger, flags, direction):
    """Advance the hysteresis flags (fallback active, label gate open) for one direction after a scoring day."""
    rows = [r for r in ledger if r["direction"] == direction]
    fb = dict(flags.get("fallback", {}))
    gate = dict(flags.get("gate", {}))
    w = rows[-opt["fb_window"]:]
    if len(w) >= opt["fb_min_days"] and all(r.get("cmp_is80") is not None for r in w):
        theirs = _mean([r["cmp_is80"] for r in w])
        ratio = _mean([r["is80"] for r in w]) / theirs if theirs > 0 else 1.0
        if fb.get(direction):
            fb[direction] = ratio >= opt["fb_release"]
        else:
            fb[direction] = ratio > opt["fb_ratio"]
    n = len(rows)
    if n >= opt["gate_predicted_days"]:
        cov = [r["cov80"] for r in rows[-20:]]
        m = _mean(cov)
        se = (sum((c - m) ** 2 for c in cov) / (len(cov) - 1)) ** 0.5 / math.sqrt(len(cov))
        lm = _mean([r["cov80"] for r in rows[-opt["gate_last_k"]:]])
        if gate.get(direction):   # once open, only clearly broken calibration closes it again
            gate[direction] = (opt["gate_close_lo"] <= m <= opt["gate_close_hi"]
                               and opt["gate_close_last_lo"] <= lm <= opt["gate_close_last_hi"])
        else:
            gate[direction] = (m - 1.96 * se <= 0.80 <= m + 1.96 * se and opt["gate_cov_lo"] <= m <= opt["gate_cov_hi"]
                               and opt["gate_last_lo"] <= lm <= opt["gate_last_hi"])
    else:
        gate[direction] = False
    return {"fallback": fb, "gate": gate}


def compute_status(opt, ledger, flags, direction, n_hist):
    """{state, label, days (history), ledger_days, readout} for the UI."""
    rows = [r for r in ledger if r["direction"] == direction]
    n = len(rows)
    out = {"days": n_hist, "ledger_days": n, "readout": None}
    if rows:
        last = rows[-20:]
        out["readout"] = {"mae_minutes": round(_mean([r["mae"] for r in last]), 2),
                          "hit_rate": round(_mean([r["cov80"] for r in last]), 3), "days": len(last)}
    if n_hist < MIN_REPLAY_HIST:
        out.update(state="collecting", label="Collecting data")
    elif flags.get("fallback", {}).get(direction):
        out.update(state="fallback", label="Typical (fallback)")
    elif n < opt["gate_calibrating_days"]:
        out.update(state="learning", label="Typical (learning)")
    elif n >= opt["gate_predicted_days"] and flags.get("gate", {}).get(direction):
        out.update(state="ready", label="Predicted")
    else:
        out.update(state="calibrating", label="Predicted (calibrating)")
    return out


# --------------------------------------------------------------------------------------------------
# The model
# --------------------------------------------------------------------------------------------------
class FinalModel(S.StateSpacePredictor):
    name = "predicted"

    def __init__(self, windows, **overrides):
        o = dict(DEFAULT)
        o.update(overrides)
        eng = {k: o[k] for k in ("scale", "half_life_days", "student") if k in o}
        eng["windows"] = {d: tuple(v) for d, v in windows.items()}
        super().__init__(**eng)
        self.windows = self.cfg["windows"]
        self.opt = o
        self.cal = PITCalibrator(o["n_bins"], o["n0"], o["decay"], o["groups"], tuple(self.windows))
        self.ledger = []
        self.flags = {"fallback": {}, "gate": {}}
        self._kcache = {}
        self.weather = {}     # date -> {"mm": precip, "src": "obs"|"fcst"}; filled by the nightly sync

    # ---------------- training ----------------
    def _weekday_ready(self):
        """True once every weekday has WEEKDAY_MIN_DAYS days of history; the engine then estimates how much of
        the day-to-day variation is a weekday effect (it shrinks to ~nothing if there is none)."""
        counts = defaultdict(set)
        for r in self._rows:
            if not r["direction"].endswith(OFF):   # the weekday pattern is a property of the normal-day streams
                counts[r["weekday"]].add(r["date"])
        return all(len(counts[w]) >= WEEKDAY_MIN_DAYS for w in range(5))

    def _refit(self, warm):
        self.cfg["weekday"] = self._weekday_ready()
        self._fit_all(warm=warm)
        self._fit_covariates()
        self._kcache = {}

    def _fit_covariates(self):
        """Ridge-fit the day-effect covariates on each stream's residual day effects (after the weekday
        mean), refit nightly like theta. Pass 1 computes residuals r_d = delta_d - a_wd_d with the
        predict-time convention (only same-weekday days before d); pass 2 regresses r on the sibling
        residual, the previous-day residual, and rain, each shrunk toward zero. The leftover variance
        becomes ve2, the innovation variance the prior uses."""
        for f in self.dirs.values():
            f.cov, f.r_hist = {}, {}
        if not self.opt.get("covariates"):
            return
        r_hists = {}
        for direction, f in self.dirs.items():
            if f.m is None or f.theta is None or len(f.days) < self.opt["cov_min_days"]:
                continue
            wts = f.day_weights()
            r = {d: f.delta_std[d] - _wd_pooled_mean(f, f.wd[d], d, wts)
                 for d in f.days[-150:] if d in f.delta_std}
            if r:
                r_hists[direction] = r
        for direction, r in r_hists.items():
            f = self.dirs[direction]
            base = direction[:-len(OFF)] if direction.endswith(OFF) else direction
            sib_r = r_hists.get(sibling_stream(direction), {})
            ve = (1.0 - f.theta.phi) * f.theta.tau ** 2
            days = sorted(r)
            xs = {"sib": {}, "prev": {}, "rain": {}}
            for d in days:
                d0 = date.fromisoformat(d)
                if base == "to_home":
                    if d in sib_r:
                        xs["sib"][d] = sib_r[d]
                else:
                    past = [d2 for d2 in sib_r if d2 < d and (d0 - date.fromisoformat(d2)).days <= 3]
                    if past:
                        xs["sib"][d] = sib_r[max(past)]
                past = [d2 for d2 in r if d2 < d and (d0 - date.fromisoformat(d2)).days <= 3]
                if past:
                    xs["prev"][d] = r[max(past)]
                w = self.weather.get(d)
                if w is not None and w.get("src") == "obs":
                    xs["rain"][d] = w["mm"]
            wet_days = sum(1 for v in xs["rain"].values() if v >= 1.0)
            if wet_days < 3:
                xs["rain"] = {}   # a wet day or two is an anecdote, not a covariate
            rain_mu = (sum(xs["rain"].values()) / len(xs["rain"])) if xs["rain"] else 0.0
            resid = dict(r)
            coefs = {}
            for key, prior_sd in (("sib", self.opt["cov_prior_sd"]),
                                  ("prev", self.opt["cov_prior_sd"]),
                                  ("rain", self.opt["cov_rain_sd"])):
                cen = rain_mu if key == "rain" else 0.0
                rows = [(x - cen, resid[d]) for d, x in xs[key].items()]
                s2 = sum(y * y for _, y in rows) / len(rows) if rows else ve
                lam = s2 / (prior_sd * prior_sd)
                den = sum(x * x for x, _ in rows)
                b = sum(x * y for x, y in rows) / (den + lam) if den > 0 else 0.0
                coefs[key] = min(0.9, max(-0.9, b))   # a covariate must never own the whole day effect
                for d, x in xs[key].items():
                    resid[d] -= b * (x - cen)
            full = [d for d in days if all(d in xs[k] for k in xs)] or days
            s2f = sum(resid[d] ** 2 for d in full) / len(full) if full else ve
            n0 = self.opt["cov_shrink_days"]
            ve2 = min(ve, max(0.25 * ve, (len(full) * s2f + n0 * ve) / (len(full) + n0)))
            f.cov = {"sib": coefs["sib"], "prev": coefs["prev"], "rain": coefs["rain"],
                     "rain_mu": rain_mu, "rain_days": wet_days, "ve2": ve2, "n": len(days)}
            f.r_hist = r

    def fit(self, history_rows, context=None):
        """Cold build: replay the most recent days one by one (score day d with the model that only knew days < d,
        then fold d in), exactly the way the nightly job runs, so the ledger is genuinely out-of-time."""
        rows = sorted((dict(r) for r in history_rows), key=lambda r: (r["date"], r["minute"], r["direction"]))
        days = sorted({r["date"] for r in rows})
        self.cal = PITCalibrator(self.opt["n_bins"], self.opt["n0"], self.opt["decay"], self.opt["groups"], tuple(self.windows))
        self.ledger, self.flags, self.dirs, self._kcache = [], {"fallback": {}, "gate": {}}, {}, {}
        if len(days) <= MIN_REPLAY_HIST:
            self._rows = rows
            self._refit(None)
            return
        k0 = len(days) - min(self.opt["replay_days"], len(days) - MIN_REPLAY_HIST)
        by_day = defaultdict(list)
        for r in rows:
            by_day[r["date"]].append(r)
        self._rows = [r for d in days[:k0] for r in by_day[d]]
        self._refit(None)
        for d in days[k0:]:
            self.update(by_day[d])

    def update(self, day_rows):
        """The nightly job: (1) score the finished day with the model that existed before it, (2) fold the PIT
        values into the recalibration layer and refresh the label flags, (3) append the day and warm-refit."""
        day_rows = [dict(r) for r in day_rows]
        if day_rows and self._rows:
            assert max(r["date"] for r in self._rows) < min(r["date"] for r in day_rows), "update() out of order"
        if self.dirs:
            self._score_day(day_rows)
        self._rows.extend(day_rows)
        self._refit(dict(self.dirs))

    def _cmp_scores(self, direction, series, rows_before, day):
        cm = Comparator(self.windows).fit(rows_before, day)
        start, end = self.windows[direction]
        last = series[-1][0]
        n = ae = is80 = 0
        for origin in [None] + list(range(start, end, ORIGIN_STEP)):
            o = start - MORNING_OF_LEAD if origin is None else origin
            if origin is not None and last < o:
                continue
            targets = [t for t in range(start, end + 1, GRID) if origin is None or t > o]
            obs = [] if origin is None else [(m, v) for m, v in series if m <= o]
            for t, q in zip(targets, cm.predict(direction, o, obs, targets)):
                act = _match(series, t)
                if q is None or act is None:
                    continue
                n += 1
                ae += abs(act[1] - q["mean"])
                is80 += (q["q90"] - q["q10"]) + 10.0 * max(q["q10"] - act[1], 0.0) + 10.0 * max(act[1] - q["q90"], 0.0)
        return (ae / n, is80 / n, n) if n else None

    def _score_day(self, day_rows):
        by_dir = defaultdict(list)
        for r in day_rows:
            by_dir[r["direction"]].append((r["minute"], r["minutes"]))
        day = day_rows[0]["date"] if day_rows else None
        for direction, series in by_dir.items():
            f = self.dirs.get(direction)
            if f is None or f.m is None or any(l["date"] == day and l["direction"] == direction for l in self.ledger):
                continue   # unfitted, or this (day, direction) is already in the ledger: never count a day twice
            series.sort()
            start, end = self.windows[direction]
            in_window = sum(1 for m, _ in series if start <= m <= end)
            if in_window < max(3, math.ceil(MIN_SCORED_FRACTION * ((end - start) // GRID + 1))):
                continue   # too little of the day was sampled to say anything about the model; it still joins the history
            last = series[-1][0]
            weekday = date.fromisoformat(day).weekday()
            sib_series = by_dir.get(sibling_stream(direction))
            cells = defaultdict(list)
            st = {"n": 0, "ae": 0.0, "cov80": 0, "cov90": 0, "cov80_raw": 0, "is80": 0.0}
            gst = defaultdict(lambda: {"n": 0, "ae": 0.0, "is80": 0.0})   # per horizon-group detail
            for origin in [None] + list(range(start, end, ORIGIN_STEP)):
                o = start - MORNING_OF_LEAD if origin is None else origin
                if origin is not None and last < o:
                    continue
                targets = [t for t in range(start, end + 1, GRID) if origin is None or t > o]
                obs = [] if origin is None else [(m, v) for m, v in series if m <= o]
                cov = covariate_term(self.dirs, self.weather, direction, day, o, sib_series)
                raw = raw_predict(f, o, obs, targets, weekday, cov)
                for t, mv in zip(targets, raw):
                    act = _match(series, t)
                    if mv is None or act is None:
                        continue
                    mu, sd = mv
                    y = act[1]
                    g = group_of(self.windows, direction, origin, t, self.opt["groups"])
                    cells[g].append((S._ND.cdf((y - 0.5 - mu) / sd), S._ND.cdf((y + 0.5 - mu) / sd)))
                    ks = _ks(self.cal, self._kcache, direction, g)
                    q10, q90, q05, q95 = mu + sd * ks[1], mu + sd * ks[5], mu + sd * ks[0], mu + sd * ks[6]
                    is80 = (q90 - q10) + 10.0 * max(q10 - y, 0.0) + 10.0 * max(y - q90, 0.0)
                    st["n"] += 1
                    st["ae"] += abs(y - mu)
                    st["cov80"] += q10 <= y <= q90
                    st["cov90"] += q05 <= y <= q95
                    st["cov80_raw"] += mu + sd * S._ND.inv_cdf(0.10) <= y <= mu + sd * S._ND.inv_cdf(0.90)
                    st["is80"] += is80
                    gst[g]["n"] += 1
                    gst[g]["ae"] += abs(y - mu)
                    gst[g]["is80"] += is80
            if not st["n"]:
                continue
            n = st["n"]
            cmp_ = self._cmp_scores(direction, series, [r for r in self._rows if r["date"] < day], day)
            if cmp_ is not None and cmp_[2] < 0.7 * n:
                cmp_ = None   # a comparator that answered few of the same targets is not comparable: leave the flag alone
            # What the model leaned on that morning, so its learning can be watched over time.
            learned = {"n_days": f.info.get("n_days", 0)}
            if f.theta is not None:
                learned["phi"], learned["tau"] = round(f.theta.phi, 4), round(f.theta.tau, 4)
            for k in ("sib", "prev", "rain", "rain_mu", "ve2"):
                v = (f.cov or {}).get(k)
                if isinstance(v, (int, float)) and math.isfinite(v):
                    learned[k] = round(v, 4)
            self.ledger.append({"date": day, "direction": direction, "n": n, "mae": st["ae"] / n,
                                "learned": learned,
                                "cov80": st["cov80"] / n, "cov90": st["cov90"] / n, "cov80_raw": st["cov80_raw"] / n,
                                "is80": st["is80"] / n, "cmp_mae": cmp_[0] if cmp_ else None,
                                "cmp_is80": cmp_[1] if cmp_ else None,
                                "groups": {g: {"n": s["n"], "mae": s["ae"] / s["n"], "is80": s["is80"] / s["n"]}
                                           for g, s in sorted(gst.items()) if s["n"]}})
            if f.info.get("n_days", 0) >= self.opt["cal_min_hist"]:
                self.cal.add_day(direction, cells)
                self._kcache = {k: v for k, v in self._kcache.items() if k[0] != direction}
            self.flags = update_flags(self.opt, self.ledger, self.flags, direction)
        self.ledger = self.ledger[-LEDGER_KEEP:]

    # ---------------- prediction ----------------
    def predict(self, direction, origin_minute, observed_today, target_minutes, observed_other=None, context=None,
                weekday=None, day=None):
        f = self.dirs.get(direction)
        if f is None or f.m is None:
            return [None] * len(target_minutes)
        cov = covariate_term(self.dirs, self.weather, direction, day, origin_minute, observed_other)
        raw = raw_predict(f, origin_minute, observed_today, target_minutes, weekday, cov)
        return _quantiles(self.windows, self.cal, self._kcache, self.opt["groups"], direction, origin_minute,
                          observed_today, target_minutes, raw)

    def status(self, direction):
        f = self.dirs.get(direction)
        n_hist = f.info.get("n_days", 0) if (f is not None and f.m is not None) else 0
        return compute_status(self.opt, self.ledger, self.flags, direction, n_hist)

    # ---------------- persistence ----------------
    def to_state(self):
        st = super().to_state()
        st.update({"final_version": CODE_VERSION, "opt": self.opt, "cal": self.cal.to_state(),
                   "ledger": self.ledger[-LEDGER_KEEP:], "flags": self.flags, "kq": self._kq_table(),
                   "weather": self.weather})
        st["cfg"] = dict(st["cfg"], windows={d: list(v) for d, v in self.windows.items()})
        return st

    def _kq_table(self):
        return {d: [[round(k, 5) for k in _ks(self.cal, self._kcache, d, g)] for g in range(self.opt["groups"])]
                for d, f in self.dirs.items() if f.m is not None}

    @classmethod
    def from_state(cls, state, history_rows):
        """Restore for the nightly job: persisted engine/calibration/ledger plus the CSV rows (the source of truth)."""
        if state.get("final_version") != CODE_VERSION:
            raise ValueError("state version mismatch")
        windows = {d: tuple(v) for d, v in state["cfg"]["windows"].items()}
        m = cls(windows, **state["opt"])
        m.cfg = dict(state["cfg"], windows=windows)
        m._rows = sorted((dict(r) for r in history_rows), key=lambda r: (r["date"], r["minute"], r["direction"]))
        m.dirs = _restore_dirs(m.cfg, state["dirs"], m._rows)
        for d in windows:
            if d not in m.dirs:
                f = S._DirFit(d, m.cfg)
                f.set_history(m._rows)
                m.dirs[d] = f
        m.cal = PITCalibrator.from_state(state["cal"])
        m.ledger = list(state.get("ledger", []))
        m.flags = {"fallback": dict(state.get("flags", {}).get("fallback", {})),
                   "gate": dict(state.get("flags", {}).get("gate", {}))}
        m.weather = {d: dict(v) for d, v in state.get("weather", {}).items()}
        return m


def _restore_dirs(cfg, dirs_state, rows=None):
    out = {}
    for d, st in dirs_state.items():
        f = S._DirFit(d, cfg)
        if rows is not None:
            f.set_history(rows)
        f.m, f.neff_slot, f.bw, f.gamma, f.mbar, f.m0 = st["m"], st["neff"], st["bw"], st["gamma"], st["mbar"], st["m0"]
        f.n_eff_days = st["n_eff_days"]
        t = st["theta"]
        f.theta = S.Theta(t["tau"], t["sx"], t["rho"], t["so"], t.get("phi", 0.0))
        f.wd_deltas = {int(k): [tuple(x) for x in v] for k, v in st.get("wd_deltas", {}).items()}
        f.r_hist = {k: float(v) for k, v in st.get("r_hist", {}).items()}
        f.cov = {k: float(v) for k, v in st.get("cov", {}).items()}
        f.info = {"n_days": st["n_days"], "n_eff_days": st["n_eff_days"]}
        out[d] = f
    return out


class PollPredictor:
    """Read-only predictor rebuilt from the persisted model state: no history, no fitting (poll time)."""

    def __init__(self, state):
        if state.get("final_version") != CODE_VERSION:
            raise ValueError("state version mismatch")
        self.windows = {d: tuple(v) for d, v in state["cfg"]["windows"].items()}
        self.cfg = dict(state["cfg"], windows=self.windows)
        self.opt = state["opt"]
        self.dirs = _restore_dirs(self.cfg, state["dirs"])
        self.cal = PITCalibrator.from_state(state["cal"])
        self.ledger = state.get("ledger", [])
        self.flags = state.get("flags", {})
        self.weather = {d: dict(v) for d, v in state.get("weather", {}).items()}
        self._kcache = {(d, g): ks for d, rows in state.get("kq", {}).items() for g, ks in enumerate(rows)}

    def predict(self, direction, origin_minute, observed_today, target_minutes, weekday=None, day=None,
                observed_other=None):
        f = self.dirs.get(direction)
        if f is None or f.m is None:
            return [None] * len(target_minutes)
        cov = covariate_term(self.dirs, self.weather, direction, day, origin_minute, observed_other)
        raw = raw_predict(f, origin_minute, observed_today, target_minutes, weekday, cov)
        return _quantiles(self.windows, self.cal, self._kcache, self.opt["groups"], direction, origin_minute,
                          observed_today, target_minutes, raw)

    def n_days(self, direction):
        f = self.dirs.get(direction)
        return f.info.get("n_days", 0) if (f is not None and f.m is not None) else 0

    def status(self, direction):
        return compute_status(self.opt, self.ledger, self.flags, direction, self.n_days(direction))


# --------------------------------------------------------------------------------------------------
# Persistence: atomic, validated, with a previous-good fallback
# --------------------------------------------------------------------------------------------------
def model_path(csv_path):
    """commute_live.csv -> commute_model.json; commute_live_<hash>.csv -> commute_model_<hash>.json (one per route)."""
    base = os.path.basename(csv_path)
    stem = os.path.splitext(base)[0].replace("commute_live", "commute_model", 1)
    if not stem.startswith("commute_model"):
        stem = "commute_model_" + stem
    return os.path.join(os.path.dirname(os.path.abspath(csv_path)), stem + ".json")


def _finite(x):
    if isinstance(x, float):
        return math.isfinite(x)
    if isinstance(x, (list, tuple)):
        return all(_finite(v) for v in x)
    if isinstance(x, dict):
        return all(_finite(v) for v in x.values())
    return True


def _digest(model):
    return hashlib.sha256(json.dumps(model, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:16]


def validate_state(st, windows=None, skip=None):
    """None if the state is usable, else a short reason."""
    try:
        if st.get("schema") != SCHEMA:
            return "schema"
        if st.get("code_version") != CODE_VERSION:
            return "code version"
        if windows is not None and st.get("windows") != {d: list(v) for d, v in windows.items()}:
            return "windows changed"
        if skip is not None and st.get("skip_key") != skip:
            return "days-off settings changed"
        body = st["model"]
        if not _finite(body):
            return "non-finite"
        if not body.get("dirs"):
            return "no direction fitted"
        for d, e in body["dirs"].items():
            if len(e["m"]) != len(e["neff"]) or not 5 <= len(e["m"]) <= 400:
                return "curve length"
            if not all(1.0 < v < 400.0 for v in e["m"]):
                return "curve range"
            t = e["theta"]
            if not (0 < t["tau"] < 60 and 0 < t["sx"] < 60 and 0 < t["rho"] < 1 and 0 <= t["so"] < 30):
                return "theta range"
            c = e.get("cov", {})
            if (any(not -1.0 <= c.get(k, 0.0) <= 1.0 for k in ("sib", "prev"))
                    or not -2.0 <= c.get("rain", 0.0) <= 2.0 or not 0.0 < c.get("ve2", 1.0) <= 60.0):
                return "covariate range"
        for day, w in body.get("weather", {}).items():
            if not isinstance(w, dict) or not isinstance(w.get("mm"), (int, float)) or w.get("src") not in ("obs", "fcst"):
                return "weather"
        for d, rows in body["kq"].items():
            for ks in rows:
                if any(b <= a for a, b in zip(ks, ks[1:])) or not (-12 < ks[0] < 0 < ks[-1] < 12):
                    return "quantile multipliers"
        return None
    except (KeyError, TypeError, AttributeError, IndexError):
        return "malformed"


def save_state(path, wrapped):
    """Temp file in the same directory, fsync, keep the previous good file as .prev, atomic rename."""
    fd, tmp = tempfile.mkstemp(prefix=".model_state.", suffix=".tmp", dir=os.path.dirname(os.path.abspath(path)))
    try:
        with os.fdopen(fd, "w") as fh:
            json.dump(wrapped, fh, separators=(",", ":"), allow_nan=False)
            fh.flush()
            os.fsync(fh.fileno())
        if os.path.exists(path):
            os.replace(path, path + ".prev")
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def load_state(path, windows=None, skip=None):
    """Recovery chain: current -> .prev -> None. Returns (state or None, source, reason)."""
    reasons = []
    for src, p in (("current", path), ("prev", path + ".prev")):
        try:
            with open(p) as fh:
                st = json.load(fh)
        except FileNotFoundError:
            reasons.append(f"{src}: missing")
            continue
        except (OSError, ValueError) as exc:
            reasons.append(f"{src}: unreadable ({type(exc).__name__})")
            continue
        why = validate_state(st, windows, skip)
        if why is None and st.get("digest") != _digest(st["model"]):
            why = "digest mismatch"
        if why is None:
            return st, src, "; ".join(reasons) or "ok"
        reasons.append(f"{src}: {why}")
    return None, "none", "; ".join(reasons)


def wrap(model, windows, skip, through, last_day):
    return {"schema": SCHEMA, "code_version": CODE_VERSION, "windows": {d: list(v) for d, v in windows.items()},
            "skip_key": skip, "through": through, "last_day": last_day, "digest": _digest(model), "model": model}


@contextmanager
def model_lock():
    """Only one training job at a time; yields False if another holds the lock."""
    with open(LOCK, "a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            yield False
            return
        try:
            yield True
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def peek_header(path):
    """Cheap read of the state header ({through, windows, ...}) without validation; None if unreadable."""
    try:
        with open(path) as fh:
            st = json.load(fh)
        return {k: st.get(k) for k in ("schema", "code_version", "windows", "skip_key", "through", "last_day")}
    except (OSError, ValueError):
        return None


def _remove_orphans(directory, min_age=300):
    """Temp files left by a training job that was killed mid-write."""
    for name in os.listdir(directory):
        if name.startswith(".model_state.") and name.endswith(".tmp"):
            full = os.path.join(directory, name)
            try:
                if datetime.now().timestamp() - os.path.getmtime(full) > min_age:
                    os.unlink(full)
            except OSError:
                pass


def sync_due(settings, now=None, csv_path=None):
    """True when a nightly job could do something: no usable-looking state, or a newer complete day exists."""
    now = (now or datetime.now(TZ)).astimezone(TZ)
    windows = windows_of(settings)
    path = model_path(csv_path or sample_path(settings, CSV))
    head = peek_header(path)
    if (head is None or head["schema"] != SCHEMA or head["code_version"] != CODE_VERSION
            or head["windows"] != {d: list(v) for d, v in windows.items()} or head["skip_key"] != skip_key(settings)):
        return True
    return (head["through"] or "") < latest_complete_day(now, windows).isoformat()


# --------------------------------------------------------------------------------------------------
# Weather: the rain covariate's input, fetched once a day inside the sync job only (the poll path never
# goes to the network). Keyless Open-Meteo; any failure just keeps the previously stored values.
# --------------------------------------------------------------------------------------------------
def _fetch_weather(lat, lon, first_day, today, timeout=15):
    """{ISO date: {"mm": daily precipitation, "src": "obs"|"fcst"}} for [first_day .. today+7d].

    Historical days come from the archive endpoint (ERA5 reanalysis — the forecast endpoint's past-day
    backfill underreports rain); it lags ~5 days, so the forecast endpoint covers only the recent gap
    and the 7-day outlook. Returns {} on any failure (the caller keeps the stored weather)."""
    out = {}

    def pull(url):
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return json.load(resp)

    try:
        recent = today - timedelta(days=5)
        url = ("https://api.open-meteo.com/v1/forecast?latitude=%s&longitude=%s&daily=precipitation_sum"
               "&past_days=7&forecast_days=7&timezone=auto" % (lat, lon))
        daily = pull(url).get("daily", {})
        for t, p in zip(daily.get("time", ()), daily.get("precipitation_sum", ())):
            if p is not None:
                out[t] = {"mm": float(p), "src": "obs" if t <= today.isoformat() else "fcst"}
        if recent.isoformat() >= first_day:
            url = ("https://archive-api.open-meteo.com/v1/archive?latitude=%s&longitude=%s"
                   "&start_date=%s&end_date=%s&daily=precipitation_sum&timezone=auto"
                   % (lat, lon, first_day, recent.isoformat()))
            daily = pull(url).get("daily", {})
            for t, p in zip(daily.get("time", ()), daily.get("precipitation_sum", ())):
                if p is not None:
                    out[t] = {"mm": float(p), "src": "obs"}   # archive wins wherever it has a value
    except Exception:
        return {}
    return out


# --------------------------------------------------------------------------------------------------
# The nightly job
# --------------------------------------------------------------------------------------------------
def sync(settings=None, now=None, csv_path=None, force_rebuild=False, log=print):
    """Fold every finished day not yet seen into the persisted model. Idempotent; safe to run any time."""
    settings = settings if settings is not None else load_settings()
    now = (now or datetime.now(TZ)).astimezone(TZ)
    windows = windows_of(settings)
    csv_path = csv_path or sample_path(settings, CSV)
    path = model_path(csv_path)
    skip = skip_key(settings)
    with model_lock() as got:
        if not got:
            return {"status": "busy"}
        _remove_orphans(os.path.dirname(os.path.abspath(path)))
        target = latest_complete_day(now, windows).isoformat()
        rows = [r for r in load_clean_rows(csv_path, settings) if r["date"] <= target]
        if not rows:
            return {"status": "no data"}   # a missing or emptied CSV must never overwrite what was learned
        st, source, why = (None, "none", "forced") if force_rebuild else load_state(path, windows, skip)
        weather = dict(((st or {}).get("model") or {}).get("weather", {}))
        lat_lon = weather_point(settings)
        if lat_lon is not None:
            fetched = _fetch_weather(lat_lon[0], lat_lon[1], rows[0]["date"], now.date())
            weather.update(fetched)   # freshly fetched wins; older stored days survive beyond the API's reach
        model = None
        if st is not None:
            try:
                last = st["last_day"] or ""
                known = len({r["date"] for r in rows if r["date"] <= last})
                if known < max((e["n_days"] for e in st["model"]["dirs"].values()), default=0):
                    raise ValueError("the CSV has fewer days than the model has learned")   # replaced or truncated data
                model = FinalModel.from_state(st["model"], [r for r in rows if r["date"] <= last])
                model.weather = weather
                pending = sorted({r["date"] for r in rows if r["date"] > last})
            except (ValueError, KeyError, TypeError) as exc:
                log(f"model state unusable ({exc}); rebuilding")
                model = None
        if model is not None:
            if source == "prev":
                log("current model state was unusable; continued from the previous one")
            by_day = defaultdict(list)
            for r in rows:
                by_day[r["date"]].append(r)
            for d in pending:
                model.update(by_day[d])
            action = f"updated {len(pending)} day(s)"
        else:
            log(f"building the model from scratch ({why})")
            model = FinalModel(windows)
            model.weather = weather
            model.fit(rows)
            action = "rebuilt"
        if not any(f.m is not None for f in model.dirs.values()):
            return {"status": "insufficient", "days": len({r["date"] for r in rows})}
        last_day = max((r["date"] for r in rows), default=None)
        wrapped = wrap(model.to_state(), windows, skip, target, last_day)
        # canary: the file we are about to promote must round-trip and validate
        if validate_state(json.loads(json.dumps(wrapped)), windows, skip) is not None:
            raise RuntimeError("refusing to write an invalid model state")
        save_state(path, wrapped)
        return {"status": "ok", "action": action, "last_day": last_day, "through": target,
                "labels": {d: model.status(d)["label"] for d in windows}}


def load_for_snapshot(csv_path, settings):
    """(PollPredictor, None) for a usable state, else (None, reason). Read-only."""
    windows = windows_of(settings)
    st, source, why = load_state(model_path(csv_path), windows, skip_key(settings))
    if st is None:
        return None, why
    try:
        return PollPredictor(st["model"]), None
    except (ValueError, KeyError, TypeError) as exc:
        return None, f"unusable ({exc})"


def main(argv=None):
    ap = argparse.ArgumentParser(description="Train / inspect the Predicted model")
    ap.add_argument("--sync", action="store_true", help="fold finished days into the model (the nightly job)")
    ap.add_argument("--rebuild", action="store_true", help="with --sync: discard the saved state and rebuild")
    ap.add_argument("--status", action="store_true", help="print the label and readout per direction")
    args = ap.parse_args(argv)
    try:
        settings = load_settings()
    except SettingsError as exc:
        print(f"Settings error: {exc}", file=sys.stderr)
        return 1
    if args.sync:
        result = sync(settings, force_rebuild=args.rebuild)
        print(json.dumps(result))
        return 0 if result["status"] in ("ok", "busy", "insufficient") else 1
    csv_path = sample_path(settings, CSV)
    pm, why = load_for_snapshot(csv_path, settings)
    if pm is None:
        print(f"no usable model state: {why}")
        return 0
    for d in pm.windows:
        print(d, json.dumps(pm.status(d)))
    return 0


if __name__ == "__main__":
    sys.exit(main())

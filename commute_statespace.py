"""State-space / Bayesian day-effect engine for the Commute Bar "Predicted" model (stdlib only).

Model (per direction), y = Google-Maps drive time in whole minutes at minute-of-day t of day d:

    y(d,t) = m(t) + s(t) * [ delta_d + x_d(t) ] + e(d,t)
      m(t)      smooth time-of-day mean curve: Gaussian-kernel smoother over days, learned jointly with the day
                effects by backfitting (= BLUP of a crossed random-effects model), exponentially forgotten by day
      s(t)      heteroscedastic scale law: 'add' s=1; 'lvl' s=m(t)/mbar (proportional, log-like); 'exc' congestion
                excess over free flow. Chosen per direction by LODO in the *auto* variant.
      delta_d   persistent day effect, N(0, tau0^2), tau0^2 = tau^2 (1 + 1/N_eff)   [parameter uncertainty of m]
      x_d(t)    OU / AR(1) within-day deviation, stationary var sx^2, per-10-min coefficient rho
      e         measurement noise: whole-minute rounding (1/12 min^2) + a small learned nugget
    delta, x are tracked by a 2-state Kalman filter over today's observations (irregular times, any gaps).
    The Kalman gain IS the live-adjustment weight curve (horizon dependent) that the shipped app hard-codes.

Predictive distribution of the *reported* value at a target time (it includes measurement noise and the
slot-specific error of m):  y_t ~ m(t) + s(t)*mu_u  +  scale * StudentT(nu),  with
    scale^2 = s^2 [Pdd + a^2 Pxx + 2 a Pdx + sx^2 (1-a^2) + sx^2/n_eff(t)] + so^2 s^2 + 1/12,   a = rho^(h/10)
nu = clamp(N_eff_days - 1) so tails are heavy while few days are available and become Gaussian by ~30 days.

Hyper-parameters (tau, sx, rho, nugget, kernel bandwidth, scale law; optional weekday share phi) are learned from PRIOR days only by
leave-one-day-out (LODO) predictive log-likelihood (a cross-validated marginal likelihood) with a weak, generic
log-normal prior (see PRIOR_*), Nelder-Mead in log space, warm-startable (`update(day_rows)` for online use).

Cheap at poll time: predict() is a lookup of the cached curve + an O(#obs) filter. State can be exported to a
small JSON-able dict (`to_state`) and a predictor rebuilt from it (`from_state`) without the history.
"""
import math
from collections import defaultdict
from datetime import date
from functools import lru_cache
from statistics import NormalDist

WINDOWS = {"to_work": (7 * 60, 11 * 60), "to_home": (17 * 60, 21 * 60)}  # defaults; a model passes its own via cfg["windows"]
GRID = 10
QLEVELS = (("q05", 0.05), ("q10", 0.10), ("q25", 0.25), ("q50", 0.50), ("q75", 0.75), ("q90", 0.90), ("q95", 0.95))
EXC_FLOOR = 0.4           # congestion-scale floor (relative sd at free flow)
PRE_WINDOW = 60            # minutes before window start whose observations are used (live filter and fit)
ROUND_VAR = 1.0 / 12.0     # whole-minute rounding variance
# A deviation far beyond the usual day-to-day spread is more likely an incident (closure, crash) than noise, and the
# fitted fast-decaying part would fade it within the hour. Only the part past INCIDENT_Z standard deviations is treated
# that way: it fades with INCIDENT_KEEP per 10 minutes (half-life ~1 h) and adds range, since whether it clears is
# unknown. Hand-chosen, not fitted: no day in the history has gone this far (the largest held-out miss was ~+4 min,
# below the ~+4.7 min this threshold means here), so the held-out scoring is untouched by it. Revisit once real cases exist.
INCIDENT_Z = 2.5
INCIDENT_KEEP = 0.89
INCIDENT_SD = 0.5          # range added at the start, as a share of the excess minutes (settles to half of it)
_ND = NormalDist()


# --------------------------------------------------------------------------------------------------
# Student-t quantiles (stdlib): regularized incomplete beta by continued fraction + bisection, cached
# --------------------------------------------------------------------------------------------------
def _betacf(a, b, x):
    tiny = 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 300):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-14:
            break
    return h


def _betainc(a, b, x):
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    bt = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x))
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(a, b, x) / a
    return 1.0 - bt * _betacf(b, a, 1.0 - x) / b


def _t_cdf(z, nu):
    x = nu / (nu + z * z)
    tail = 0.5 * _betainc(nu / 2.0, 0.5, x)
    return 1.0 - tail if z > 0 else tail


@lru_cache(maxsize=4096)
def t_quantile(p, nu):
    """Quantile of a standard Student-t with (possibly fractional) nu degrees of freedom; nu >= 30 -> normal."""
    if nu >= 60.0:
        return _ND.inv_cdf(p)
    if p == 0.5:
        return 0.0
    if p < 0.5:
        return -t_quantile(1.0 - p, nu)
    lo, hi = 0.0, 1.0
    while _t_cdf(hi, nu) < p:
        hi *= 2.0
        if hi > 1e8:
            break
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if _t_cdf(mid, nu) < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


@lru_cache(maxsize=256)
def _t_logconst(nu):
    return math.lgamma((nu + 1.0) / 2.0) - math.lgamma(nu / 2.0) - 0.5 * math.log(nu * math.pi)


def nu_from_days(n_eff):
    return round(min(60.0, max(2.0, n_eff - 1.0)), 1)


# --------------------------------------------------------------------------------------------------
# Generic Nelder-Mead (minimise f over R^k); small and deterministic
# --------------------------------------------------------------------------------------------------
def nelder_mead(f, x0, step=0.4, iters=140, tol=1e-4):
    k = len(x0)
    pts = [list(x0)]
    for i in range(k):
        p = list(x0)
        p[i] += step
        pts.append(p)
    vals = [f(p) for p in pts]
    for _ in range(iters):
        order = sorted(range(k + 1), key=lambda i: vals[i])
        pts = [pts[i] for i in order]
        vals = [vals[i] for i in order]
        if abs(vals[-1] - vals[0]) < tol:
            break
        cen = [sum(p[j] for p in pts[:-1]) / k for j in range(k)]
        refl = [cen[j] + (cen[j] - pts[-1][j]) for j in range(k)]
        fr = f(refl)
        if fr < vals[0]:
            exp = [cen[j] + 2.0 * (cen[j] - pts[-1][j]) for j in range(k)]
            fe = f(exp)
            pts[-1], vals[-1] = (exp, fe) if fe < fr else (refl, fr)
        elif fr < vals[-2]:
            pts[-1], vals[-1] = refl, fr
        else:
            con = [cen[j] + 0.5 * (pts[-1][j] - cen[j]) for j in range(k)]
            fc = f(con)
            if fc < vals[-1]:
                pts[-1], vals[-1] = con, fc
            else:
                for i in range(1, k + 1):
                    pts[i] = [pts[0][j] + 0.5 * (pts[i][j] - pts[0][j]) for j in range(k)]
                    vals[i] = f(pts[i])
    i = min(range(k + 1), key=lambda i: vals[i])
    return pts[i], vals[i]


# --------------------------------------------------------------------------------------------------
# Hyper-parameter container, priors
# --------------------------------------------------------------------------------------------------
# Weak GENERIC priors (hand-chosen before looking at test scores; not taken from the 9 days' measured SDs):
# day effect and OU sd each ~7% of the mean level, lag-10-min correlation 0.6, nugget 0.3 min. Log-sd 0.7.
PRIOR_CV = 0.07
PRIOR_RHO = 0.6
PRIOR_NUGGET = 0.3
PRIOR_LOGSD = 0.7


class Theta:
    """tau, sx (minutes at mean level), rho (per 10 min), so (extra obs sd, minutes)."""
    __slots__ = ("tau", "sx", "rho", "so", "phi")

    def __init__(self, tau, sx, rho, so, phi=0.0):
        # phi = share of the day-effect variance that is a weekday effect (0 = weekday effects switched off)
        self.tau, self.sx, self.rho, self.so, self.phi = tau, sx, rho, so, phi

    @staticmethod
    def from_vec(v):
        c = lambda z, lo, hi: min(hi, max(lo, z))
        phi = 1.0 / (1.0 + math.exp(-c(v[4], -7.0, 3.0))) if len(v) > 4 else 0.0
        return Theta(math.exp(c(v[0], -4, 3)), math.exp(c(v[1], -4, 3)),
                     math.exp(-math.exp(c(v[2], -4.5, 2.0))), math.exp(c(v[3], -5, 1.5)), phi)

    def vec(self):
        v = [math.log(self.tau), math.log(self.sx), math.log(-math.log(self.rho)), math.log(self.so)]
        if self.phi > 0.0:
            v.append(math.log(self.phi / (1.0 - self.phi)))
        return v

    def asdict(self):
        return {"tau": self.tau, "sx": self.sx, "rho": self.rho, "so": self.so, "phi": self.phi}

    @staticmethod
    def prior(mbar, weekday=False):
        return Theta(PRIOR_CV * mbar, PRIOR_CV * mbar, PRIOR_RHO, PRIOR_NUGGET, PRIOR_PHI if weekday else 0.0)


# Weak prior on the weekday share of the day-effect variance (only used when weekday=True). Wide on purpose: a commute's
# weekday pattern (light Mondays, lighter Fridays) is often a large part of the day-to-day variation, and a tight prior
# pinned it near its centre. On this route's first month the fitted share moved to 0.4-0.55 once the prior allowed it, and
# held-out interval scores improved (docs/MODEL.md).
PRIOR_PHI = 0.30
PRIOR_PHI_LOGSD = 1.5


def _prior_penalty(v, mbar):
    p = Theta.prior(mbar, weekday=len(v) > 4).vec()
    pen = 0.0
    for i, (a, b) in enumerate(zip(v, p)):
        sd = PRIOR_PHI_LOGSD if i == 4 else PRIOR_LOGSD * (1.4 if i == 3 else 1.0)
        pen += 0.5 * ((a - b) / sd) ** 2
    return pen


# --------------------------------------------------------------------------------------------------
# Kalman filter for  z_i = delta + x(t_i) + e_i  (state [delta, x]; irregular times)
# --------------------------------------------------------------------------------------------------
def kf_prior(th, prior):
    """prior = (mean, var) of the day effect before any observation of today."""
    return (prior[0], 0.0, prior[1], 0.0, th.sx * th.sx)  # mu_d, mu_x, Pdd, Pdx, Pxx


def kf_advance(state, a, sxsq):
    """Propagate the OU part by a = rho^(dt/10)."""
    md, mx, pdd, pdx, pxx = state
    return (md, a * mx, pdd, a * pdx, a * a * pxx + sxsq * (1.0 - a * a))


def kf_update(state, z, r):
    md, mx, pdd, pdx, pxx = state
    s = pdd + 2.0 * pdx + pxx + r
    kd, kx = (pdd + pdx) / s, (pdx + pxx) / s
    inn = z - (md + mx)
    md, mx = md + kd * inn, mx + kx * inn
    # P <- P - K S K^T (Joseph-free; fine in double precision for 2 states)
    pdd2 = pdd - kd * kd * s
    pdx2 = pdx - kd * kx * s
    pxx2 = pxx - kx * kx * s
    return (md, mx, max(pdd2, 1e-12), pdx2, max(pxx2, 1e-12))


def kf_forecast(state, a, sxsq):
    """Mean and variance of u(t_last + dt) (day effect + OU), without measurement noise."""
    md, mx, pdd, pdx, pxx = state
    mean = md + a * mx
    var = pdd + a * a * pxx + 2.0 * a * pdx + sxsq * (1.0 - a * a)
    return mean, max(var, 1e-12)


# --------------------------------------------------------------------------------------------------
# One-direction model
# --------------------------------------------------------------------------------------------------
class _DirFit:
    """History of one direction + the learned curve and hyper-parameters."""

    def __init__(self, direction, cfg):
        self.direction = direction
        self.cfg = cfg
        s, e = cfg["windows"][direction]
        self.s, self.e = s, e
        self.lo, self.hi = s - PRE_WINDOW, e              # observation range used by the filter
        self.g0 = self.lo
        self.G = list(range(self.lo, self.hi + 1, GRID))
        self.days = []        # sorted list of dates
        self.obs = {}         # date -> sorted [(minute, y)] in [lo-pad, hi+pad]
        self.theta = None
        self.bw = cfg["bw_grid"][len(cfg["bw_grid"]) // 2]
        self.gamma = cfg["scale"] if cfg["scale"] is not None else "add"
        self.m0 = 15.0
        self.m = None
        self.neff_slot = None
        self.n_eff_days = 0.0
        self.mbar = 20.0
        self.info = {}
        self.delta_std = {}
        self.wd_deltas = {}
        self.wd = {}
        self.r_hist = {}      # date -> residual day effect (delta minus its predict-time weekday mean)
        self.cov = {}         # fitted day-effect covariates: {"sib","prev","rain","rain_mu","ve2","n"}

    # ---- data -------------------------------------------------------------------------------
    def set_history(self, rows):
        pad = int(3 * max(self.cfg["bw_grid"])) + GRID
        obs = defaultdict(list)
        for r in rows:
            if r["direction"] == self.direction and self.lo - pad <= r["minute"] <= self.hi + pad:
                obs[r["date"]].append((r["minute"], r["minutes"]))
        self.obs = {}
        for d, v in obs.items():
            v.sort()
            if sum(1 for t, _ in v if self.lo <= t <= self.hi) >= self.cfg["min_day_obs"]:
                self.obs[d] = v
        self.days = sorted(self.obs)
        self.wd = {d: date.fromisoformat(d).weekday() for d in self.days}

    def day_weights(self, days=None):
        days = self.days if days is None else days
        hl = self.cfg["half_life_days"]
        n = len(days)
        if not hl:
            return {d: 1.0 for d in days}
        return {d: 0.5 ** ((n - 1 - i) / hl) for i, d in enumerate(days)}

    # ---- kernel sums -------------------------------------------------------------------------
    def _sums(self, bw):
        """Per day: (num_y[g], den[g]) kernel sums on the grid for Gaussian kernel bandwidth bw."""
        out = {}
        reach = 2.6 * bw
        nG = len(self.G)
        for d in self.days:
            num, den = [0.0] * nG, [0.0] * nG
            for t, y in self.obs[d]:
                gi0 = max(0, int(math.ceil((t - reach - self.g0) / GRID)))
                gi1 = min(nG - 1, int(math.floor((t + reach - self.g0) / GRID)))
                for gi in range(gi0, gi1 + 1):
                    u = (t - self.G[gi]) / bw
                    w = math.exp(-0.5 * u * u)
                    num[gi] += w * y
                    den[gi] += w
            out[d] = (num, den)
        return out

    @staticmethod
    def _curve(sums, delta, wts, days):
        nG = len(next(iter(sums.values()))[0]) if sums else 0
        m = [float("nan")] * nG
        for gi in range(nG):
            a = b = 0.0
            for d in days:
                num, den = sums[d]
                w = wts[d]
                a += w * (num[gi] - delta.get(d, 0.0) * den[gi])
                b += w * den[gi]
            if b > 0.35:
                m[gi] = a / b
        return _fill(m)

    @staticmethod
    def _neff(sums, wts, days, nG):
        out = []
        for gi in range(nG):
            sw = sw2 = 0.0
            for d in days:
                a = wts[d] * min(sums[d][1][gi], 1.0)
                sw += a
                sw2 += a * a
            out.append((sw * sw / sw2) if sw2 > 0 else 0.0)
        return out

    # ---- curve helpers -----------------------------------------------------------------------
    def _m_at(self, m, t):
        x = (t - self.g0) / GRID
        i = int(math.floor(x))
        nG = len(m)
        if i < 0:
            return m[0]
        if i >= nG - 1:
            return m[-1]
        f = x - i
        return m[i] * (1.0 - f) + m[i + 1] * f

    def _scale_at(self, m, t, mode):
        """Relative sd of the day/OU deviation at time t. add: 1; lvl: m/mbar; exc: floor + share of the
        excess-over-free-flow (m - m0)/(mbar - m0) (variability comes from congestion; free flow is steady)."""
        if mode == "add":
            return 1.0
        mt = self._m_at(m, t)
        if mode == "lvl":
            return max(0.4, min(2.5, mt / self.mbar))
        den = max(self.mbar - self.m0, 1.0)
        return max(EXC_FLOOR, min(2.5, EXC_FLOOR + (1.0 - EXC_FLOOR) * max(mt - self.m0, 0.0) / den))

    def _neff_at(self, neff, t):
        return max(0.5, self._m_at(neff, t))

    def day_resid(self, d, m, gamma, delta_hint=None):
        """Standardised residual series (t, z, s) for a day's observations inside the filter range."""
        out = []
        for t, y in self.obs[d]:
            if self.lo <= t <= self.hi:
                s = self._scale_at(m, t, gamma)
                out.append((t, (y - self._m_at(m, t)) / s, s))
        return out

    # ---- filtering ---------------------------------------------------------------------------
    def filter_states(self, series, th, prior):
        """Sequential filter; returns list of (t, state_after) in time order."""
        sxsq = th.sx * th.sx
        st = kf_prior(th, prior)
        last = None
        states = []
        for t, z, s in series:
            if last is not None:
                a = th.rho ** ((t - last) / GRID)
                st = kf_advance(st, a, sxsq)
            r = (th.so * th.so) + ROUND_VAR / (s * s)
            st = kf_update(st, z, r)
            last = t
            states.append((t, st))
        return states

    def day_prior(self, th, n_eff_days, same_wd, cov=None):
        """Prior (mean, var) of today's standardised day effect. same_wd = [(delta_hat, weight)] of prior days
        with the same weekday (only when th.phi > 0): delta = a_wd + e, Var(a_wd) = phi tau^2, Var(e) = (1-phi) tau^2,
        a_wd posterior from the weekday's earlier day effects (treated as exact). The (1 + 1/N) factor covers the
        uncertainty of the mean curve (which absorbs the average day effect).

        cov = {"mean", "var", "ve2"} layers learned day-effect covariates (sibling direction, yesterday's own
        residual, forecast rain) on top: mean shifts the prior, var adds the covariates' estimation variance
        (e.g. the sibling delta's posterior variance), and ve2 is the post-covariate innovation variance which
        replaces the e-variance everywhere it appears."""
        infl = 1.0 + 1.0 / max(n_eff_days, 1.0)
        tsq = th.tau * th.tau
        va, ve = th.phi * tsq, (1.0 - th.phi) * tsq
        if cov is not None and cov.get("ve2") is not None:
            ve = min(ve, max(1e-3, cov["ve2"]))
        if th.phi <= 0.0 or not same_wd:
            mean, var = 0.0, (va + ve) * infl
        else:
            nw = sum(w for _, w in same_wd)
            sw = sum(d * w for d, w in same_wd)
            prec = 1.0 / va + nw / ve
            mean, var = (sw / ve) / prec, (ve + 1.0 / prec) * infl
        if cov is not None:
            mean += cov.get("mean", 0.0)
            var += cov.get("var", 0.0)
        return mean, var

    def predict_target(self, m, neff, gamma, th, prior, state, last_t, t):
        """(mean_minutes, scale_minutes) for target time t given filter state at last_t (None = prior)."""
        s = self._scale_at(m, t, gamma)
        sxsq = th.sx * th.sx
        if state is None:
            mean_u, var_u = prior[0], prior[1] + sxsq   # nothing observed today: prior day effect + stationary OU
        else:
            a = th.rho ** (max(t - last_t, 0) / GRID)
            mean_u, var_u = kf_forecast(state, a, sxsq)
        slot = sxsq / self._neff_at(neff, t)
        var = s * s * (var_u + slot + th.so * th.so) + ROUND_VAR
        mean = self._m_at(m, t) + s * mean_u
        if state is not None and t >= last_t:
            now_u = state[0] + state[1]   # today's current deviation (day effect + OU), in standardized units
            excess_u = now_u - INCIDENT_Z * math.sqrt(prior[1] + sxsq)
            if excess_u > 0.0:   # the part beyond what ordinary days reach
                keep = INCIDENT_KEEP ** ((t - last_t) / GRID)
                mean += s * min(excess_u * keep, max(0.0, now_u - mean_u))   # never forecast more deviation than there is now
                var += (INCIDENT_SD * s * excess_u * (0.5 + 0.5 * keep)) ** 2
        return mean, math.sqrt(var)

    # ---- fitting -----------------------------------------------------------------------------
    def _als(self, sums, th, gamma, days, wts, iters=5):
        """Backfit curve and day effects. Returns m, delta (dict), and the per-slot n_eff."""
        delta = {d: 0.0 for d in days}
        m = None
        n_eff_days = self._neff_days(wts, days)
        prior = (0.0, th.tau * th.tau * (1.0 + 1.0 / max(n_eff_days, 1.0)))   # no weekday term while backfitting
        dstd = {}
        for _ in range(iters):
            m = self._curve(sums, delta, wts, days)
            new = {}
            for d in days:
                ser = self.day_resid(d, m, gamma)
                if not ser:
                    new[d] = 0.0
                    continue
                # the day's own effect is estimated from the full day (posterior mean of delta)
                st = self.filter_states(ser, th, prior)[-1][1]
                dstd[d] = st[0]
                new[d] = st[0] * (sum(s for _, _, s in ser) / len(ser))   # delta in minutes at mean scale
            delta = new
        self.delta_std = dstd
        return m, delta, self._neff(sums, wts, days, len(self.G))

    @staticmethod
    def _neff_days(wts, days):
        sw = sum(wts[d] for d in days)
        sw2 = sum(wts[d] ** 2 for d in days)
        return sw * sw / sw2 if sw2 > 0 else 0.0

    def _loo_cache(self, sums, delta, wts, days):
        """For each of the most recent `lodo_days` days d: curve m_{-d}, n_eff_{-d} from all other days
        (day effects frozen at their full-fit values). Capping the number of held-out days bounds the cost."""
        out = {}
        for d in days[-self.cfg["lodo_days"]:]:
            others = [x for x in days if x != d]
            if len(others) < 2:
                continue
            m = self._curve(sums, delta, wts, others)
            out[d] = (m, self._neff(sums, wts, others, len(self.G)), self._neff_days(wts, others))
        return out

    def _targets_for_day(self, d):
        """[(origin_idx_info...)] pairs used by the LODO objective: for each origin the targets and actuals."""
        ser = dict(self.obs[d])
        s, e = self.s, self.e
        times = sorted(ser)
        cache = []
        origins = [None] + [o for o in (s, s + 60, s + 120, s + 180) if o < e]
        targets = []
        for t in range(s, e + 1, GRID):
            best = None
            for tt in times:
                dd = abs(tt - t)
                if dd <= 5 and (best is None or (dd, tt) < (best[0], best[1])):
                    best = (dd, tt)
            if best:
                targets.append((t, ser[best[1]]))
        last_t = times[-1]
        for o in origins:
            if o is not None and last_t < o:
                continue
            ts = [(t, y) for t, y in targets if (o is None or t > o)]
            if ts:
                cache.append((o, ts))
        return cache

    def _pack_loo(self, loo, days, tgt, gamma):
        """Everything in the LODO objective that does not depend on theta, computed once per curve."""
        packs = {}
        for d in days:
            if d not in loo:
                continue
            m, neff, nd_eff = loo[d]
            ser = self.day_resid(d, m, gamma)
            times = [t for t, _, _ in ser]
            pairs = []
            for o, ts in tgt[d]:
                k = 0 if o is None else sum(1 for t in times if t <= o)
                last = times[k - 1] if k else None
                for t, y in ts:
                    pairs.append((k, last, t, y, self._m_at(m, t), self._scale_at(m, t, gamma),
                                  1.0 / self._neff_at(neff, t)))
            if pairs:
                packs[d] = (ser, pairs, nd_eff)
        return packs

    def _lodo_nll(self, vec, packs, days, wts, dstd=None):
        th = Theta.from_vec(vec)
        sxsq, sosq, rho = th.sx * th.sx, th.so * th.so, th.rho
        total = 0.0
        for d in days:
            pk = packs.get(d)
            if pk is None:
                continue
            ser, pairs, nd_eff = pk
            same = ([(dstd[o], wts[o]) for o in days if o != d and o in dstd and self.wd[o] == self.wd[d]]
                    if (th.phi > 0.0 and dstd) else None)
            prior = self.day_prior(th, nd_eff, same)
            states = self.filter_states(ser, th, prior)
            nu = nu_from_days(nd_eff) if self.cfg["student"] else 1e9
            tconst = _t_logconst(nu)
            day_nll = 0.0
            for k, last, t, y, mt, sc, inv_neff in pairs:
                if k:
                    mean_u, var_u = kf_forecast(states[k - 1][1], rho ** (max(t - last, 0) / GRID), sxsq)
                else:
                    mean_u, var_u = prior[0], prior[1] + sxsq
                var = sc * sc * (var_u + sxsq * inv_neff + sosq) + ROUND_VAR
                zz2 = (y - (mt + sc * mean_u)) ** 2 / var
                if nu >= 60.0:
                    day_nll += 0.5 * zz2 + 0.5 * math.log(var) + 0.9189385
                else:
                    day_nll += (nu + 1.0) / 2.0 * math.log1p(zz2 / nu) + 0.5 * math.log(var) - tconst
            total += wts[d] * day_nll / len(pairs)
        return total

    def fit(self, warm=None):
        cfg = self.cfg
        days = self.days
        if len(days) < cfg["min_days"]:
            self.m = None
            return
        wts = self.day_weights()
        allw = [y for d in days for t, y in self.obs[d] if self.s <= t <= self.e] or [20.0]
        self.mbar = sum(allw) / len(allw)
        self.n_eff_days = self._neff_days(wts, days)
        th = warm.theta if (warm is not None and warm.theta is not None) else Theta.prior(self.mbar, cfg["weekday"])
        if cfg["weekday"] and th.phi <= 0.0:
            th = Theta(th.tau, th.sx, th.rho, th.so, PRIOR_PHI)
        if not cfg["weekday"] and th.phi > 0.0:
            th = Theta(th.tau, th.sx, th.rho, th.so, 0.0)
        tgt = {d: self._targets_for_day(d) for d in days[-cfg["lodo_days"]:]}
        gamma_grid = [cfg["scale"]] if cfg["scale"] is not None else list(cfg["scale_auto"])
        # free-flow reference: lowest value of the (day-effect-free) curve inside the window
        sums0 = self._sums(cfg["bw_grid"][-1])
        c0 = self._curve(sums0, {}, wts, days)
        win = [v for g, v in zip(self.G, c0) if self.s <= g <= self.e]
        self.m0 = min(win) if win else self.mbar - 3.0
        sums_by_bw = {bw: self._sums(bw) for bw in cfg["bw_grid"]}
        best = None
        # round 1: structure (bandwidth, gamma) at the current theta
        for gamma in gamma_grid:
            for bw in cfg["bw_grid"]:
                sums = sums_by_bw[bw]
                m, delta, _ = self._als(sums, th, gamma, days, wts, iters=3)
                loo = self._loo_cache(sums, delta, wts, days)
                if not loo:
                    continue
                packs = self._pack_loo(loo, days, tgt, gamma)
                f = (self._lodo_nll(th.vec(), packs, days, wts, dict(self.delta_std))
                     + _prior_penalty(th.vec(), self.mbar))
                if best is None or f < best[0]:
                    best = (f, bw, gamma)
        if best is None:
            self.m = None
            return
        _, self.bw, self.gamma = best
        sums = sums_by_bw[self.bw]
        # round 2/3: theta by Nelder-Mead on LODO NLL (+prior), curve re-learned in between
        x = th.vec()
        for rnd in range(1 if warm is not None else cfg["rounds"]):   # warm start: one refinement round
            th = Theta.from_vec(x)
            m, delta, _ = self._als(sums, th, self.gamma, days, wts, iters=3)
            loo = self._loo_cache(sums, delta, wts, days)
            dstd = dict(self.delta_std)
            packs = self._pack_loo(loo, days, tgt, self.gamma)
            obj = lambda v: self._lodo_nll(v, packs, days, wts, dstd) + _prior_penalty(v, self.mbar)
            x, fval = nelder_mead(obj, x, step=0.35 if rnd == 0 else 0.15, iters=cfg["nm_iters"])
        self.theta = Theta.from_vec(x)
        self.m, self.delta, self.neff_slot = self._als(sums, self.theta, self.gamma, days, wts, iters=5)
        self.wd_deltas = defaultdict(list)
        for d in days:
            if d in self.delta_std:
                self.wd_deltas[self.wd[d]].append((self.delta_std[d], wts[d]))
        self.info = {"bw": self.bw, "gamma": self.gamma, "n_days": len(days), "n_eff_days": self.n_eff_days,
                     "lodo_nll": fval, "theta": self.theta.asdict(), "mbar": self.mbar}

    def delta_posterior(self, observed, origin, weekday=None, cov=None):
        """(mean, var) of this direction's standardised day effect given observations up to `origin`.
        With no usable observations it is the prior: the caller's covariate mean then contributes zero."""
        th = self.theta
        if self.m is None or th is None:
            return 0.0, 1.0
        same = self.wd_deltas.get(weekday) if (th.phi > 0.0 and weekday is not None) else None
        prior = self.day_prior(th, self.n_eff_days, same, cov)
        series = []
        for t, y in observed:
            if self.lo <= t <= min(self.hi, origin):
                s = self._scale_at(self.m, t, self.gamma)
                series.append((t, (y - self._m_at(self.m, t)) / s, s))
        states = self.filter_states(series, th, prior)
        if not states:
            return prior[0], prior[1]
        st = states[-1][1]
        return st[0], st[2]

    # ---- prediction --------------------------------------------------------------------------
    def predict(self, origin, observed, targets, weekday=None, cov=None):
        th = self.theta
        if self.m is None or th is None:
            return [None] * len(targets)
        same = self.wd_deltas.get(weekday) if (th.phi > 0.0 and weekday is not None) else None
        tau0sq = self.day_prior(th, self.n_eff_days, same, cov)
        series = []
        for t, y in observed:
            if self.lo <= t <= min(self.hi, origin):
                s = self._scale_at(self.m, t, self.gamma)
                series.append((t, (y - self._m_at(self.m, t)) / s, s))
        states = self.filter_states(series, th, tau0sq)
        st, last = (states[-1][1], states[-1][0]) if states else (None, None)
        nu = nu_from_days(self.n_eff_days) if self.cfg["student"] else 1e9
        out = []
        for t in targets:
            mu, sd = self.predict_target(self.m, self.neff_slot, self.gamma, th, tau0sq, st, last, t)
            q = {k: mu + sd * (t_quantile(p, nu) if nu < 60 else _ND.inv_cdf(p)) for k, p in QLEVELS}
            q["mean"] = mu
            out.append(q)
        return out

    # ---- state export ------------------------------------------------------------------------
    def to_state(self):
        return {"direction": self.direction, "g0": self.g0, "m": [round(v, 4) for v in self.m],
                "neff": [round(v, 3) for v in self.neff_slot], "theta": self.theta.asdict(),
                "bw": self.bw, "gamma": self.gamma, "mbar": self.mbar, "m0": self.m0, "n_eff_days": self.n_eff_days,
                "n_days": len(self.days), "last_day": self.days[-1] if self.days else None,
                "r_hist": {d: round(v, 4) for d, v in self.r_hist.items()},
                "cov": {k: round(v, 6) if isinstance(v, float) else v for k, v in self.cov.items()},
                "wd_deltas": {str(k): [[round(a, 4), round(w, 4)] for a, w in v] for k, v in self.wd_deltas.items()}}


def _fill(m):
    """Fill NaNs by nearest finite value (flat extrapolation at the ends, linear in the gaps)."""
    idx = [i for i, v in enumerate(m) if v == v]
    if not idx:
        return [20.0] * len(m)
    out = list(m)
    for i in range(len(m)):
        if out[i] != out[i]:
            left = max((j for j in idx if j < i), default=None)
            right = min((j for j in idx if j > i), default=None)
            if left is None:
                out[i] = m[right]
            elif right is None:
                out[i] = m[left]
            else:
                f = (i - left) / (right - left)
                out[i] = m[left] * (1 - f) + m[right] * f
    return out


# --------------------------------------------------------------------------------------------------
# Public predictor (harness interface)
# --------------------------------------------------------------------------------------------------
DEFAULT_CFG = {
    "scale": "add",               # add | lvl | exc ; None -> choose from scale_auto by LODO
    "scale_auto": ("add", "lvl", "exc"),
    "bw_grid": (7.0, 10.0, 15.0, 22.0),   # Gaussian kernel sd, minutes (chosen by LODO)
    "min_day_obs": 3,
    "min_days": 3,
    "half_life_days": 60.0,       # exponential forgetting over weekdays (inert with < 60 days of data)
    "rounds": 2,
    "lodo_days": 60,              # held-out days used for hyper-parameter CV (cost cap; most recent days)
    "nm_iters": 100,
    "student": True,
    "weekday": False,             # weekday random effect (switched on automatically once every weekday has enough days)
    "windows": WINDOWS,
}


class StateSpacePredictor:
    name = "ss_additive"
    uses_context = False
    uses_other_direction = False
    cfg_overrides = {}

    def __init__(self, **overrides):
        self.cfg = dict(DEFAULT_CFG)
        self.cfg.update(self.cfg_overrides)
        self.cfg.update(overrides)
        self.dirs = {}
        self._rows = []

    def fit(self, history_rows, context=None):
        self._rows = [r for r in history_rows]
        self._fit_all(warm=None)

    def _fit_all(self, warm):
        for d in self.cfg["windows"]:
            f = _DirFit(d, self.cfg)
            f.set_history(self._rows)
            f.fit(warm=(warm or {}).get(d))
            self.dirs[d] = f

    def update(self, day_rows):
        """Online hook: append the finished day and refit with a warm start (theta from the last fit)."""
        self._rows.extend(dict(r) for r in day_rows)
        self._fit_all(warm=dict(self.dirs))

    def predict(self, direction, origin_minute, observed_today, target_minutes, observed_other=None, context=None):
        wd = context["weekday"] if context else None
        return self.dirs[direction].predict(origin_minute, observed_today, target_minutes, weekday=wd)

    def to_state(self):
        return {"version": 1, "cfg": {k: v for k, v in self.cfg.items()},
                "dirs": {d: f.to_state() for d, f in self.dirs.items() if f.m is not None}}

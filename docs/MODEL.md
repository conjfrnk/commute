# The "Predicted" model

**Predicted** is a probabilistic forecast of the drive time Google Maps will show for a departure time, with an 80% range. It is learned from your own commute and retrained every day with no action from you.

## What it models

For each direction, the drive time at time-of-day `t` on day `d` is

    y(d, t) = m(t) + delta_d + x_d(t) + noise

- `m(t)`: a smooth time-of-day curve learned from all previous days (a kernel smoother whose width is chosen by cross-validation; older days fade slowly).
- `delta_d`: a persistent *day effect*: some days are slow or fast as a whole.
- `x_d(t)`: a mean-reverting wiggle within the day (traffic drifts, then returns).
- `noise`: whole-minute rounding plus a small learned nugget.

A Kalman filter tracks `delta_d` and `x_d` from today's samples (including the hour before the window opens). Its gain *is* the "how much of today's deviation carries forward" weight: it is learned and fades with horizon, so no weights are hand-set. Each forecast is Gaussian, with a standard deviation that includes the uncertainty of the curve itself.

**Weekday effect.** Once every weekday has at least two days of history, part of the day effect is attributed to the weekday (partial pooling: Monday's effect is estimated from earlier Mondays and shrunk toward zero when there are few). The prior on that share is deliberately wide (centred at 0.3, log-odds sd 1.5).

**Day-effect covariates.** On top of the weekday prior the model learns three covariates from the residual day effects, refit nightly by ridge-shrunk regression: each coefficient starts near zero and earns weight only as the data support it, is capped below 1, and stays off until a stream has 10 days of history.

- *Sibling direction*: the evening forecast learns from the same morning's to-work deviation; the morning forecast learns from the previous day's to-home residual.
- *Own yesterday*: day effects run in streaks (gaps over 3 days are treated as too stale to use).
- *Rain* (opt-in): the day's precipitation in mm from Open-Meteo, fetched nightly and stored in the model state so the poll path stays offline. It needs `weather_lat` and `weather_lon` in `commute_settings.json` and stays off until at least three days with 1 mm or more exist.

The leftover residual variance replaces the plain innovation variance in the prior (floored at a quarter of it), so covariates can only narrow the day prior, never widen it. A missing input degrades term by term to the plain prior. Scoring uses observed rain while a live forecast only has forecast rain, which flatters wet-day scores slightly.

**Days off.** Weekends, holidays and the dates you list are separate *streams* (`to_work@off`, `to_home@off`) with their own curve, hyper-parameters, calibration and ledger, using the same windows. They never feed the weekday curve and weekday samples never feed them. Each stream earns its label independently, so days off show "Typical (learning)" (the simple curve over earlier days off) for the first few weeks.

## How it trains itself

After each day's last window the sampler starts `commute_model.py --sync` as a separate process (with a catch-up after sleep or downtime):

1. **Score first.** The finished day is forecast with the model that existed *before* it (from the morning and every 30 minutes through the window) and compared with what happened. That yields one ledger row per day and direction: average miss, how often the real value fell in the 80% range, and an interval score. Because the day is scored before it is learned, the ledger is out-of-sample by construction.
2. **Recalibrate.** The probability-integral-transform value of every realized drive time goes into a slowly forgetting histogram per direction and horizon group. If real values land in the tails more or less often than the model claims, the issued quantiles are bent to match (Kuleshov et al. 2018). With no evidence it changes nothing.
3. **Learn the day.** The day is added; the curve and hyper-parameters (day-effect size, wiggle size, persistence, nugget, smoothing width) are re-fitted by leave-one-day-out likelihood from prior days only, warm-started from the previous night. The covariates are then ridge-fit on the refreshed residuals and, if enabled, the rain table is refreshed.
4. **Check itself.** A fallback flag compares the model's recent interval score with a simple 30-day-average model's. If the model is more than 25% worse over a week, the app shows the simple one until it is within 10% again.

The job is idempotent: the state file records the last day folded in, a day is never counted twice, a damaged state file falls back to the previous good one and then to a rebuild from the CSV, changed windows or days off trigger a rebuild, and the job never blocks sampling.

## What the label means

| Label | Meaning |
|---|---|
| Collecting data | fewer than 3 days of history; the simple curve is shown |
| Typical (learning) | not yet scored on about 5 days; the simple curve is shown |
| Predicted (calibrating) | the model is shown; its intervals are still correcting themselves |
| Predicted | about 15 scored days and realized 80% coverage consistent with 80% (stays on unless calibration is clearly broken) |
| Typical (fallback) | the model scored clearly worse than the simple curve; the simple one is shown |

The accuracy line under the graph is computed from the ledger.

## Limits

- It forecasts what Maps will say at that time, not the realized trip, and it does not foresee incidents. It is wrong about 1 time in 5 *by design* at the 80% level.
- The effective sample size is days, not samples, so claims firm up over weeks: expect "Predicted" without "(calibrating)" after about 15 scored days.
- Seasons, school terms and events are not modelled. Weekday effects are noisy until each weekday has several days; weekend and holiday forecasts need a few weekends before their label moves past "Typical (learning)".
- A 90% range is not offered: verifying it takes on the order of 60 days.

## Files

- `commute_statespace.py`: the engine (Kalman day effect plus mean-reverting wiggle, hyper-parameter fitting). Standard library only.
- `commute_model.py`: calibration, ledger, labels, fallback, persistence and the nightly job (`--sync`, `--status`).
- `commute_data.py`: reads the state at poll time (about 1 ms) and builds the snapshot; falls back to the simple curve whenever the model is unavailable.
- `commute_model_*.json`: learned state, one per route (git-ignored; rebuilt from the CSV at any time).

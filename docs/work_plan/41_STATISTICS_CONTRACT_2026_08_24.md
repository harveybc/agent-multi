# 41. Statistics Contract for Screen Selection (SOTA-R07 / C4)

Authority: Musashi SOTA return correction order 2026-08-24. Binds every
DSR/SPA/IQM/bootstrap procedure named in docs 38 §23 and 40 BEFORE any
screen runs. Deviations require a predeclared amendment here, never a
post-hoc choice.

## 1. Return objects

- **Unit series**: per-bar simple net returns of the arm's equity curve
  at H4 cadence, one value per scored bar (2,190 bars per scored
  calendar year; 2,196 for 2024), net of the screen's full cost engine.
  Retained verbatim in each arm return (doc 40 shared contract).
- **Loss differential for SPA**: d_t = r_t(candidate) − r_t(benchmark),
  per bar, on the identical scored index. Benchmark per screen: Screen
  B G1 uses the best rule arm; other screens use their predeclared
  control (A0: sign mapping; R: R0 frozen; C: C1 flat MLP).
- **Annualization**: factor sqrt(2190) on per-bar Sharpe. Nominal T is
  per-bar count; the dependence corrections below, not naive T, carry
  the inference.

## 2. Dependence handling

- **Bootstrap**: stationary bootstrap (Politis-Romano). Expected block
  length chosen by the Politis-White automatic rule computed on the
  CONTROL ARM'S OWN PER-BAR NET RETURN SERIES ALONE — a single series,
  no candidate data and no differential enters the computation
  (SOTA-C06). Computed once per screen, before any candidate outcome is
  examined; the chosen length is logged in the screen's aggregation
  config and reused for every candidate in that screen.
- **Resamples**: B = 10,000; RNG seed for the bootstrap = 20260824
  (fixed, logged).
- **SPA**: Hansen (2005) studentized statistic with the sample-dependent
  null; consistent p-value reported (lower/upper also logged). RC
  (White 2000) may be reported alongside, never substituted.

## 3. Trial counting (DSR)

- **Prospective ledger**: from 2026-08-24 every config × seed ×
  calibrated-threshold × lookback × screen row is one trial, recorded
  in the OLAP cube at materialization time (not at result time).
- **Historical count**: the reconstructed pre-ledger count from OLAP +
  campaign records is a DOCUMENTED LOWER BOUND, labeled
  `n_trials_lower_bound`, until an explicit completeness audit passes.
  DSR computed with a lower-bound N is an UPPER bound on significance
  and is labeled as such wherever reported.
- **Correlated trials**: effective-N is not estimated implicitly; we
  report DSR at (a) raw N and (b) arm-level N (seeds of one arm
  collapsed), bracketing the truth. Any single-number claim uses (a),
  the conservative side.
- **DSR inputs**: observed SR (per-bar, annualized as §1), T = scored
  bars, skewness and excess kurtosis of the per-bar series, variance of
  SR across trials from the ledger.

## 4. Seeds, folds, aggregation

- **Primary comparison**: paired per-seed, per-origin differentials
  (candidate − control), aggregated as the interquartile mean (IQM)
  across the seed × origin grid.
- **Intervals**: stratified bootstrap CIs (strata = origins, resampling
  seeds within origin), percentile method, B as §2, alpha = 0.05.
  rliable-style performance profiles are attached when arms ≥ 3.
- **Vote rules in doc 40 (≥2/3 origins, ≥3/4 seeds)** are screening
  gates, not inference; the IQM + CI + SPA/DSR block is the evidence a
  selection claim stands on.
- **Overlapping folds**: the three development origins score disjoint
  years (2022/2023/2024), so no overlap correction applies; if a future
  screen uses overlapping windows it must amend this section first.

## 5. Reporting floor

Every screen aggregation publishes: per-arm per-bar series hashes, the
block length and its computation inputs, B, bootstrap seed, SPA
p-values (consistent/lower/upper), DSR at both N conventions with the
lower-bound label where applicable, IQM + CI tables, and the trial
ledger extract for the screen. Absence of any element blocks the
screen's decision claim.

# 270. Causal STL selected-input side lane

Owner approved2026-11-10. Separate from the selected69 critic-ranking repair
after269. No changes to the champion, phase2 selection ledger or live model.

## Hypothesis and limits

Replace EACH of69 encoded inputs with its STL seasonal component; discard
trend and remainder only in these experimental arms. Neither the remainder
being pure noise nor the seasonal component being stationary is assumed.
Slow economic/regime features may lose most useful information. Keep them in
this hypothesis test and report their variance/seasonal evidence, not silently
drop or substitute them. No synthetic market rows, padding or imputation.

## Causal contract

Use statsmodels STL on a trailing finite history ending at the CURRENT CLOSED
bar and emit ONLY the last seasonal value. Never decompose the full dataset
and slice its seasonal output. Never recompute a past emitted value when later
bars arrive. Feed windows contain the previously emitted causal values.
History length=max(64,4*period), no revised past values, no centered future
samples. Missing input invalidates that column's endpoint until its required
history is finite again. No interpolation. Observation clock matches existing
row-stride windows: periods count observed admitted bars, not wall-clock hours;
weekend gaps are not filled. This is explicitly not a calendar-time seasonality
estimator. Timestamp/data availability and existing feature encodings remain.

Hyperparameters and scaling are fitted in TRAIN only and frozen for validation
and inference. Causality tests hold fitted constants fixed, as with the existing
TRAIN scaler. A separate test perturbs validation and verifies fitted parameters
do not change. Live use requires the same encoded input history and serialized
transform; raw-input loaders must reject a transformed recipe without its adapter.

## Per-column parameter selection

Candidates: period={6,12,24,48,120} observed bars, seasonal smoother={7,13},
robust={false,true};20 candidates. Use statsmodels' documented odd trend/low-pass
defaults and degree1; jumps all1 (no approximation). Context=max(64,4*period).
Select using16 equally distributed eligible TRAIN origins, each with the same
24-observation forward scoring interval, all wholly within TRAIN. Fit history
only at each origin. Forecast by repeating the last seasonal cycle plus a
linear extrapolation of the fitted trend over its last period. Minimize mean
absolute error normalized by the common max-context history's absolute first
difference. Report trend-only and persistence errors, residual seasonal strength
and periodic autocorrelation; these diagnostics do not claim predictive trading
value or independently proven noise removal. Deterministic ties favor shorter
period, smaller smoother and non-robust fit. No validation/trading-score tuning.
If no eligible origins exist, fail with the column identity; never invent data.

## Training contrast

Four full fresh discrete-SAC fits, seed7 initially, same69 groups/TCN/attention
architecture and current effective hyperparameters, raw vs STL continuous vs
STL five TRAIN-quantile bins vs STL binary sign. Fresh weights/replay for ALL
arms: existing raw-trained encoders cannot silently serve as a fair seasonal
initialization. Keep original raw TRAIN scaler; seasonal continuous uses new
fixed TRAIN constants. Ordinal codes are[-1,-.5,0,.5,1], binary codes0/1,
neither subsequently standardized. Duplicate quantile edges are reported.

Same original four TRAIN years and2024 validation, original13xstride2 model
window, native risk/cost/calendar, no state inputs. Common decision support is
the intersection of original validity and finite causal seasonal windows.
Report every warm-up/missing-history exclusion and all calendar months; raw
control gets precisely that support, so filtering cannot win by changing dates.
Full native account periods are retained, with existing automatic holds when
an observation is unavailable. These holds are not zero-valued observations.
The historical champion is contextual only, not the matched fresh control.

Use existing epoch trainer:2000 maximum complete TRAIN traversals, patience20,
min_delta1e-8, validation return/week, select eligible checkpoints from epoch1
while delaying patience increments through20. Strictly>2 orders/week in both
periods required for checkpoint eligibility. No eligible model means INVALID,
not promotion of a zero-order epoch. Save weights, replay recovery, effective
config and TRAIN/validation diagnostic plots. Binary/ordinal remain separate
ablations, not silently adopted on better TRAIN alone. No automatic promotion.

## Gates and allocation

| Gate | Evidence |
|---|---|
| STL1 | Prefix equality, future corruption, stream/batch identity |
| STL2 | No imputation; insufficient history/NaN remains unavailable |
| STL3 | TRAIN-only tuning/scaling/quantiles; validation mutation invariant |
| STL4 | Shared support, original architecture/window, artifact identities |
| STL5 | Export/reload logits, transformed-recipe raw-loader rejection |
| STL6 | Activity-aware monitor, complete account metrics/plots, resumable queue |

CPU column decomposition is restartable and distributed. GPU training follows
preprocessing/causality gates automatically. Reserve the fastest GPU first for
the short main-lane critic-only contrast, then give it side-lane training;
secondary GPU runs the other queue concurrently. Native input transformations
are CPU STL, not falsely advertised as GPU-accelerated. Per-column timings
determine preprocessing ETA; epoch timings determine training ETA.

Reference: https://www.statsmodels.org/stable/generated/statsmodels.tsa.seasonal.STL.html

## Execution record

Runtime a0c8f542, identity
`04fb822af81df2708dae1fbf7bc1105e152760244d757e5d76a734b3ac6996b8`.
Five persistent CPU queues launched2026-11-10T21:11Z across three workers.
45 focused CPU tests pass, including common-support/stride parity, transformed
checkpoint cold reload and a finite SAC update. Both remote consumers have the
pinned runtime and data; secondary source verification returns30859x13x69.
First three completed columns took113/186/173 seconds on their assigned CPUs;
other robust fits are substantially slower. No trading result exists yet.

Automatic dependency chain: all69 column receipts -> assembly/stream parity ->
per-month common-support report (zero-availability month refuses training) ->
two sequential GPU queues. Primary RTX5090: raw then ordinal, after271 completes.
Secondary RTX4090: continuous then binary. Collector verifies hashes and gathers
metrics, weights and plots, excluding multi-gigabyte mutable replay copies.
Services survive a host restart; column checkpoints resume every256 rows and
training restores committed epochs. Three failed automatic starts in one hour
stop that service visibly rather than silently accepting an incomplete result.
No changes to live trading, champion weights or the phase2 inclusion ledger.

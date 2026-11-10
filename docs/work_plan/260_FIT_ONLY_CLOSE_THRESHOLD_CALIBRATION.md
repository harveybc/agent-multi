# Fit-Only Close Threshold Calibration

Owner authorized after259. Freeze every archived head and score. Do not update
actor, encoders, features, native order mechanics or the phase2 evaluator.
This CPU-only diagnostic requires no new model training or GPU allocation.

## Fixed Design

Reuse all24 archived geometric-interface endpoints:2 interfaces x2 seeds x2
market contexts x3 objectives (SmoothL1, SIGN, SIGN_PAIR). Primary contrast is
surface81 SIGN_PAIR with actual market, both seeds, one common threshold.
Directional thresholds are a declared secondary diagnostic, not an option
chosen afterward from monitor results. Keep all constant-market controls.

For score z and native account gap y=Q(CLOSE)-Q(HOLD), choose threshold t by
minimizing bank-balanced fit regret E[max(y,0)-y*1(z>t)]. Use every fit bank
from2020-2022, never2023 monitor labels. Baseline t=1e-8 reproduces the existing
zero-score-with-tolerance decision. Ties at the threshold always HOLD.

Sort the unique fit scores and exhaustively evaluate all decision partitions,
including all-CLOSE and all-HOLD. No DEAP or arbitrary numeric grid is needed.
Within a partition, choose the representable threshold nearest the baseline;
among losses within1e-12 of the minimum choose the threshold nearest baseline,
then the smaller threshold if distance ties. Explicitly persist the tolerance.
Report zero/all-close solutions without disguising them as useful calibration.
Do not invent an activity constraint to change this diagnostic's optimum.

The comparator is float64 `score > threshold` on the stored mean-twin float32
score converted to float64. Do not subtract an offset and round back to float32:
that can change tie behavior. These thresholds do not calibrate probabilities.

Common calibration uses all fit banks. Directional calibration solves the same
problem independently in each direction, preserving within-direction bank
weights. Each returned threshold is sealed before ANY monitor evaluation.
Fit-window records only enter the calibrator; perturbing all monitor scores
and labels must leave the result unchanged.

## Tests And Evidence

| Requirement | Test/evidence |
|---|---|
|Global empirical optimum|Randomized brute-force enumeration against sorted-prefix solver|
|Ties, neutral gaps, duplicate scores|Keep baseline on equal utility; strict comparison and serialization|
|Correct bank weighting|Unequal bank sizes cannot silently become row weighting|
|No monitor fitting|Replace monitor scores/labels and reproduce calibration exactly|
|No direction mixing|Shared baseline vs separate directional fit fixtures|
|No source/model modification|Authenticate all parent collections before and after|
|No metric substitution|Same native account regret, local-close harm and conflict strata|
|Honest promotion status|Diagnostic only; no actor transfer or new portfolio Sharpe|

Write the72 baseline/common/directional evaluations, selected thresholds, fit
objective, all candidate cuts, conflict strata and the existing critical witness.
No recalibration after reading monitor. Check baseline parity against archived
metrics. Publish all fixed-endpoint results, not only the favorable seeds.

## What Follows

1. Decide from calibration evidence whether an integration contrast is justified.
   The new market-only close fields are not yet the live/floating-selection
   actor. If justified, keep original entries/risk and compare native sequential
   accounts with/without the external closing controller, including positions
   reached only under the changed policy, fees, barriers, calendar, activity,
   long/short behavior, full metrics and unchanged-policy control replay.
2. Preserve the existing selected69 actor (28 groups), checkpoint and exact
   compaction evidence from225. A frozen selection pass on it remains possible
   independently; it would not incorporate these diagnostic close fields.
3. Once an evaluator is fixed, run the next removal/reinsertion pass with
   component weight-space/activation diagnostics, cumulative phase2 reinsertion
   pool, unchanged scoring contract and exact functional export checks. Follow
   with a separate matched refit/refinement; do not conflate masking results
   with freshly learned weights.
4. In parallel, audit historical source coverage per retained feature and its
   ancestors, event availability, gaps, timezone and fitted producer artifacts.
   More price history alone does not extend macro/cross-asset availability.
   Measure real-data learning curves before declaring data volume the cause.
5. Synthetic augmentation is a separate controlled experiment, not admission
   by default: fit generators on TRAIN only, preserve joint temporal/asset
   dependence, generate coherent source paths and causally recompute features,
   carry generator warm-up/state, and exclude splice-boundary artifacts.
   Compare real-only against real+synthetic with matched optimizer budgets;
   select/evaluate on real periods only. Do not independently synthesize each
   indicator or fill historical gaps with invented observations.

The proposed data order is an engineering choice, not a claim that more data
cannot help. TimeGAN explicitly evaluates temporal/predictive fidelity using
train-on-synthetic/test-on-real sequence prediction; that does not establish
profitability or a valid joint generator for this RL stack. Quant GANs study
financial return generation and temporal dependence, likewise not the complete
selected-feature contract. Useful methodological references, not a recommendation
to replace the present pipeline without qualification:

- [TimeGAN, NeurIPS2019](https://proceedings.neurips.cc/paper/2019/file/c9efe5f26cd17ba6216bbe2a7d26d490-Paper.pdf)
- [Quant GANs](https://arxiv.org/abs/1907.06673)

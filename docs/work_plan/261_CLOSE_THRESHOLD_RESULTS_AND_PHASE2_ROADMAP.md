# Close Threshold Results And Phase2 Roadmap

2026-11-10. Owner-authorized [design260](260_FIT_ONLY_CLOSE_THRESHOLD_CALIBRATION.md)
completed. CPU only,4.37 seconds,24 frozen endpoints,48 calibration variants,
72 baseline/common/directional evaluations,84180 fit decision partitions.
No retraining, no actor change, no new data, no feature removal in this run.

## Primary Result: Do Not Adopt The Calibration

Primary comparison fixed before execution: surface81 SIGN_PAIR, actual market,
both seeds. All thresholds fit solely on2020-2022 banks;2023 is the internal
monitor. Source encoders previously used2023. These are diagnostic continuation
decisions, not a new trading portfolio or an independent generalization test.

| Calibration | Mean FIT regret | Mean monitor regret | Harmful closes | Beneficial closes | Total effective closes |
|---|---:|---:|---:|---:|---:|
|Baseline1e-8|0.140247|0.128891|241 (39.64%)|367|608|
|Common fit-only|0.136627|0.130963|300 (42.98%)|398|698|
|Directional fit-only|0.135060|0.129488|295 (42.45%)|400|695|

Counts are summed over two seeds on the same counterfactual monitor positions;
they are not independent new orders or portfolio profit. The exact empirical
fit optimum improves FIT, as expected, but worsens monitor regret and harmful
close counts. Both calibration modes worsen primary monitor regret in each
seed, not just on their mean. Retain baseline thresholds and all artifacts.
This does not invalidate the paired-loss improvement from259; it rejects this
additional intercept adjustment as an improvement for the primary comparison.

### Per-Seed Thresholds And Regret

| Seed | Common threshold | Long threshold | Short threshold | Monitor baseline | Common | Directional |
|---|---:|---:|---:|---:|---:|---:|
|19|-0.1050759107|-0.1185349077|-0.1531287730|0.130983|0.133743|0.131305|
|43|-0.1498466134|-0.1498466134|-0.0300577283|0.126798|0.128183|0.127671|

The primary critical witness still fails: seed19 changes HOLD,HOLD,HOLD to
CLOSE,CLOSE,CLOSE; seed43 remains HOLD,HOLD,HOLD. Required: CLOSE,HOLD,HOLD.
Correct relative ordering alone did not imply a fit-derived global or
directional offset would resolve it. No threshold was adjusted afterward to
force that monitor example to pass.

Across all24 endpoints, the witness passes in4 common and6 directional variants,
versus0 baselines. These are not the primary surface81 SIGN_PAIR cases. In
particular, progress9 SmoothL1 passes in both seeds after calibration while its
aggregate monitor regret worsens. Passing one witness is not global acceptance.

## Full Market-Input Comparison

All values are two-seed mean bank-balanced monitor regret; lower is better.
No arm is omitted or selected from this table for deployment.

| Interface | Objective | Baseline | Common | Directional |
|---|---|---:|---:|---:|
|progress9|SmoothL1|0.133868|0.140396|0.135262|
|progress9|SIGN|0.131140|0.141991|0.137953|
|progress9|SIGN_PAIR|0.132324|0.134257|0.132173|
|surface81|SmoothL1|0.132305|0.133428|0.134916|
|surface81|SIGN|0.133515|0.132737|0.133293|
|surface81|SIGN_PAIR|0.128891|0.130963|0.129488|

Constant-market surface81 SIGN_PAIR also worsens: baseline0.129421,
common0.133488,directional0.132017. Complete constant-market comparisons,
per-seed counts, direction-specific metrics and conflicting/nonconflicting
strata are retained. Small improvements in other arms are descriptive; no
post-hoc switch of the predeclared primary hypothesis or threshold rule.

## Method And Verification

The calibrator sorts the unique fit scores, evaluates every possible strict
threshold partition (including all-HOLD/all-CLOSE), and minimizes the same
bank-balanced native account regret used in259. Ties within1e-12 prefer the
threshold closest to the original1e-8, then the smaller threshold. Separate
directional thresholds use the same conditional bank weighting. All complete
candidate curves are retained outside the repository.

Each threshold uses the direct float64 comparison `score > threshold` on the
stored mean-twin score, avoiding a subtract-and-round conversion. Ties HOLD.
Calibrations are written and their index hashed before monitor evaluation.

-28 focused tests pass, including randomized exhaustive-search parity, unequal
  bank weighting, duplicate scores, neutral labels, exact ties, JSON reload,
  directional behavior and monitor perturbation invariance.
-48 real calibration perturbation controls replace all monitor scores/labels;
  every resulting fit calibration remains identical.
-All24 uncalibrated baselines reproduce archived metrics, including each
  direction and in/out-of-support slice. No source head was changed.
-All parent collections authenticated before and after execution;55 output
  files verified. No GPU or optimizer update. No failed execution/retry.
-The source/runtime commit is b3220476. Run identity:
  `9d04a479b4d2203f73a4924cfb4f8bcc29005e2235a94e03a4c85585e9794bbe`.

Evidence: [summary](generalization/close_threshold_20261110/SUMMARY.csv),
[all endpoints](generalization/close_threshold_20261110/ENDPOINTS.csv),
[strata](generalization/close_threshold_20261110/STRATA.csv),
[threshold index](generalization/close_threshold_20261110/CALIBRATION_INDEX.json),
[witness](generalization/close_threshold_20261110/WITNESS.json).
These compact public copies are subsets of the full sealed55-file collection.

## Recommended Next Work

The calibration is complete and is not adopted. Do not keep optimizing this
offset against the monitor. Distinguish the established inference candidate
from the experimental critic/close-field line:

1. **Next phase2 pass:** return to the preserved69-input,28-group actor from225,
   with unchanged dataset, scaling, windows, native strategy, activity gate and
   selection score. Run ordered deletion then reinsertion and component-level
   weight/activation diagnostics. This does not require promoting the new close
   fields or waiting for every critic research question to be resolved.
2. **Restore candidates correctly:** establish the historical phase2 reinsertion
   inventory and donor checkpoints before evaluating it. A previously removed
   branch needs compatible recipe, scaler and weights; do not replace it with
   a fresh random branch and call it a frozen-input ablation. Phase1 exclusions
   require a separately justified rescue test, not automatic readmission.
3. **Preserve any gain:** physically export selected branches/weights and prove
   inference/account parity with the masked result before further refinement.
   Separate frozen-selection improvement from a fresh/warm training result.
4. **Keep critic research separate:** refinement can still damage the source
   policy, and the experimental position-field controller has not been qualified
   in full sequential accounts. Those are real remaining research tasks, not
   reasons to discard the preserved actor or prevent a frozen phase2 pass.
5. **Audit data coverage in parallel:** determine the earliest usable history for
   every retained input and producer ancestor, including cross-asset and macro
   publication time, gaps, timezone and frozen learned producers. No conclusion
   has been reached that longer joint history is unavailable. Compare real-data
   learning curves before diagnosing sample count as the limiting factor.

The next phase2 run is recommended, not launched by this calibration command.
The selected69 artifact remains
`8ab11f607dc18c53ce45b34dc628e21272767702d60e0bfbe272620761de4e40`;
its original70-column control and forward-lane artifact remain unchanged.

## Synthetic Data: Reuse Work, Qualify Compatibility

Read-only inspection found the existing predictor phase4 configurations:
`examples/config/phase_4_1/phase_4_1_transformer_1h_12600_config.json`,
and corresponding phase4_2/phase4_3 configs. They reference synthetic-prepended
training files and real validation/test files. Their presence is useful prior
engineering, not evidence that their generators reproduce today's69-feature
joint input process or are appropriate for the RL/native simulator contract.

The older predictor handoff
`docs/handoffs/MUSASHI_SYNTHETIC_CATALOG_AND_FOUR_CONSUMERS_ACCEPTANCE_2026_09_14.md`
explicitly certifies schema/transport, not financial indicator semantics or
chronologically independent partitions. Do not silently admit that catalog as
new RL training history. No sibling code, data or services were changed here.

Prefer an audited extension of real joint history where available. If synthetic
augmentation is tested, reuse relevant phase4 components after checking their
fit boundary, generator, units and multivariate temporal semantics. Recompute
all derived features causally from coherent source paths; do not fabricate
independent indicators or append segments without boundary/state treatment.
Use real-only versus real+synthetic controls with matched training budgets and
real-period evaluation. Synthetic samples are augmentation, not newly observed
market history. References and the proposed acceptance criteria are in260.

This ordering leaves the data question open to measurement. Nothing here
establishes that input signal is absent, that more data will necessarily help,
or that the remaining system issues can be fixed by increasing dataset size.

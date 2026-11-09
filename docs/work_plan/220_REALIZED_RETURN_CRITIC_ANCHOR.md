# Realized Return Critic Anchor

Continuation of [219](219_FULL_TRAIN_CRITIC_RESULTS.md), not a trading-model
promotion. Determine whether target feedback rather than head trainability
maintains the extreme negative calibration bias.

Restore epoch40 and its optimizer. Freeze actor, temperature and all encoders.
Use the complete authenticated native TRAIN replay from218. The only learning
change is the critic target: realized discounted soft return for the sampled
action, computed from the frozen policy trajectory, replaces the bootstrap.
These retrospective TRAIN quantities are target labels only, never actor inputs.
They are neither ZigZag demonstrations nor additional observation features.

Use batch128 and4096 updates with a covering permutation schedule. Its first1024
updates must exactly match the completed full-TRAIN comparison schedule. Compare
the anchor at1024 against the two existing controls at equal budget; report2048
and4096 as extended-budget diagnostics, not equal-budget superiority claims.
Measure all TRAIN rows under action seeds7/19/43 at0/1/64/256/1024/2048/4096.
Only seed7 labels enter training. No validation or test outcomes enter fitting.

## Acceptance Tests

- Labels must be finite floating tensors with one scalar per transition; detach
  them from autograd and require terminal labels to equal terminal rewards.
- Fail if the return field is absent. No silent fallback to a bootstrap target.
- Compare cached replay and source-control identities, support and schedule
  prefix; missing or changed evidence prevents execution.
- Freeze and verify actor parameters, alpha and encoders at each measurement.
- Persist full residual summaries, calibration bins, elapsed time and diagnostic
  critic checkpoints at1024 and4096. Never overwrite the selected model.

If calibration remains poor, inspect critic feature/target compatibility and
optimization before changing actor capacity. If it improves, test a guarded
joint-training correction separately: restoring calibration for a fixed policy
does not alone establish a stable SAC learning recipe or profitable strategy.

## Calibration References

A constant-zero Q predictor has TRAIN soft-return RMSE0.295424,0.279264 and
0.293172 for action seeds7/19/43. Using the seed7 mean return as a constant gives
0.293706,0.277302 and0.291706. These are value-calibration references, not trading
policies. Removing a large offset alone does not establish useful action ranking.

## Launch Receipt

Runtime `4ce3eba7`; identity
`085b3c5b79768fea34f813693b59be3699de00af96e494f3c5cb9c70945c4305`.
Launched on RTX5090 at17:24 UTC on2026-11-08. Source-control files authenticated,
all24025 TRAIN rows covered, initial three-trajectory calibration reproduced,
and the first256 updates completed with actor and encoders unchanged. The source
replay is reused read-only. Focused tests:33 passed, one CUDA-only test skipped.
This receipt does not assert the4096-update endpoint has completed.

## Verified Completion

The worker completed at2026-11-08T17:28:52.935828Z (12:28:52 America/Bogota),
in262.264 seconds. All six files listed in DONE.json, including both critic
checkpoints, were collected and independently checked against their SHA256
digests. Actor and frozen-parameter equality hold at every recorded checkpoint.

| Updates | TRAIN critic bias, seed7 | TRAIN critic RMSE, seed7 |
|---:|---:|---:|
| 0 | -25.021730 | 25.525045 |
| 256 | -21.563786 | 22.132774 |
| 1024 | -14.637509 | 15.427805 |
| 2048 | -8.224373 | 9.448890 |
| 4096 | -2.474531 | 5.006720 |

At4096, alternate action trajectories19/43 have RMSE5.008122/5.001600.
These are different action draws over the same TRAIN period, not independent
market years. The endpoint remains worse than the constant-zero reference
(seed7 RMSE0.295424). Reduced offset is therefore not sufficient calibration,
and no trading-model promotion or generalization improvement follows.

Interpretation: the critic heads can move substantially when trained against
realized returns, unlike the earlier bootstrap controls at equal1024-update
budget. The extended4096 result does not isolate why residual error remains.
Next diagnostic design: separate retained optimizer/head initialization from
feature-to-target fit on the same fixed replay, with actor/encoders frozen and
constant predictors as explicit controls. No successor run was launched by
this completion review; the selected epoch0 policy remains unchanged.

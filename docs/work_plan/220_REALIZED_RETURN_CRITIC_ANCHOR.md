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

# Full TRAIN Critic Confirmation Results

Completed 2026-11-08 at16:59:15 UTC in250.36 seconds on RTX5090.
Protocol: [218](218_FULL_TRAIN_CRITIC_CONFIRMATION.md).
Runtime `bf44f2b9`; identity
`7a82b3f3f1a963e9a09bbf560a9350965edf7b8df11080a4af2b34e75baeb754`.
Eight terminal files independently authenticated after collection.

All24025 TRAIN2020-2023 transitions were reconstructed in the native engine with
zero reward discrepancy, matching account P&L and trade count, one terminal,
unchanged windows and features. Every transition appeared in the common update
schedule. Both arms used1024 updates of batch128, restoring epoch40 and freezing
actor, encoders and temperature. Initial calibration matched all three sealed
source trajectories. Actor logits and frozen parameters remained exact.

## Results

Calibration is measured on all24025 decisions per trajectory. Bias is predicted
minimum-twin Q minus realized discounted soft return, in percentage-point reward
units relative to initial capital, NOT account profit or annual return.

| Variant | Seed7 bias | Seed7 RMSE | Seed19 bias | Seed19 RMSE | Seed43 bias | Seed43 RMSE |
|---|---:|---:|---:|---:|---:|---:|
| Initial epoch40 | -25.021730 | 25.525045 | -25.020402 | 25.523861 | -25.024437 | 25.526793 |
| Minimum target | -25.112238 | 25.619132 | -25.110916 | 25.617956 | -25.114949 | 25.620874 |
| Mean target | -23.577465 | 24.070096 | -23.576064 | 24.068790 | -23.579844 | 24.071294 |

Seed7 supplies the training trajectory; seeds19/43 are different action draws
on the same market dates. They are not independent model fits or held-out years.
No statistical significance is claimed. Mean targets again reduce pessimism
relative to the paired minimum-target control, but substantial bias remains.
The mean realized seed7 soft return is -0.031811, compared with predicted mean Q
-25.144048 in production and -23.609276 under mean targets.

This confirms the short-test direction with complete TRAIN coverage. It does
not prove that the entire historical drift has a single cause, nor that mean
targets are ready for joint training. In this diagnostic the actor never
changes, so no improved trading performance is claimed. The selected epoch0
model and its previously verified +13.8085% VALIDATION account remain intact.

The current artifact contains means, RMSE, MAE and calibration bins, but not
individual residual quantiles promised by the broad protocol. The next contrast
adds those summaries and persists critic checkpoints; no missing tail statistics
are inferred from the aggregate bins.

## Next Diagnostic

Use the same frozen policy and full TRAIN replay, replacing bootstrapped targets
with its authenticated realized soft returns. This tests whether the existing
critic heads can recover calibration without recursively fitting their own Q
estimates. Keep all other controls fixed, record an equal-budget1024-update
comparison, and extend to4096 updates as a separately labeled diagnostic budget.
Do not resume actor refinement or floating selection solely on the present
partial improvement. Design: [220](220_REALIZED_RETURN_CRITIC_ANCHOR.md).

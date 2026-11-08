# Fixed Actor Critic Audit Results

Date: 2026-11-08. Protocol: [216](216_FIXED_ACTOR_CRITIC_AUDIT.md).
Preceding full-account diagnosis: [215](215_FROZEN_POLICY_RETURN_CALIBRATION_RESULTS.md).

## Scope

These are critic-only diagnostic interventions, not new account strategies or
candidate models. The selected epoch0 actor and its +13.8085% VALIDATION account
remain unchanged. No refinement or feature selection has resumed.

The five-arm test reuses 4096 authenticated TRAIN transitions, 1024 paired
updates per active arm, batch128, and a fixed512-decision diagnostic panel spread
over TRAIN2020-2023. It is a bounded mechanism test, not full-data training. The
panel uses frozen-policy realized soft returns, not validation fitness. A single
sampling seed and temporally dependent decisions do not support significance
claims or certainty about long-run effects.

## Arithmetic and Implementation Checks

Real epoch0/6/40 checkpoints were checked on native replay, forced-terminal
fixtures and duration4 fixtures. Maximum independent target discrepancy was
`1.52587890625e-05`; maximum reconstructed critic-loss discrepancy was
`2.384185791015625e-07`. Terminal bootstrap exclusion passed. Actor loss produced
zero critic gradient and critic loss produced zero actor gradient. Checked
Polyak updates matched their declared formula exactly. These checks found no
sign, duration, loss-reduction, gradient-leak or target-update implementation
error in the inspected cases; they do not certify every possible trajectory.

Actor logits and frozen parameters remained exactly unchanged in the paired
tests. The no-update control was invariant. No optimizer state or original
checkpoint was overwritten.

## Epoch0 Single-Factor Test

| Arm | Mean online Q after1024 | Panel Q-minus-return bias | Panel RMSE |
|---|---:|---:|---:|
| No update | 1.167031 | 1.082869 | 1.291207 |
| Production | 1.074794 | 0.984257 | 1.205754 |
| No core dropout | 1.074517 | 0.983933 | 1.205463 |
| No weight decay | 1.074802 | 0.984267 | 1.205770 |
| Target frozen | 1.165386 | 1.082031 | 1.290371 |

The initial downward movement improves calibration because epoch0 overestimates
the stochastic-policy return. It is not itself a failure. Removing dropout or
weight decay changes little in this short paired test. Keeping targets fixed
nearly stops the movement, implicating moving-target bootstrapping in its
propagation, not proving that frozen targets are a useful final algorithm.

## Exact Residual Decomposition

On the same128-transition replay probe, the mean residual is the sum of reward,
entropy bonus, mean-twin backup minus current online Q, and the minimum-twin
penalty. Q/reward units are discounted percentage-point rewards relative to
initial capital, NOT annual return percentages or USD.

| Epoch | Reward | Entropy bonus | Mean backup minus online | Minimum-twin penalty | Residual |
|---|---:|---:|---:|---:|---:|
| 0 | -0.001189 | 0.002414 | 0.004962 | -0.021599 | -0.015412 |
| 6 | -0.001189 | 0.001169 | 0.028112 | -0.031048 | -0.002957 |
| 40 | -0.001189 | 0.000103 | 0.427070 | -0.362098 | 0.063886 |

The twin-minimum term is much larger at epoch40 and offsets most of the positive
mean-backup correction. Its mathematical presence is expected in SAC; a large
negative calibration bias is not automatically a coding defect. This motivates
the explicitly experimental minimum-versus-mean target intervention, rather
than silently replacing SAC's clipped-double-Q estimator.

In the epoch40 production continuation, the sampled critic loss decreases from
0.7757 at update1 to 0.5829 at update1024 while panel bias changes from -25.0690
before updates to -25.0900 afterward. Lower training TD loss is therefore not a
sufficient acceptance criterion for this critic. These losses use different
scheduled batches and are descriptive, not an equal-batch convergence proof.

## Diagnostic Numerical Incident

The first epoch40 comparison stopped at the final mean-target measurement:
subtracting large float32 Q values left a residual identity difference of
`5.340576171875e-05`, exceeding a residual-scale tolerance. Its failed receipt is
retained under identity `2b491f4ebbde7ffea8214fd19fcb02774370b3e2d925f76f514a1f00b9514f16`.
The checker now compares the reconstructed target at native target scale, with
the same `rtol=2e-5, atol=2e-5`, and separately records residual roundoff. A
regression accepts this cancellation case but rejects an added unit of error.
No learning arithmetic changed. Both arms are rerun under a new pinned identity;
the failed run is not promoted to completed evidence.

## Epoch40 Paired Intervention

Completed at 16:44:16 UTC. Both arms restore the same epoch40 optimizer, replay
schedule, actor and temperature. Only the target twin aggregation changes.
Actor logits and frozen parameters remain exact; no account performance change
is expected or claimed from these critic-only updates.

| Variant | Panel bias | Panel RMSE |
|---|---:|---:|
| Epoch40 before updates | -25.069047 | 25.542060 |
| Minimum-twin target, 1024 updates | -25.089978 | 25.581378 |
| Mean-twin target, 1024 updates | -23.456710 | 23.913255 |

The panel's mean realized soft return is -0.031391 in both arms. The mean target
reduces pessimism relative to the paired control, while leaving substantial
miscalibration. This establishes an effect of target aggregation in this
controlled continuation, not the complete historical cause of the 40-epoch drift.
It does not establish that mean targets will improve a jointly trained policy.
In particular, replacing the minimum can reintroduce overestimation.

The corrected checker reproduced every previously completed numerical point
exactly, including all five production points and the first four mean-target
points. Maximum recorded residual roundoff was 2.3842e-05 in production and
5.3406e-05 in the mean arm. The restart changed only verification arithmetic.

## Execution Receipts

- Five arms: runtime `bd2e97de`, identity
  `fd1518bea0842e3ac288bb71b2b3bde4d8a75753b3f8b16253cf3149484f1b4c`;
  369.62 seconds, 12 output files independently hash-verified.
- Read-only decomposition: runtime `e8de6f66`, identity
  `f22cf53954dbc5ab9160467fba8ac37ed98ed701deecffd82a62259e9b4b66a5`;
  4.87 seconds, 2 output files independently hash-verified.
- Latest focused CPU suite: 29 passed, 1 CUDA-only test skipped. Actual diagnostic
  updates and snapshot checks ran on RTX5090. No claim of a full-suite run.
- Completed epoch40 contrast: runtime `ad89fadd`, identity
  `fdd5fb3cbb823744141f88fc06110c7de3e40c4a8048369c62ae6620f2216d60`;
  220.03 seconds, 6 output files independently hash-verified. Seven paired arms
  in total completed across the two intervention experiments.

## Decision Boundary

Do not infer improved trading from reduced Q calibration error while the actor
is fixed. Before refining the actor, confirm any critic correction using a
full-TRAIN replay with a frozen-policy realized-return reference, monitor both
underestimation and overestimation, and preserve the epoch0 baseline. Only then
run guarded refinement and another deletion/addition feature pass. The
stochastic-versus-greedy deployment gap remains a separate issue to measure.

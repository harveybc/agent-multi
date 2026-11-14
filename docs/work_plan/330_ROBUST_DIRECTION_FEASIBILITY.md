# 330: Robust Direction Feasibility

Status: IMPLEMENTED_AND_UNIT_VERIFIED_BEFORE_RUN.

## Question And Scope

After329, does changing the direction of an actual AdamW displacement permit
nonzero expanded-FIT learning while preserving original-FIT decisions robustly?
This is a bounded feasibility diagnostic, not a replacement training algorithm,
new actor, or proof of generalization. Source299, its71 features, frozen encoders,
and actor remain unchanged. STL stays paused. No validation2024 or TEST access.

## Frozen Protocol

Use the four paired fold/seed controls from322 and their four8192-step protected
states from329. At each of these eight starts, independently propose the next
eight scheduled batches (64 proposals, not sequential training). Initial states
have fresh AdamW; protected states retain their saved optimizer moments. Every
proposal starts from the exact same saved weights and optimizer for its cell.
The batches come from the unchanged expanded324 FIT support and327 schedule.

Compare two directions for each actual AdamW proposal: unmodified displacement
and closest linearized feasible displacement. Obtain the latter by projecting
onto original-FIT action-gap halfspaces using a CPU float64 Jacobian and a
nonnegative dual quadratic solve. Protect the native GPU starting action per
row; this is deliberately stricter than merely bounding mean regret. A negative
result therefore does not establish infeasibility of all safe updates.

The target margin per row is1e-5 times max(1, maximum absolute starting Q),
fixed before proposals. Require separation from both alternative actions.
This is a numerical robustness buffer, not a confidence/calibration claim.
Limit corrected displacement norm to4 times the proposed norm. For each arm
try scales1,1/2,...,1/128, in that order; do not adjust these after results.

An accepted endpoint must have nonzero parameter motion, preserve all starting
FIT actions with the fixed margin on native GPU32, CPU32 and independent NumPy64,
and respect the original322 regret limit+1e-8 on all three. Record actual finite
nonlinear checks, not only linearized constraint residuals. Tiny score agreement
alone is insufficient. Reject nonfinite values and solver failures.

Measure expanded-FIT MSE and action regret against each start. A learning witness
requires accepted motion and full expanded-FIT MSE decrease>1e-10 in both native
GPU and independent NumPy64 evaluation; separately
report regret improvement>1e-8 and nondegradation. MSE alone cannot confer trading
or actor improvement. No audit/stop labels enter proposal, acceptance or selection.
Keep all attempts including failures. No automatic model promotion or long fit.

## Test Design And Traceability

| Requirement | Test/evidence |
| --- | --- |
| Real AdamW, not raw-gradient substitution | Save pre/proposed weights and optimizer hashes; paired exact displacement |
| Direction correction is real | Analytic2D projection, feasible identity, contradictory constraints reject |
| Numerical robustness | Known fragile-tie fixture rejected; margins checked in three implementations |
| Nonlinear final authority | Feasible linear witness with failed final margin must reject |
| Genuine learning | Full expanded-FIT loss/regret before/after; zero-motion cannot pass |
| No hidden data/model mutation | Parent seals, supports, source hash before/after; no actor loads for updates |
| Reproducibility | Independent serialized endpoint replay and rerun proposal witnesses |
| Bounded/resumable execution |64 independent paired proposals, per-cell seals, supervisor exit receipt |

Implementation proceeds only after these tests are defined. Results and exact next
action will be appended here and to the persistent method state.

Focused CPU regression suite:25 passed in3.50s. It includes the projection
controls, independent NumPy forward, finite-difference Jacobian witness,
direction-to-parameter mapping, transactional329 tests and native warm-start
regressions. These are implementation checks, not experimental acceptance.

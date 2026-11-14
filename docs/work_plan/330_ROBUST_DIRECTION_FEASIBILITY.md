# 330: Robust Direction Feasibility

Status: COMPLETE_PARTIAL_FEASIBILITY_NO_PROMOTION.

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

## Executed Results

Runtime code:e3206fd1. Identity:
`32a49fbb371a3637e669df9095026be9fab5e711b5942916f2ae330f53058621`.
The bounded probe completed64 independent paired proposals in3.355 seconds
inside the runner, after GPU readiness passed. Supervisor exited0 with zero
restarts. This is not64 training runs or epochs: only one actual AdamW step per
proposal, with frozen historical embeddings and small critic heads.

| Start state | Raw accepted /32 | Projected accepted /32 | Projected MSE learning | Projected regret improvement |
| --- | ---: | ---: | ---: | ---: |
| Four original controls | 32 | 32 | 32 | 0 |
| Four saved8192 states | 0 | 9 | 9 | 2 |
| Total | 32/64 | 41/64 | 41/64 | 2/64 |

The informative difference is at the saved stalled states, not the healthy
initial controls. Nine updates previously unavailable under the robust raw
guard become feasible. By fold and seed, from eight independent proposals each:

| Saved8192 state | Raw accepted | Projected accepted | Projected regret improvement |
| --- | ---: | ---: | ---: |
| A /19 | 0 | 0 | 0 |
| A /43 | 0 | 5 | 0 |
| B /19 | 0 | 2 | 0 |
| B /43 | 0 | 2 | 2 |

All41 projected accepted updates reduce full expanded-FIT MSE in both native
GPU and independent NumPy evaluation. Only two, B43 offsets2 and4, also reduce
expanded-FIT action regret:0.2479163481 to0.2474497334. These are two correlated
one-step witnesses from the same state, not independent seeds or proof of
sustained improvement. Actor and account trading performance were not evaluated.

## Why23 Projected Proposals Fail

All64 linear solves pass their declared feasibility checks and all corrected
directions stay inside the4x norm cap. The23 projected refusals occur in the
finite nonlinear margin check. At full scale, all23 still preserve the intended
actions in all three implementations, but miss the predeclared safety margin
by between1.62e-8 and9.67e-7 Q units. They are not accepted merely because their
actions happen to agree. Smaller scales also fail to produce an acceptable
endpoint. The projection is tangent to its linearized boundary; the actual
network and finite precision do not guarantee the requested margin there.

This separates failure to preserve an action from failure to preserve its
requested numerical buffer. Do not attribute all23 to forgetting or claim
the whole nonlinear feasible set is empty. Conversely, do not relax the
acceptance margin after inspecting these outcomes.

## Independent Verification

-65 seals authenticated, including all64 proposal units.
-530 finite endpoint trials retained, including every rejected trial.
-73 accepted endpoints (32 raw +41 projected) independently reconstructed
  from their initial weights, directions and scales.
-Independent NumPy metric maximum difference:2.22e-16.
-Local CPU32 and NumPy64 guard failures:0; all protected action vectors and
  numerical margins survived this additional host replay.
-Separate native GPU process reproduced all64 actual AdamW proposals AND
  their optimizer states exactly, plus all73 accepted endpoint actions/metrics.
-Maximum normalized linear constraint violation:8.85e-10, below1e-7 gate.
-Source299 and all parent snapshots unchanged. No2024 or TEST evaluation.

## Interpretation And Next Action

There is a constructive partial answer: direction adjustment can recover
nonzero robust learning where scalar backtracking alone stalls, in3/4 saved
states. It is not yet a complete refinement repair: A19 has no accepted
proposal, most projected steps at stalled states are refused, and only one
state improves action regret. Do not restart feature selection on a claimed
repaired critic or promote a new trading model from this diagnostic.

Next bounded test: retain the existing1e-5 acceptance margin and4x norm cap,
but solve toward an interior margin and relinearize failed finite proposals
within a fixed small repair budget. Freeze that protocol before running; report
separately numerical robustness, full-FIT MSE and action regret. Only after
this finite feasibility check should a short sequential-update contrast test
whether the improvement persists. No successor or long training is running.

## Evidence

-[All proposal metrics, without bulky per-row action vectors](generalization/robust_direction_20261114/SUMMARY.json)
-[Frozen runtime configuration](generalization/robust_direction_20261114/CONFIG.json)
-[Independent CPU/NumPy verification](generalization/robust_direction_20261114/INDEPENDENT_VERIFY.json)
-[Exact GPU proposal and endpoint replay](generalization/robust_direction_20261114/GPU_REPLAY.json)
-[Feasibility and rejected margins plot](generalization/robust_direction_20261114/FEASIBILITY.png)

Full per-row action vectors, optimizer states and proposal/endpoint weights
remain in the sealed run store; the public summary records the full-result
SHA256 and explicitly identifies the omitted vectors.

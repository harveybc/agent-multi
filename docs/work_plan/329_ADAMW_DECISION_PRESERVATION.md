# Actual AdamW Decision Preservation

Owner authorized the contrast on2026-11-14 after328. COMPLETE_NO_PROMOTION.
18 focused tests pass, including transactional rollback and resume parity.
Runtime commit `db1a2dd6`, identity
`995c5a1f3b21bf729134b08ab429030e5844447fc7f6570db0efc41dd41e92e4`.

## Frozen Protocol

Four new full-head fits, foldsA/B x seeds19/43, starting at the exact322
control weights. Reuse the four unprotected327 full-head controls after exact
replay; do not retrain them. Same expanded324 FIT, targets, covering schedules,
initial weights, LR3e-5, AdamW weight decay0.01, batch128, MSE and norm clip10.
Both begin with fresh optimizer state. Only the update acceptance changes.

At each proposal, save weights and AdamW state. Apply the actual AdamW step,
including moments, preconditioning and decay, to obtain displacement d.
Try theta+alpha*d at alpha=1,1/2,...,1/128, in that order. Accept the largest
scale whose mean original-FIT action regret is <= the initial control+1e-8
and whose rounded parameter displacement is nonzero. This protects aggregate
original-FIT decision quality, not every individual original action.

For an accepted reduced step retain the newly computed AdamW moments/time
index and damp the complete displacement, including decay; do not pretend
this is unchanged AdamW. On total rejection restore BOTH parameters and
optimizer state. The minibatch schedule advances on every proposal, including
rejections, so the next proposal can use different examples. Exceptions also
restore state before failing closed. Zero parameter movement is not accepted.

There is no extra monotone expanded-loss condition: that would add a second
change to the experiment. Expanded-FIT learning is measured, not assumed.
Original-FIT guard uses native targets only; no stop/audit decisions. Preserve
the original239-row inference batch geometry for exact control comparability;
only original-FIT indices/targets enter the guard.

Primary8192 and secondary32768 attempted updates, not accepted updates.
Run every scheduled proposal even if a rejection streak reaches256; flag
stalls and report longest streak rather than hiding them through early exit.
Record accepted/reduced/rejected/rounded-zero counts, displacement norms,
every tested scale's regret, full-proposal actions and retained FIT actions.
Fixed endpoints and curves at0,64,256,1024 and every2048; resumable snapshots
include the trace, optimizer state and RNG. No held-out stopping or selection.

Joint diagnostic gate: old-FIT regret does not worsen+1e-8 AND expanded-FIT
regret improves>1e-8. Nonzero movement and accepted counts are also required;
preservation without learning fails. Report the4/4 consistency rule, numeric
MSE and stop/audit descriptively. These are TRAIN-internal diagnostics with
previously full-TRAIN-fitted encoders, not independent trading validation.
Source299 actor,71 inputs,49 cumulative return candidates, data and strategy
remain unchanged; STL stays paused. No2024/TEST inference or live promotion.

## Requirement To Evidence

| Requirement | Test / runtime evidence |
| --- | --- |
| Real AdamW proposal, not raw-gradient proxy | Alpha1 state and optimizer parity with unprotected step |
| Reduced step includes every parameter | Known displacement barrier, exact alpha1/2 reconstruction |
| Rejection is transactional | Nonempty optimizer rollback, exception/NaN rollback |
| No zero-motion success | Rounded-zero negative control |
| Correct resume | Interrupted versus uninterrupted protected steps |
| Same underlying experiment | Four replayed controls, initial/support/schedule hashes |
| Preservation at every retained state | Full retained-action ledger and independent regret recomputation |
| Learning rather than stalling | Accepted counts, displacement, expanded metrics and rejection streaks |

This tests a bounded mechanism, not a guaranteed cure. If it stalls or only
preserves the old decisions without learning, document that result; do not
relax the gate after observing outcomes or promote a zero-update model.

## Interim Portability Witness

Recorded before all four fits finished, without changing the running rule:
the A19 final checkpoint's GPU retained-FIT regret is0.000418584 against an
initial control0.000886435. Independent NumPy float64 inference changes one
original-FIT action and yields0.002114517, violating that same guard. Native
CPU PyTorch float32 also changes two original-FIT actions relative to the GPU
ledger; the decisive Q values are separated at floating-point precision.
These are copies of the diagnostic critic, not modifications to source299.

Keep this failure visible in independent verification. GPU self-replay and
cross-implementation robustness are separate checks; do not change tolerance
or report CPU action parity as passed merely because GPU reloading succeeds.
The remaining fits retain the prespecified protection and budgets.

## Completed Results

All four fits completed in798.58 seconds (13min19s) on the primary GPU,
with zero supervisor restarts and exit0.12 unprotected control metric replays
were exact. A separate GPU process reproduced all eight saved endpoint action
vectors and12 resumed proposal transitions exactly.18 focused tests pass.

Independent verification authenticated five seals,131072 proposals and
9633792 retained original-FIT action decisions. All recorded GPU retained
states satisfy their frozen regret bounds. Only1416 proposals were accepted
(1.0803%):907 full and509 reduced;129656 were rejected (98.9197%). No clipping
or rounded-zero acceptance occurred. All four cases had rejection streaks
over256; longest streaks were21038,26861,28590 and13103 respectively.

Equal-weight means across the four fold/seed pairs, native action regret
(lower is better, NOT trading returns):

| Variant | Attempted updates | Original FIT | Expanded FIT | Descriptive audit | FIT passes |
| --- | ---: | ---: | ---: | ---: | ---: |
| Initial control | 0 | 0.008043 | 0.232719 | 0.236451 | reference |
| Unprotected327 | 8192 | 0.025809 | 0.116318 | 0.227021 | 0/4 |
| Protected329 | 8192 | 0.007054 | 0.231466 | 0.231290 | 2/4 |
| Unprotected327 | 32768 | 0.027790 | 0.037762 | 0.261350 | 2/4 |
| Protected329 | 32768 | 0.007054 | 0.231349 | 0.231290 | 2/4 |

Protected FIT passes are A43 and B19 at both endpoints. A19 and B43 do not
improve expanded-FIT regret against their own initial control. The protection
preserves original decision quality on its execution device but largely
arrests learning; it does not meet4/4 consistency. The descriptive audit does
not override that outcome. Every rejected step is recorded, not counted as
an optimizer update. The data, proposal order and budgets were not reduced.

## CPU Portability Failure

Independent NumPy float64 and native PyTorch CPU float32 both fail the same
original-FIT guard at two of eight checkpoints, four failed implementation/
checkpoint combinations. This is a failed robustness check, not an accepted
approximate replay. The small value-RMSE discrepancy (maximum3.133e-7 for
NumPy) does not make the changed decisions harmless.

| Checkpoint | GPU regret | CPU NumPy64 | CPU Torch32 | Original limit |
| --- | ---: | ---: | ---: | ---: |
| A19 /32768 | 0.000419 | 0.002115 | 0.009149 | 0.000886 |
| B43 /8192 | 0.010464 | 0.012330 | 0.012330 | 0.010464 |

At A19/32768, NumPy changes one retained action and CPU float32 changes two.
At B43/8192 both implementations change one. The relevant Q differences are
at floating-point precision. Native GPU cold replay and resumed transitions
remain exact, distinguishing reproducible device-specific execution from
robust cross-implementation decision preservation. Source299 remains unchanged;
these are diagnostic critic heads, not a new actor or live-trading model.

## Decision And Next Step

Do not adopt this scalar-backtracking mechanism as the general refinement
repair or restart floating selection with a supposedly repaired critic.
It has exposed two limitations: it only shortens the proposed direction,
and its discrete guard can accept numerically fragile decision boundaries.
This does not prove that no nonzero safe learning direction exists.

Next design a small FIT-only feasibility test of a direction-adjusted actual
AdamW proposal with explicit decision-margin robustness, followed by the real
finite-step decision check. Compare against these preserved controls and
measure new-example learning, not only constraint satisfaction. First verify
that nonzero robust proposals exist; do not launch another long fit merely
to repeat prolonged rejection. No such successor is implemented/launched here.

## Evidence

- [All paired endpoints](generalization/adamw_preservation_20261114/RESULTS.md)
- [Full metrics and accounting](generalization/adamw_preservation_20261114/RESULT.json)
- [Independent ledger and CPU verification, including failures](generalization/adamw_preservation_20261114/INDEPENDENT_VERIFY.json)
- [Exact native GPU and resume replay](generalization/adamw_preservation_20261114/GPU_REPLAY.json)
- [A19 curves](generalization/adamw_preservation_20261114/A_19_curves.png)
- [A43 curves](generalization/adamw_preservation_20261114/A_43_curves.png)
- [B19 curves](generalization/adamw_preservation_20261114/B_19_curves.png)
- [B43 curves](generalization/adamw_preservation_20261114/B_43_curves.png)

Curves are retained beside those reports. Full action/trial ledgers, weights
and optimizer snapshots remain in the sealed run store. No2024/TEST inference,
feature changes, STL resume, actor update or live-model promotion occurred.

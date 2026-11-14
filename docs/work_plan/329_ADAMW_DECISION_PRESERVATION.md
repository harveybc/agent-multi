# Actual AdamW Decision Preservation

Owner authorized the contrast on2026-11-14 after328. TESTS_PASS_PREPARED.
18 focused tests pass, including transactional rollback and resume parity.

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

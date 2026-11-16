# 365 - Restore best validation state at LR reductions

## Owner Direction And Scope

2026-11-16: reduce from the best validation checkpoint rather than the deteriorated
latest weights. Patience becomes half of early stopping (20/2 = 10 completed
nonimproving observations). This is a new optimizer control policy, not the
standard behavior of ReduceLROnPlateau and not proof of improved generalization.
Follow-up owner instruction explicitly prioritizes immediate restart. Seal and
stop364 A/D/control, preserving the fixed-LR evidence. Restart all three as365
from the SAME selected299 structural initializations and fresh replay/seed7,
with identical new scheduling rules. Do not silently resume362's old scheduler.

## State Machine

1. Evaluate full TRAIN and VAL accounts once per completed epoch, as before.
   Only VAL return/week determines improvement; require >2 orders/week in both
   periods. TRAIN return cannot compensate a decline in VAL.
2. Preserve historical global best, absolute improvement threshold1e-8. Count
   ten nonimproving observations; torch uses `bad > patience`, so its internal
   patience must be9 for exactly ten misses. Cooldown0; no extra hidden delay.
3. On an actual actor LR decrease (factor0.5, floor1e-7), authenticate the stored
   best bundle and restore actor, twin online/target critics, temperature and
   their AdamW moments. Keep the current reduced LR, not the historical LR
   inside that bundle. Critic/temperature LRs remain at their current values.
4. Keep current replay, sampler/random streams, global epoch/update counters and
   scheduler state. This is SAC parameter rollback with recent off-policy
   experience, not a historical time rewind. No duplicate replay traversal.
5. Never reset global early-stopping patience merely because restoration took
   place. Allow at least10 new epochs after a real reduction. Do not schedule
   another reduction when early stopping is already due. Minimum100 total,
   monitoring warmup20, maximum2000, remain explicit safety bounds.
6. At the LR floor there is no actual reduction and therefore no restoration or
   artificial extension. If no eligible best exists, refuse this fork.

## Evidence And Recovery Semantics

The row's metrics and weight snapshot describe the evaluated pre-rollback model.
An event stores both hashes and the best epoch restored. `continuation_metrics`
identifies the already-measured best checkpoint for the post-rollback recovery,
without pretending those were the last epoch's earned results. Recovery contains
the post-rollback model/moments, current replay/RNG and scheduler atomically.
Reloading a bundle must not leave the scheduler attached to a discarded optimizer.

## Paired Experiment

All three365 arms share seed7, initial role LRs3e-6, minimum100/max2000 epochs,
full TRAIN/VAL data and common initial tensors/moments inherited by name from299.
Only their branch architectures differ as declared in364. All three receive
the same rollback policy; no architecture is given preferential scheduling.
The incomplete364 fixed-LR curves remain historical controls, NOT proof that
restoration improved learning: that claim would additionally need matched
reduce-only (ten-miss rule) versus rollback experiments. Those are not launched
as part of this priority restart.
No automatic model promotion, phase2 closure, DEAP, STL or live deployment.

## Requirements To Tests

| Requirement | Predeclared test |
| --- | --- |
| Ten means ten | misses1-9 do not reduce,10 does; signed tiny improvements and ineligible positive results |
| No best-by-TRAIN | VAL selected checkpoint restored despite higher TRAIN in the latest state |
| Complete parameter consistency | actor/critics/targets/alpha and named Adam moments equal selected checkpoint |
| Keep reduced rate | two reductions restoring the same best still decrease cumulatively |
| Stable optimizer binding | scheduler and trainer retain the same optimizer object |
| Recent replay/RNG retained | event does not change random state or replay length/state |
| Honest metrics | pre-rollback row distinct from continuation checkpoint metrics |
| No infinite extensions | stop check takes precedence over another reduction; floor no rollback |
| Crash recovery | uninterrupted versus interrupted just after rollback yields identical next updates and recovery |
| Fail closed | altered bundle or score mismatch rejected before model mutation |

Legacy362 defaults and tests remain valid but its queues stay parked. The opt-in
version365 alone changes restoration and patience semantics. Implementation and
qualification evidence follows. The restart replaces the earlier proposal to
wait for364 to finish; the new owner instruction takes precedence.

## Qualification Before Restart

45 focused CPU tests pass: exact tenth miss, positive but activity-ineligible
metric, negative signed best, tiny improvement, full SAC/Adam restoration,
cumulative LR decreases from the same best, optimizer object identity, rejected
tampering/score/path mismatch, and interrupted versus uninterrupted recovery
through a real rollback. Replay and random state reproduce exactly; early stop
prevents an extra reduction at its boundary. Legacy362 regression tests pass.
Touched Python files pass repository Ruff rules.

364 sealed epochs: A8, D18, control3, before stopping/disabling their workers.
Its observer and automatic old362 resumption are disabled. Checkpoints and
historical evidence remain intact. The source and all71 inputs remain unchanged.

Native scheduler semantics reference:
[PyTorch ReduceLROnPlateau](https://docs.pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ReduceLROnPlateau.html).

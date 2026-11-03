# Paired stopping-rule comparison

## Owner Amendment:20-Epoch Monitor Warm-up

2026-11-03: epochs1-19 remain measured but cannot establish a selected best,
increment patience or reset it. After completing epoch20, its score establishes
the first best and counter0. Epoch21 is the first comparison. For the guarded
arm, the first previous-epoch reference is20, not19. The earliest patience stop
is epoch40. This monitor warm-up is distinct from4096 initial replay warm-up
decisions; do not repeat the replay warm-up on continuation.

Both runs were below20 when the owner requested the change. Preserve the9 and11
complete learning epochs respectively, rather than retrain them. A new runtime
and new run identities copy their last full-epoch weights, optimizer, replay,
writer and RNG files byte-for-byte. Only monitor bookkeeping is recomputed from
the existing per-epoch metrics, with best unset and patience0. Original metrics
and original monitor decisions remain available. An unfinished epoch restarts
from its committed boundary. Old early inference bundles remain historical,
not candidates for the warmed-up selection.

Migration refuses if the source is active, any source/data/recovery hash differs,
or the available prefix reaches the new monitoring epoch.43 focused tests pass,
including warm-up boundaries, exclusion of early transient maxima, bookkeeping
rebasing, and exact learning resume under all stopping modes with/without warm-up.

Warm-up runtime`ee6e71a0` resumed at2026-11-03T15:59UTC. Both workers advanced
beyond their retained prefix with zero restarts: primary epoch10, secondary
epoch12. Copied model and learning-state hashes match their source recovery
receipts exactly. No early selected best is carried into the amended monitor.

| Warmed-up arm | New identity |
|---|---|
| validation | `d53a3c061a816b7d5564cc0795cad540df82b94da65e33114d331c7930dddf96` |
| mean_no_regression | `6dfead4e19c2fe0beffab3f2b05be4d214cb33b0627ccc5dc183e6459b1fc530` |

Owner request2026-11-03, superseding further execution of193's mean-only fit.
The former checkpoints and results remain available. No DEAP or new selection.

## Two Arms

Both arms train from scratch with seed7 and exactly193's108 inputs,31 groups,
13x2 window, architecture, optimizer, batch128, native risk/costs, fixed TRAIN
scaler and full TRAIN2020-2023 / VALIDATION2024 accounts. TEST2025 stays closed.
Maximum2000 complete epochs; patience20; minimum improvement1e-8.

-Primary RTX5090:`validation`. Maximize validation return per week. TRAIN is
 still learned from and fully evaluated, but does not select checkpoints.
-Secondary RTX4090:`mean_no_regression`. Maximize the mean of TRAIN and
 validation return per week, accepting a new checkpoint only if neither
 component nor their mean decreased from the immediately preceding evaluated
 epoch. Require improvement over the best *accepted* mean by more than1e-8.

The first finite epoch establishes a best. Every rejected epoch increments
patience exactly once, including a mean improvement rejected by the guard.
Only acceptance resets patience. Update the previous-epoch reference after
every evaluation, including rejection, and persist it across restarts.
At stopping, restore each arm's accepted best, not the last epoch.

The guarded rule follows the owner's previous-epoch definition; it does not
guarantee validation is nondecreasing relative to the accepted best. Recovery
after a rejected dip can still accept a validation value below that older
checkpoint. Report both references explicitly rather than silently strengthen
the rule. Validation-only patience permits temporary drops during training but
never replaces its best with a worse validation checkpoint.

## Interpretation

The old mean monitor allowed large TRAIN growth to offset a validation decline.
Early stopping controls checkpoint choice; it does not force gradient updates
to improve validation. SAC learns from TRAIN rewards through actor/critic
losses; those losses do not guarantee monotonic realized trading returns.

This is a single-seed two-rule diagnostic, not a multi-seed superiority claim.
Hardware differs; identical seeds do not guarantee bit-identical GPU learning.
Log both stopping rules on each arm's actual trajectory as shadow decisions,
without affecting training, to distinguish stopping logic from device drift.
No extension of a stopped arm is implied by its shadow monitor.

## Tests Before Launch

| Requirement | Test |
|---|---|
| TRAIN cannot compensate in validation-only mode | Extreme TRAIN improvement with validation decline is rejected |
| Either component can veto guarded mean | Separate TRAIN and validation decreases; strict equality boundaries |
| Historical best still matters | Joint recovery that does not beat accepted best cannot reset |
| Previous means previous evaluated epoch | Rejected dip followed by recovery uses the dip, not accepted best |
| Patience20 and1e-8 | Tiny improvement, exact tie,20 consecutive rejections |
| Resume is semantically identical | Checkpoint previous values, counters, replay and next-update parity |
| No input/split drift | Identical recipe, support, risk and effective-model digests |

Best inference bundles and automatic remote collection remain mandatory.

## Launch Receipt

Runtime`e689390b`;36 focused tests pass, including actual interrupted/resumed
native learning for all three stopping modes. Both complete real-data
preflights pass. Recipe, input files, support, model configuration, source code
and seed equality were checked before launch; only the stopping definition and
hardware differ. The parent mean-only primary was stopped with29 full epochs
recoverable; its retained inference bundles and the completed secondary's
results were copied before the new comparison.

Both new workers were verified training at2026-11-03T13:47UTC, with zero
restarts and65 gradient updates each. Persistent services and a180-second
collector preserve results without intermediate approval.

| Arm | Identity |
|---|---|
| validation | `556e47c27dfde31d121b2e517cca230aef6a1c716e71c993d9e2af574e9dfc59` |
| mean_no_regression | `97a4c2f8581a03e0083f15992e3d7310d094d67e4370e09ca93e6411f1b3dec0` |

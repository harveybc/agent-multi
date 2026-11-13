# Automated Critic Representation Diagnostic

Owner2026-11-13 authorizes automatic continuation after314 without intermediate
approvals. STL remains paused. Preserve selected299, its71 columns and49
cumulative phase2 return candidates. This diagnostic precedes a new actor
refinement or floating pass; it cannot promote a trading model.

## Fixed Design

Authenticate314 completion, source checkpoint and partition first. Reuse its
expanded146 true-label head-only endpoints for seeds19/43 without retraining.
For each seed sequentially fit three new arms: heads with shuffled labels,
critic representation with true labels, critic representation with shuffled
labels. All start from the original299 encoders and identically reset critic
heads, use the same146 FIT rows,4096 updates,batch128 and existing optimizer
parameters/loss. Only the scope of trainable critic tensors differs in the
true-label comparison. Actor, actor encoder and entropy temperature stay fixed.
Target critics synchronize only after fitting, as in300/314.

Shuffle whole three-action target vectors ONLY among FIT origins of the same
year and position/legality class. Keep all monitor labels untouched. The mapping
depends on seed and strata, not on returns. Record singleton strata and the
actual moved-label count. This is a negative control retaining class/year
structure, not a proof of independence or a replacement for temporal testing.
No shuffled target is used for reporting performance against truth.

Reuse exactly55 held-out origins and38 old monitor origins. Report all fixed
checkpoints and the4096 endpoint; never choose updates using these metrics.
These panels were inspected earlier and windows can overlap. Results describe
out-of-fit diagnostics, not independent confirmation. Historical mixed-COT
source restriction remains; Retsu's source qualification is not bypassed.

## Requirements And Tests

| Requirement | Check |
| --- | --- |
| Dependencies complete and unchanged | Hash receipts, source/support/partition identities, fixed113/146/55 counts |
| Shuffle is an honest negative control | Deterministic permutation within FIT year/state, monitor unchanged, vectors kept intact, no mutation of input records |
| Scope actually changes | CPU real-model regression verifies critic encoder moves only in representation arm; actor and all noncritic tensors exact |
| Comparable optimization | Same seeds, head initializer, learning rate, loss, batch size and schedule; budgets fixed before execution |
| Inference preservation | Full TRAIN actor logits exact and saved checkpoint cold-reload actor/Q parity |
| Resume | Per-arm receipts; completed arms skipped, interrupted arm restarts deterministically; exclusive lock |
| No babysitting | Sequential finite service follows314, writes curves, PNG, tables, paired deltas and final recommendation |

Primary descriptive comparison is true-label representation versus true-label
heads on the shared55, reported separately per seed. A recommendation to
design a bounded own-policy actor step requires non-worse pair accuracy and
mean regret in BOTH seeds, a strict improvement in at least one metric per
seed, and lower regret than the matched shuffled representation in BOTH seeds.
Otherwise recommend diagnosing partial observation/representation or obtaining
more native own-policy FIT coverage. This predeclared recommendation is not a
significance test, a release gate, or an automatic actor update. No fitting or
inference on2024/2025. No live deployment. Stop after the finite chain and
report the measured result; never run meaningless repeats to keep a GPU busy.

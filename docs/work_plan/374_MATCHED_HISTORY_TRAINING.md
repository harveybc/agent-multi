# 374: matched-update fresh-network learning curves

## Frozen design before implementation

Question: under the fixed selected71 representation, network, risk profiles and
business rules, does increasing the available TRAIN history improve learning?
This is not end-to-end feature discovery or a claim that six months suffice to
discover the already selected representation and risk profiles.

Use372 stores for6/12/24/48 months, the same2024 validation timestamps, seeds7
and19, fresh actor/twin critics/targets/Adam/replay, and source299 architecture
and effective hyperparameters only. Never initialize from trained299 weights.
Upstream fitted producers and fixed scalers differ by subset as documented372.
The12-month constant Kalman residual remains present and explicitly reported.

One complete TRAIN traversal collects initial replay without gradient updates.
Then100 complete traversals, each with6006 evenly distributed optimizer steps
(floor(24025/4), the largest arm's previous per-pass rate). No truncated window,
partial year masquerading as four years, synthetic data or repeated validation
per tick. Evaluate full TRAIN and validation once per traversal; save curves,
native trading metrics, weight diagnostics and epoch-boundary recovery.

This matches gradient budget, not environment interactions or sample reuse:
smaller histories see fewer unique transitions and more updates per decision.
Record both axes. It is a bounded fixed-budget diagnostic: no early stopping
or LR scheduler, common original fixed LR3e-5, not another altered plateau
experiment. The100 update-pass budget is not a claim of complete convergence.
Report endpoint and activity-eligible balanced-min best separately, including
early transients. Zero-trade checkpoints cannot become the selected best.

## Requirement-to-test matrix

| Requirement | Evidence required before launch |
| --- | --- |
| Exact update budget | Schedule sums6006 for each observed TRAIN length; invalid schedules rejected |
| No trained-state reuse | Fresh same-seed parameter digests equal across recipes; Adam empty; source untouched |
| Input and identity |372 receipts verified; manual fixed scaling/window parity; shared validation support |
| Recovery | Interrupted epoch-boundary continuation equals uninterrupted tiny native training |
| No regression to existing runners | New schedule opt-in; existing structural-refit tests pass |
| Sequential unattended work | Per-device finite pinned queue, exclusive lock, memory/GPU admission, terminal receipts |
| Resource ownership | Only freed devices used; existing365/367 and live services untouched |

## Execution

42 targeted tests passed, including real tiny native training with exact resumed
weights and Adam, and existing structural-refit regressions. CPU preflight seed7
passed on all four real stores: manual window/scaling parity, identical initial
parameters across recipes, empty Adam and cold inference-bundle reload.
The model has92200 trainable parameters. TRAIN decisions per traversal are
2780/5891/11841/24025; validation support is identical in all four arms.

Deployment uses two queues, each traversing all four histories so hardware is
not uniquely assigned to a dataset size. Seed7 order48/6/24/12; seed19 order
6/48/12/24. Failed units stop their queue rather than being silently counted as
valid experiments. No neural performance result is claimed by this preflight.

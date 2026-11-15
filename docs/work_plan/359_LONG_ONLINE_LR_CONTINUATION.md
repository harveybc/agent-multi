# Long Online Learning-Rate Continuation

Owner execution order2026-11-15. Continue357's diagnosis, not DEAP, live work or
new feature selection.357 tested512 fixed-replay updates and cannot exclude
late recovery. Here each epoch is a COMPLETE native2020-2023 TRAIN traversal,
with online collection and replay learning;2024 is evaluated, never replayed.
Preserve selected299's71 columns,28 groups,window13/stride2,scaler,architecture,
native engine, risk and costs. All online network parameters remain trainable.

## Frozen Contrast Before Implementation

Three arms: control, actor_lr01, all_lr01. Three seeds7/19/43, same checkpoint
and inherited AdamW moments within every pair; separate replay per run.
Base LR=3e-5; actor_lr01 uses actor3e-6/critic3e-5/temperature3e-5;
all_lr01 uses3e-6 for every role. Keep batch128,replay65536,update_every4,
warmup4096 with existing policy-warmup flag,gamma0.99,tau0.005 unchanged.
These are continuations, not fresh-initialization replication claims.

Maximum2000 epochs. Minimum100 COMPLETE epochs before stopping is allowed,
chosen explicitly to test recovery beyond the former40-epoch termination.
Validation return/week controls checkpoint selection; min_delta1e-8;
activity >2 orders/week in BOTH TRAIN/validation. Save eligible improvements
from epoch0. Keep the existing20-epoch grace period for the patience counter
and patience20. When its threshold is reached early, record the raw counter
and deferred-stop flag; do NOT terminate before100 and do NOT reset the counter
artificially. The same rule applies to all three arms. Later recovery is possible,
not guaranteed;100 is a diagnostic minimum, not a proof of sufficient convergence.

Full frozen-policy TRAIN/validation accounts are evaluated once per epoch.
The stochastic changing-policy TRAIN rollout is not the same estimator, so it
does not replace that checkpoint evaluation. Do not add a second identical
checkpoint evaluation inside the epoch. At completion retain/report selected
and last metrics separately, including epoch0 if it remains best.

## Test Design

| Requirement | Acceptance witness |
|---|---|
| Grouped LR persists | Named optimizer layout round-trips with unchanged moments/LRs and next update |
| Legacy compatibility | Old single-group bundles still load and reproduce logits |
| Bad checkpoint rejected | Missing/duplicate/unknown optimizer names rejected |
| Sufficient budget | Patience reached before minimum does not stop; it does stop at minimum unless improved |
| Full recovery | Split-run versus uninterrupted grouped native training identical in model/optimizer/RNG/replay outcome |
| Diagnostics inert | Epoch hook leaves RNG unchanged; same accounts feed curves and stopping |
| No early-transient concealment | Plot epoch0 and every subsequent epoch, selected and last separately |
| Finite automatic fleet | One queue per worker, locks, identities, reboot recovery, verified aggregation |

Use an opt-in named optimizer layout in checkpoints and optional epoch hook/minimum
in the existing fit loop. Legacy defaults retain existing behavior. Never edit
the G3 runtime in place. Per-epoch CPU plots, module weight drift and fixed TRAIN
probe action/Q/entropy diagnostics share the epoch index. Save per-epoch weight
snapshots, two full replay recovery generations and best inference bundles.

Each worker executes all arms for its own seed, with rotated starting arms for
early coverage. Within-seed comparisons avoid treating hardware/seed differences
as treatment effects. The read-only collector records progress, actual epochs,
patience, stop reason, metrics and elapsed timings; no automatic promotion.
GPU admission is bounded by host RAM as well as VRAM. G3 remains separate.

## Execution Contract

| Worker seed | Sequential order |
|---|---|
|7|actor_lr01, all_lr01, control|
|19|control, actor_lr01, all_lr01|
|43|all_lr01, control, actor_lr01|

Each persistent user service has a dedicated output lock, pinned code/data
identity,5GiB RAM ceiling,2-core CPU quota and one Torch/BLAS thread. Three
bounded failure restarts per hour; a reboot resumes the last committed epoch,
not an uncommitted partial epoch. A separate read-only collector polls every
180 seconds, verifies completed artifacts and writes the nine-run table.
Private host assignments and service files are kept outside version control.

Before launch,43 focused CPU tests passed: grouped named optimizer persistence,
legacy load, exact next update, malformed group rejection, native uninterrupted
versus interrupted continuation (including the actual memmapped JointReplay),
minimum-epoch stopping, epoch0 plots and collector corruption rejection.
The worker additionally requires the exact selected299 source/support hashes,
native initial TRAIN/validation account parity and cold grouped optimizer parity.

Saved evidence per arm: `EFFECTIVE.json` (actual role rates, not only base LR),
`EPOCH_ZERO.json`, `HISTORY.json`, `PROGRESS.json`, `weights/epoch_*.pt`,
`diagnostics/*.png`, two `recovery/` generations, best inference bundles and
final `RESULT.json` with selected/last metrics and stop reason. No synthetic
replay enters359; the independent G3 contrast remains unchanged.

## Decision

Full TRAIN/validation trajectories and weight/function diagnostics determine
whether lower LR permits recovery, delays destruction or prevents meaningful
learning. Do not decide from the best seed alone or mean TRAIN compensating
validation decline. No phase2 closure, specialist launch or broad DEAP follows
automatically. The conditional small training-DEAP option remains available.

## Launch Receipt

Launched2026-11-15 at19:57UTC from code `a5a0e50d`; identical runtime archive
SHA256 `76105b037cd55362eab97790744ad4856c80f64f46b63e5d697526b512c36ca6`
on all three workers. Worker and collector services enabled; no new worker
restarts at launch. Original G3 service remains active without runtime changes.

Initial native accounts on all three devices: TRAIN return16.05822355000215%,
745 trades; validation return18.369945750000117%,207 trades,Sharpe2.690712133251012.
These are reproduced source results, NOT improvements from the new training.
All three grouped optimizer cold-load checks passed with their intended rates.
The first epochs are collecting real transitions and applying online updates.
First-epoch timings are not yet complete; the observer will calculate measured
minutes/epoch and ETA to the minimum budget automatically, not a claimed time
to convergence. Queue completion depends on the actual stopping epochs.

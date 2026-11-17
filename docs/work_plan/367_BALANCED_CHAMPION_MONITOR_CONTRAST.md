# 367: Balanced Versus Strict Champion Monitor

## Contract And Test Design

Owner accepts comparing balanced-min versus strict champion nonregression.
This complements366 and does not alter running365 or claim a solution to SAC
learning dynamics. No live, STL, new features or broad DEAP work.

Let T and V be full-account return percentage points per week on TRAIN2020-2023
and VALIDATION2024. Both accounts must exceed2 orders/week. TEST remains closed.
The sole changed factor is checkpoint/monitor policy:

* `balanced_min`: score=min(T,V); new best requires score>best+1e-8.
  It deliberately permits the higher metric to decline while the lower improves.
  Equal score is not an improvement; no hidden tie-break or sign-dependent product.
* `champion_nonregression`: score=V; new best requires V>bestV+1e-8 AND
  T>=bestT of the SAME saved champion. Negative T/V are allowed. No comparison
  against the immediately preceding epoch, and no unannounced economic tolerance.

Acceptance is shared by checkpoint save, patience, scheduler and rollback.
Rejected higher-V observations must NOT update the scheduler's best.
Restoring the best never counts as improvement or resets global patience.
The existing actor-only LR schedule is identical in both treatments; critics
and temperature retain their current rate. Its limitations remain documented366.

Common source selected299, original control architecture,71 inputs,28 groups,
fixed TRAIN scaling, window13 stride2, batch128, unchanged full historical data,
native strategy and costs. Copy all source parameters and named optimizer
moments. Start a fresh paired replay/random stream per seed, not an in-place
migration of365. Seed7 and seed19 are separate stochastic replications, with
both policies run sequentially on the SAME device within each seed; reverse
policy order in the second replication. Epoch0 native accounts and cold reload
must reproduce the source before training. No cross-device bitwise claim.

Same maximum budget2000, minimum100 epochs, ES patience20, monitor warmup20
(allow early champion saving), absolute improvement1e-8, LR misses10,
factor0.5 and floor1e-7. Actual stopping epochs may differ; report matched
epoch100 AND selected/last metrics and actual exposure. Do not call different
stopping exposures an equal-training-budget treatment effect.

| Requirement | Test before deployment |
|---|---|
| Signed scores | Negative-return improvements accepted; Gaussian multiplication counterexample |
| Strict reference | Reject rebound below champion TRAIN; update both champion references atomically |
| No compensation | High TRAIN cannot mask falling V under either selected rule |
| Shared decision | Rejected V record neither resets scheduler nor moves its best |
| Exact patience |10 misses reduce LR;20 misses stop after common minimum/post-drop budget |
| Rollback binding | Saved score/epoch and restored model refer to the same accepted champion |
| Cold recovery | Interrupted run matches uninterrupted monitor, scheduler, optimizer and replay |
| Backward compatibility | Existing three monitor modes and365 runtime remain unchanged |
| Queue | Wait for authenticated predecessor completion, resume per-unit state, never duplicate GPU work |

Historical replays are descriptive only: alternative monitors would alter
rollback timing and therefore future training. They cannot predict treatment
results. No checkpoint will be promoted merely from this historical comparison.

## Qualification And Historical Results

55 focused CPU tests pass: both signed monitors, champion reference, shared
scheduler decision, native synthetic-engine interrupted recovery for both new
modes, existing rollback/branch/structural refit regressions and resumable queue.
Ruff passes on changed Python files. No current365 worker was stopped or edited.

The sealed366 history prefixes end at A25/D53/control10. Applying each rule to
those same observations (not simulating alternative future learning) selects:

| History | Validation only | Mean | Old mean-no-regression | Balanced minimum | Strict champion |
|---|---:|---:|---:|---:|---:|
| A | 5 | 5 | 5 | 6 | 0 |
| D | 6 | 8 | 8 | 8 | 6 |
| Control | 0 | 0 | 0 | 0 | 0 |

Numbers are selected epochs, NOT profit improvements. Actual metrics, rejection
counts, accepted points and source hashes are in
[historical replay](generalization/monitor367_historical_replay.json).

The implementation adds opt-in monitor modes. Legacy shadow-mode keys remain
unchanged to preserve old recoveries. New modes require `shared_monitor_decision`;
misconfigured direct calls fail. Rejected records cannot update scheduler best.
The separate legacy best-after-warmup diagnostic is disabled for these modes so
it cannot silently export a checkpoint rejected by the common rule.

## Execution Plan

Two persistent sequential queues wait for their pinned365 predecessor to finish
and its RESULT/file hashes to verify. Source299 and runtime identities are pinned.
Resource admission additionally requires the assigned GPU to have no compute
process and available host memory. No source service is stopped by the queue.
After a reboot, completed units verify and skip; partial units resume their full
recovery. No fallback to a different dataset, GPU or source checkpoint.

* RTX5090: seed7, balanced minimum then strict champion.
* RTX4090: seed19, strict champion then balanced minimum.
* RTX4070: existing365 original control continues unchanged, no new job stacked.

These are two paired stochastic replications, not four independent architecture
searches. Source actor/critic/Adam weights are identical before each run; fresh
replay and random streams are paired within a seed. No new fit is claimed until
the queue reports TRAINING and the worker emits its qualification/progress.

## Deployment Receipt

Verified2026-11-17 02:14 UTC (2026-11-16 local date): both new queue services
enabled, active, zero restarts, state `WAITING_PREDECESSOR`, no failure receipt.
Both original365 predecessor services remain active with zero restarts. The
4070 original control also continues outside these new queues.

Pinned runtime revision: `672535cb`.
Archive SHA256: `84c4bcf184016f94c1fa8096478f367fcd7ade8268bedb1d6775751285d8c856`.
Python source identity, independently verified on both devices:
`b638adcd34309642ae0058d90c24a8e3ce866fcfb54701b85695295b968013e9`.

| Device | Seed | Queue config SHA256 | Initial state |
|---|---:|---|---|
| RTX5090 |7|`4ecee9103f2e9c18941fabba3c08e9635b957e74570688b18cea5cf4ed424688`|WAITING_PREDECESSOR|
| RTX4090 |19|`ec93720835c35e355404cac0b9c497f37230e1f546cc0a12200a669f7eb459bc`|WAITING_PREDECESSOR|

Private operator fleet manifest: `/var/tmp/monitor367-deploy/FLEET.json`.
Each worker writes `QUEUE_STATUS.json`, per-policy `PROGRESS.json`, full
recoveries, selected bundles, epoch weight snapshots, diagnostics and RESULT.
New training results do not exist yet. No further owner approval is required
to start after the authenticated predecessor/resource gates pass.

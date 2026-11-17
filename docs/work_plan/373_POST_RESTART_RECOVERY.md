# 373: recovery and resource allocation after maintenance

Snapshot: 2026-11-17, approximately 19:40 UTC. Owner confirmed all machines
restarted and authorized the previously recommended recovery/allocation.

## Execution

All four workers initially resumed automatically. Verified each atomic RESUME
pointer, manifest identity and SHA256 of STATE.json, model.pt and learning.pt.
The recovery loader restores model/optimizer, replay, RNG and monitor/scheduler
state. Recovery is at completed-epoch boundaries: an interrupted partial epoch
is repeated, not claimed as saved mid-epoch progress.

| Run | Last committed epoch | Allocation after inspection |
| --- | ---: | --- |
| 367 strict monitor, seed19 | 66 | Continue automatically |
| 365 control, seed7 | 52 | Continue automatically |
| 367 balanced monitor, seed7 | 17 | Paused; parent queue disabled |
| 368 preserved-champion refinement | 49 | Paused; service disabled |

Both paused checkpoints were reverified after stopping. Their results are
incomplete, not definitive negative conclusions. Source299 remains unchanged.
Inactive completed predecessor services and completed369 were also disabled
to avoid redundant startup work. No live/paper service or VM was modified.

During deliberate interruption,368 logged KeyboardInterrupt followed by a
TensorDict context-unwind RuntimeError. Its failed systemd status was cleared
only after checking the traceback and verifying the committed checkpoint.
The log and FAILURE artifact remain preserved. No corrupted checkpoint was
loaded or training result relabeled successful.

The bounded CPU observer370 remains enabled. Its first post-boot collection
overlapped a worker's INPUT_QUALIFICATION progress record, which has no identity,
and reported KeyError instead of silently accepting an unidentified record.
After workers stabilized, a restarted collection authenticated all four runs
with zero errors. The observer's retained histories do not imply that the two
paused services are running; this allocation record takes precedence.

## Metrics and estimate

These are the last completed epochs, not a new champion:

| Run | TRAIN return | Validation return | Validation trades | Validation Sharpe |
| --- | ---: | ---: | ---: | ---: |
| 367 strict, epoch66 | +18.53% | -1.35% | 143 | -0.222 |
| 365 control, epoch52 | +10.28% | +6.71% | 247 | +0.962 |

Best continuation validation returns remain +13.23% at strict epoch57 and
+12.19% at control epoch47. Neither beats source299's +18.37%; both selected
checkpoints remain epoch0. Recent eight-epoch medians are831s and1605s,
respectively: about8h and21-22h to the minimum100 epochs. These are estimates
for the current arms only, not a promise that the entire queued contrast ends
then. The strict monitor queue retains its originally configured following
arm; runtime admission still checks GPU and host memory before dispatch.

## Next work

The next priority remains the data-quantity comparison, using the four
subset-fitted stores from372. Its neural runner is not yet implemented or
launched. Keep the freed devices available for it after initialization,
input-parity and matched-update checks. Do not replace this with another
unchanged refinement, close phase2, run broad DEAP or start synthetic training.

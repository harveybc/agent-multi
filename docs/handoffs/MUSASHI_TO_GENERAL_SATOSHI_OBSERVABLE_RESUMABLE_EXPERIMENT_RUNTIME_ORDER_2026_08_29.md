# Musashi to General Satoshi: observable and resumable experiment runtime order

Date: 2026-08-29  
Priority: immediate  
Scope: all experiments, screens, training and optimization expected to exceed 30 minutes

## Incident

The first positive-skill screen ran for approximately 18 hours 50 minutes as one CPU process, consumed about 2.9 CPU cores and 2.2 GB RAM, and persisted no completed-cell result, progress journal, checkpoint or defensible ETA. It was paused for inspection and terminated by Musashi. It produced no scientifically usable result.

The source loop was finite, but finiteness is not observability. CPU utilization proves activity, not correct progress. This execution design is rejected and must not be reused.

## Permanent launch invariant

No job expected to exceed 30 minutes may start unless all of the following exist and pass a preflight:

1. Atomic work-unit identity: experiment, family, window, latent, budget, seed, origin, treatment and code/data/config digests.
2. Durable state per work unit: `PENDING`, `RUNNING`, `COMPLETED`, `FAILED`, `TIMED_OUT` or `INTERRUPTED`.
3. Atomic result persistence immediately after every completed unit.
4. Progress heartbeat at least every five minutes with current unit, completed/total units, elapsed time and last durable result.
5. ETA derived from completed comparable units, including median, p90 and pessimistic remaining time.
6. Per-unit timeout derived from a bounded smoke, plus a campaign wall-clock ceiling.
7. Idempotent resume that skips exact completed units and creates a fresh attempt for failed or interrupted units.
8. stdout/stderr captured durably by unit; no sole pipe to an interactive agent.
9. Immutable inputs outside volatile `/tmp` agent scratch directories.
10. A watchdog that detects stale heartbeat, dead process, thermal failure, disk pressure and identity drift.
11. A machine-readable status command that requires no process attachment or agent interpretation.
12. Graceful cancellation that preserves every completed result and marks the active unit interrupted.

The launcher must refuse when any item is absent. A monolithic final-write-only runner is prohibited.

## Positive-skill screen redesign

Replace `tools/positive_skill_screen.py` as an execution driver with three layers:

### Materializer

Materialize every cell and decision replicate before execution. Invalid topology cells are absent with typed reasons. Persist the complete prospective ledger and its digest.

### Worker

Execute exactly one atomic unit per invocation. A unit may be:

- one minimum/mid/top-budget cell;
- one survivor replicate for one seed and origin;
- one matched random control;
- one persistence baseline evaluation.

Each invocation writes an atomic report and exits. It never launches the next unit itself.

### Supervisor/aggregator

Schedule independent units across available machines and devices, enforce budgets/timeouts, compute ETA, and aggregate only complete verified units. Successive-halving decisions are persisted as separate artifacts before the next round is materialized.

## Compute policy

- Run one CPU and one CUDA benchmark cell first.
- Choose CPU or CUDA per architecture from measured throughput, not habit.
- Use all available GPUs for GPU-beneficial independent cells without oversubscribing memory.
- CPU work uses an explicit core/thread limit and leaves operating headroom.
- Do not run TimesNet or other expensive branches for hours on CPU when the measured CUDA path is materially faster.
- No full screen starts until the benchmark yields a measured per-cell ETA and campaign p90.

## Recovery of the interrupted screen

The terminated process has no reusable scientific output. Do not claim partial completion and do not reconstruct results from memory. Re-materialize the same predeclared scientific design under the new runtime. The scientific rule may remain unchanged; only execution mechanics change.

The candidate generation must be copied from its verified restricted artifact store into an immutable run input identified by digest. Do not depend on an agent-owned scratch path under `/tmp`.

## Required tests

- crash after one completed unit preserves and reuses it;
- crash during a unit marks only that unit interrupted;
- exact resume never reruns completed units;
- changed code/data/config refuses resume;
- stale heartbeat triggers timeout and preserves evidence;
- concurrent workers cannot claim the same unit;
- duplicate result with identical digest is idempotent;
- conflicting duplicate result refuses;
- aggregation refuses missing, duplicated or foreign units;
- ETA updates after each completed comparable unit;
- disk-full and write/fsync failures never report completion;
- SIGTERM produces a durable interrupted state.

## Immediate execution order

1. Preserve an incident record containing command, PID, elapsed time, resource usage and termination disposition, without private topology in public evidence.
2. Implement the materializer, single-unit worker, supervisor, status command and aggregator.
3. Run adversarial tests and a two-unit CPU restart smoke.
4. Run one bounded CPU/GPU throughput comparison.
5. Publish the measured campaign ETA and proposed fleet mapping.
6. After Musashi reproduces the runtime package, restart the predeclared positive-skill screen with fresh attempt identities.
7. Continue the fill-lineage correction in parallel; it does not require this screen.

## Status contract

Every status report must state, without internal finding codes as a substitute for facts:

- what scientific question is executing;
- exact current work unit;
- completed/total units by round;
- accepted, failed, timed-out and running counts;
- host/device class for each active unit, sanitized publicly;
- last durable completion time;
- measured median/p90 unit duration;
- ETA range and its assumptions;
- preliminary results clearly marked nonterminal;
- next automatic action.

“Process active” and aggregate CPU/GPU utilization are health observations, never progress evidence.

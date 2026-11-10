# 272. Memory-admitted distributed STL column queue

Owner authorized2026-11-10: replace fixed five shards with work-conserving CPU
dispatch across three existing workers. Experiment identity, parameters, data,
outputs and downstream GPU dependency gates remain unchanged.

## Requirements and tests before implementation

| Requirement | Acceptance / unit check |
|---|---|
| Keep finished work | Verify existing DONE file hashes; never schedule those columns |
| Preserve partial work | Pin each started column to its checkpoint host; resume in place |
| Prevent duplicates | Single coordinator flock; persist assignment before remote start; retain owner across restart/network failure |
| Use available CPUs | Dispatch next unassigned column to any host with a free admitted slot |
| Bound memory | Per-host cap, free-memory reserve and per-slot reservation; account for workers still starting |
| Survive restarts | Persistent per-column service and durable ownership ledger; idempotent start |
| No scientific changes | Call the existing pinned column runner; do not modify statistical code or prior evidence |
| Fail visibly | Bad receipts, ambiguous owners and runtime mismatch fail closed; failed units are reported |

Tests cover completed-column exclusion, disjoint assignment, partial ownership,
memory reservation during startup, unknown/offline owners and corrupted receipts.
Integration evidence must show old services stopped, monotone resumed row counts,
more active workers on all three hosts, and intact existing terminal receipts.

Each column remains single-threaded to preserve the pinned calculation. Parallelism
is across processes/columns. Separate subprocesses avoid Python-thread contention.
Provisioned maxima are6/8/8 processes on primary/secondary/auxiliary, reduced
automatically if available memory minus reserve cannot admit another slot.
Reserve3GiB on the smaller host,4GiB on the others; reserve1GiB per admitted
worker, including the not-yet-materialized portion of starting workers.
Per-process MemoryMax1.5GiB. These are resource limits, not model parameters.

Only the five owner-authorized old STL CPU queues are stopped. Running column
checkpoints are atomic every256 rows; an interruption may replay at most255
uncommitted rows. The original collector and automatic GPU queues remain.

## Deployment evidence

Five focused tests passed. The coordinator and per-column workers are persistent
user services. At handoff46 columns were complete with verified receipts;
22 assignments were durably recorded before launch:6 primary,8 secondary,8
auxiliary. The first status cycle observed47 complete, no failed column, no
network error or duplicate owner. Column30 resumed above row20000 and column46
above row23000; neither restarted its historical output from zero.
Available RAM after launch was approximately5.8/9.4/14.2GiB respectively.
Remaining unassigned work is admitted when any slot becomes free; once assigned,
a column stays on that host across coordinator or worker restarts.

Runtime configuration, topology and systemd units are private operator artifacts.
Coordinator status: `operations/DYNAMIC_STATUS.json`; ownership ledger:
`operations/DYNAMIC_ASSIGNMENTS.json`; terminal receipt:
`operations/DYNAMIC_DONE.json` below the existing STL evidence root.
The statistical worker runtime remains a0c8f542 and the experiment identity
remains04fb822af81df2708dae1fbf7bc1105e152760244d757e5d76a734b3ac6996b8.

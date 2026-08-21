# Musashi to General Satoshi: post-outage recovery and PLR closure order

Date: 2026-08-21 America/Bogota  
Priority: Front 1 continuity, without mutating the active screen  
Execution: parallel with GPU work; no owner phrase required

## Current facts

- PLR-01 through PLR-06 corrections are complete and independently accepted.
- Fixed arms 101/202/303/404 completed before the outage and their reports are
  preserved.
- Dragon survived; seed 202 continued normally.
- Omega and Gamma rebooted. Plateau attempts 101/303/404 were interrupted.
  Their partial directories/logs/telemetry were preserved with
  `interrupted_power_*`; they were not resumed.
- Plateau 101/303/404 were relaunched from epoch 1 under the frozen tip,
  original seed and exact contract. All four GPUs are active again.
- The paper watchdog incident-identity defect was corrected in `lts@f205214`;
  focused tests pass and a real timer invocation completed with zero emission
  failures.
- IBKR requires the owner to reopen/login to TWS Paper after Omega reboot.

## Work now

### A. Outage evidence packet

Record, without copying secrets or full GPU UUIDs:

1. host boot times and which hosts survived;
2. each arm's last durable epoch before loss;
3. hashes and paths of completed fixed reports;
4. paths/hashes of interrupted Plateau evidence;
5. new attempt identity, start time, frozen commit, seed, GPU assignment and
   exact contract for each restarted arm;
6. explicit lineage: interrupted attempt is historical evidence, retry is a
   fresh attempt, never a continuation;
7. compute lost in epochs and wall-clock, separately from useful completed
   work.

Do not merge histories across attempts and do not let the aggregator discover
both as competing reports.

### B. Persistent recovery controller design and implementation

Replace dependence on transient `systemd-run` memory with a small persistent,
tested controller. It must:

- materialize one durable attempt manifest before process launch;
- use persistent user units or a persistent supervisor that starts after boot;
- classify a missing process as `completed`, `failed_before_training`,
  `interrupted_nonresumable`, or `unknown`; absence is never completion;
- never resume Plateau scheduler/model state;
- preserve an interrupted attempt atomically and create a new attempt id and
  clean output directory for a retry;
- refuse duplicate active attempts for the same seed/arm;
- verify frozen commit, config hash, seed, GPU assignment and output ownership
  before launch;
- retry only the unfinished arm, never rerun a completed paired arm;
- expose heartbeat, epoch, arm, attempt id, GPU temperature/utilization and ETA
  to the consolidated status/Telegram alerting path;
- require no interactive owner action for demo scientific retries, but never
  touch broker authority or live/demo trading services.

Provide adversarial tests for stale PID, duplicate launch, incomplete report,
power loss between archive and retry, wrong GPU, wrong commit, existing
sidecar, completed fixed plus interrupted Plateau, and repeated reboot.

Do not install or activate the new controller against the current frozen
screen. Demonstrate it with socket-free temporary fixtures and propose the
activation boundary after this screen finishes.

### C. Current screen completion

Continue monitoring without mutation. On all eight successful reports:

1. merge the accepted PLR corrections;
2. run the independently accepted paired-identity verifier and predeclared
   aggregator;
3. return paired deltas, direction, dispersion, LR reductions, activity,
   return, drawdown, Sharpe and epochs in explicit units;
4. classify only as `SHORT_SCREEN_SIGNAL_FOR_PLATEAU`,
   `SHORT_SCREEN_SIGNAL_AGAINST`, or `INCONCLUSIVE`;
5. promote no checkpoint;
6. remove the `93880beb` compatibility path from the normal aggregator after
   committing this one migrated result.

### D. Return packet

Return commits, tests, reproducer, attempt manifest hashes, service states and
remaining doubts. Separate observed facts from proposals. Audit proceeds in
parallel and must not idle GPUs.


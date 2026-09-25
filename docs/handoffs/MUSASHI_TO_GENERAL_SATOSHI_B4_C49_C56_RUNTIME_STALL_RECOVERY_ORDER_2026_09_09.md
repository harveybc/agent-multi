# Musashi to General Satoshi: B4 C49-C56 runtime-stall recovery

Priority: P0

Audit input:
`docs/audits/MUSASHI_B4_V7_RUNTIME_STALL_AND_QUARANTINE_2026_09_09.md`.

## 1. Governing disposition

The B4 v7 service is stopped. Cell `o2022_seed303` is quarantined as
`AMBIGUOUS_CLAIM`; two earlier cells remain `COMPLETED_VERIFIED`; nine cells
remain pending. The GPU is free.

At the next safe commit boundary, preserve the current M4 C31A-C31F worktree
and switch to this order. Do not discard or rewrite that work. Return this P0
before resuming M4.

This order authorizes CPU-only design, implementation, tests and bounded
mechanics probes. It does not authorize GPU use, a B4 retry, a fourth cell,
scientific scoring, sealed-2025 access, promotion, service installation or
campaign launch.

## 2. C49: Freeze the exact failure

1. Consume the audit input by exact bytes and commit its identity in the PRE.
2. Reproduce from the preserved v7 root that the third cell has claim + lease,
   no terminal, no seal, and adjudicates `AMBIGUOUS_CLAIM`.
3. Prove the two completed terminal/seal digest pairs match the audit input.
4. Freeze the stale-status facts: epoch 111, 2,231,000 progress steps, last
   write time, 43,200-second cell ceiling and operator-stop time.
5. Preserve `CAMPAIGN_STOP` and the cell `STOP`; no test may mutate the real
   root.

## 3. C50: Reproduce callback starvation

Build a deterministic subprocess test in which the child:

- enters the same parent-to-cell execution seam as B4;
- stops calling the in-process callback after observable progress;
- ignores the stop file and exceeds a short test wall;
- remains alive until an external supervisor acts.

The PRE must show that the current in-process wall/stop callback cannot bound
this state. Do not use a sleep-only fake that bypasses the production process
boundary; exercise the real orchestration seam with a controlled child body.

## 4. C51: Put every cell in a supervised child process

Refactor the B4 orchestrator so the campaign owner process never executes
`model.learn` in its own process. One cell attempt runs in one child process
with a capability bound to:

- generation, cell and attempt;
- materialization and exact cell config;
- authorization and recovery record;
- parent PID/session identity;
- per-cell wall deadline and remaining global GPU budget.

The parent must use a monotonic deadline independent of callbacks, Python's
GIL, model code, CUDA progress and child telemetry. No epoch, patience,
timesteps or plugin configuration may weaken it.

## 5. C52: Bounded stop escalation and reap

When the hard wall, campaign stop or telemetry-stall rule fires, the parent
must execute and persist this ordered protocol:

1. stop request written durably;
2. bounded graceful interval;
3. `SIGTERM` if still alive;
4. second bounded interval;
5. `SIGKILL` only if still alive;
6. mandatory `waitpid`/reap and proof that the process group is empty;
7. CUDA-process inventory after reap;
8. no scheduling of another cell.

The parent may write a supervisor incident record. It must not fabricate a
scientific terminal on behalf of a child. A child terminal is accepted only if
it was durably complete before termination and passes the existing full
verifier; otherwise the attempt is `QUARANTINED_RUNTIME_STALL` or
`QUARANTINED_EXTERNAL_STOP`, never `COMPLETED`.

## 6. C53: Independent liveness and accounting

1. Add a parent-observed heartbeat/progress contract with monotonic sequence,
   child identity and bounded maximum silence. Choose and justify the timeout
   before the recovery mechanics result.
2. A busy CPU or GPU is not progress. Unchanged progress plus expired silence
   must stop the child.
3. Close GPU accounting at the externally observed reap time even when the
   child wrote no terminal. The close fact must be a separate append-only
   supervisor record; it may not rewrite the v7 claim or lease.
4. Recompute all prior and current-generation charges from durable intervals.
   Never use the rounded 44.69/51.30 summary as an input.
5. Failed and quarantined time counts against the global ceiling.

## 7. C54: Recovery generation

Prepare, but do not launch, a new append-only B4 generation that:

- supersedes v7 while preserving its complete history;
- changes only runtime supervision, liveness and accounting identity;
- carries `scientific_change: NONE`;
- starts every rerun from its original zero-update genesis;
- does not reuse any artifact from the quarantined third cell;
- keeps the same twelve cells, order, data, costs, comparators, seeds,
  observation contract, model recipe and decision rule;
- imports the two completed v7 cells only if the final verifier proves their
  full bindings and exact bytes; otherwise rerun decisions remain external.

The recovery generation must remain launch-closed until Musashi reviews it and
the owner separately authorizes the resulting dispatch scope.

## 8. C55: Acceptance battery

At minimum, prove all of the following with real subprocesses where stated:

1. child stops consuming callbacks but the parent terminates and reaps it;
2. child ignores the stop file and requires escalation;
3. graceful stop produces a valid typed child terminal when genuinely written;
4. partial/missing/late terminal remains quarantined;
5. a terminal written after deadline cannot become `COMPLETED`;
6. `CAMPAIGN_STOP` prevents the next claim;
7. restart sees the quarantined attempt and never resumes it automatically;
8. no child or process-group member survives;
9. GPU time closes once and includes failed time;
10. removing the external wall guard reopens the frozen starvation PRE;
11. deleting the liveness guard admits a busy-but-stalled child;
12. changing any scientific field in the recovery generation refuses;
13. the two completed v7 digest pairs remain byte-identical;
14. dry-run and mechanics probes write nothing to the real v7 root.

Include mutation tests for the parent deadline, terminal classification,
process-group reap, accounting close and no-next-cell rule.

## 9. C56: Return and stop line

Return one packet with:

- PRE and POST outputs;
- root-cause statement at the exact call boundary;
- process timeline and escalation evidence;
- tests and mutation counts read from final-tip output;
- proposed recovery-generation identities and executable diff;
- exact remaining GPU budget re-derived from durable records;
- confirmation that M4/T2 and both completed B4 cells were untouched;
- an exact launch command that still refuses without a new external record.

Required terminal disposition:

`B4_V7_QUARANTINED_AND_EXTERNAL_WATCHDOG_RECOVERY_READY_FOR_MUSASHI_REVIEW`

Stop there. No GPU dispatch and no campaign service.

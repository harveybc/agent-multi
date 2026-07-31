# Codex Audit Finding Closure

Closure ID: CODEX-AUDIT-CLOSURE-20260730-01
Timestamp and timezone: 2026-07-30 22:25 America/Bogota
Verifier: Musashi (Codex technical lead), independent of the Satoshi reporter
Requested by: user
Scope: independently reproduce and close findings opened by
`AUDIT_BOOTSTRAP_2026_07_30.md` and `AUDIT_STATUS_2026_07_30.md`
Excluded scope: no broker orders, canaries, campaign restart, blockchain
mutation or Hermes authority change

## Provenance

| Repository/system | Evidence |
| --- | --- |
| `lts` | `12d389de83d3995c9150799f7a674fe636c29a03` |
| `agent-multi` status/recovery correction | `2617f4ccf62ed9e675087141f77127496d0e34a2` |
| Active campaign | plan `phase-1-protected-execution-fleet-v2`, hash `b43844a7ebd7c85a782c557a8c3459622e1cb353a5d33391816e85f107cb6b21` |
| IBKR observer | TWS Paper `7497`, read-only, API `2.1.0`, server `178` |

## Closures

### AUD-F2-20260730-004

State: `verified_closed`

The defect reproduced: TCP reachability could be reported as IBKR availability
while the authenticated observer failed. After the user accepted the TWS Paper
API disclaimer:

- a manual preflight and the systemd-managed preflight completed;
- all six configured contracts qualified;
- positions, open orders and submitted orders remained zero;
- the timer remained active at five-minute cadence;
- the watchdog was changed to require a recent completed session joined to its
  reconciliation snapshot;
- socket reachability remains a separate diagnostic;
- a nonblocking file lock rejects concurrent functional preflights;
- missing, stale and unexpected-exposure states have regression coverage.

The client-ID-collision hypothesis in the source report was not reproduced. A
fresh client ID received the same disclaimer followed by the generic secondary
message before acceptance. It is not retained as a root cause.

Verification:

```text
LTS full suite: 205 passed, 1 dependency deprecation warning
IBKR completed reconciled sessions at verification: 222
IBKR positions/orders: 0/0
watchdog active events: mt5_bridge_missing, oanda_practice_not_configured
IBKR observer event: none
```

### AUD-GEN-20260730-001

State: `verified_closed`

Document 13 now records protected v2 as running on four workers, changes the
deployed contracts from `pending_deploy` to `deployed_four_worker_running`, and
replaces already-completed deployment steps with the actual campaign handoff.

### AUD-F1-20260730-002

State: `verified_closed`

Document 13 now records approximately 1.73 fleet candidates/hour, a 10-14 day
full remaining budget subject to early stopping, and an explicit end-of-stage-1
decision point. The review examines throughput, diversity, activity, fitness
progression and invariants; it is not a positive-profit promotion gate.

### AUD-GEN-20260730-003

State: `verified_closed`

The Codex technical-lead recovery prompt is now version 1.1.0. Its broad load
includes documents 08, 11, 14 and 16, and its snapshot states that protected v2
is running and that Alpaca/IBKR observers are authenticated.

## Runtime Note

At verification, the four campaign workers shared plan, job, domain, seed,
genesis, generation, shared population fingerprint, component revisions and
finalized anchor, with four distinct claims. A warning remained because Dragon
reported a different unfinalized tip at the same height. This closure does not
classify or repair that warning; stronger evidence is required before any chain
mutation.

## Remaining Observation

`OBS-20260730-C` remains open: the compact audit path does not yet collect
per-worker GPU utilization/temperature, RAM and swap. Existing operational
watchdogs continue to own alerts, but the next audit snapshot should ingest
their sanitized machine evidence instead of inferring health from silence.

## Safety Result

Alpaca and IBKR remain observation-only. No canary or broker order was enabled
or submitted by this closure.

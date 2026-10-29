# Crossed panel: implementation and launch

Date: 2026-10-29. Implementation pin: `e889a2d4`; native engine: `e0a1c2b`.
Plan: [140](140_CROSSED_PANEL_AUTOMATION_PLAN.md).

Full-run identity:
`4415bd3fdab307bf0869051a37f8a69035d1a727a6b642cb9afd880aa0725455`.

## Verification before full launch

- 162 focused tests pass with zero skips, including exact serial/sharded native
  account equality, independent price/cost/quality transformations, disjoint
  smoke/scientific seed banks and the repaired native execution witnesses.
- End-to-end smoke completed tests, all35 control jobs,20x2 DEAP, finalist freeze,
  five-quality confirmation, fixed-size comparisons and result export.
  It evaluated60 distinct candidates. This is **SMOKE_ONLY**, not optimization
  evidence. The rerun returned COMPLETE_ALREADY_AUTHENTICATED, with no new fits.
- Ruff correctness checks and git whitespace checks pass.
- [Smoke manifest](../audits/evidence/crossed_account_queue_20261029/SMOKE_MANIFEST.json),
  [tests](../audits/evidence/crossed_account_queue_20261029/SMOKE_TESTS.json), and
  [completed smoke](../audits/evidence/crossed_account_queue_20261029/SMOKE_COMPLETE.json)
  are exported without private paths. Intermediate smoke identities are not
  scientific runs. The pre-search seed-isolation correction is recorded in140.

## Running chain

The enabled user service `crossed-account-v2-20261029.service` runs from a detached
immutable checkout and a separately pinned native engine. User linger is enabled,
so closing a shell does not stop the queue and the enabled service starts after
reboot. No transient interactive terminal owns the job.

1. Witness preflight: **162/162 passed on the actual runtime checkout**.
2. Frozen controls:35 policy/quality jobs,360 account cells each (12,600 accounts).
3. DEAP:20 population x300 generations x3 seeds, sequential candidates; only the
   correlated-error calibration rung is fitness. Four persistent CPU shards
   evaluate the scenarios of each candidate, with ordered exact aggregation.
4. Representatives frozen on distinct calibration objective vectors, explicit
   lexicographic tie rule; no choice from confirmation.
5. Confirmation:five quality rungs over a fresh360-cell bank, controls and frozen
   representatives, plus size=1 diagnostic versions where applicable.
6. Authenticated final JSON, compressed full account replays, behavior-class
   counts and human-readable trading tables. No financial training follows.

At the launch check, the service was RUNNING in controls,7/35 jobs completed,
zero automatic restarts, memory peak below0.7 GiB. This is a dated observation,
not a current-progress promise. Full-run outcome is not yet known.

## Resource and recovery policy

Four CPU processes, one numerical thread each; CPU quota400%, MemoryHigh4 GiB,
MemoryMax6 GiB, swap disabled for this service, priority nice10. No GPU is used.
The [service template](../../examples/systemd/crossed-account-v2.service.template)
contains no machine-specific paths.

An exclusive lock prevents duplicate coordinators. Source/config/version identities
must match on resume. Per-control results and seed completion receipts are sealed.
Partial seeds replay deterministic evolution against authenticated candidate
receipts; final reports and replays are checked on completed-run re-entry.
An abnormal process exit may restart up to the service's bounded start limit.
A typed scientific/integrity failure is recorded as FAILED, not hidden or turned
into a successful result. There is no automatic witness/threshold relaxation.

## Read-only status and results

```bash
systemctl --user show crossed-account-v2-20261029.service \
  --property=ActiveState,SubState,MemoryCurrent,MemoryPeak,NRestarts
python -m json.tool <output>/STATE.json
python -m json.tool <output>/PROGRESS.json
```

`STATE.json` identifies phase and progress. During DEAP, `PROGRESS.json` contains
recent seconds per full-panel candidate and a conservative remaining-search ETA
from the maximum evaluation budget. Confirmation time is additional and explicitly
excluded from that estimate. An accurate full-search ETA requires full-panel
measurements, not multiplication of the two-world smoke time alone.

`CONTROLS.md` appears when all calibration controls finish; `RESULTS.md` and
`COMPLETE.json` appear at final completion. Tables include return, worst DD,
orders, buy/sell counts, win rates by direction and early-close counts. A missing
denominator renders '-' rather than a fabricated success rate. Raw replay files
retain costs, native short-week Sharpe, exposure and every trade. Those short-week
Sharpe values are diagnostics, not reliable annualized estimates.

## Scope

The price bank has30 independent draws, not360 independent markets. Mirrors,
envelopes and spreads are paired transformations. Even the imperfect reports are
synthetic stress models, not evidence of learned forecasting skill. Neither this
launch nor eventual Pareto membership authorizes floating feature selection until
the strategy's new results and report-interface limitations have been reviewed.

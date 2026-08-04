# Satoshi III Live Alerting and Business-Evidence Audit

Date: 2026-08-04 America/Bogota
Auditor: General Musashi, temporary independent auditor
Subject: Satoshi III delivery through `agent-multi@f32f6bdb` and
`lts@9a8d568`
Disposition: **not accepted; corrections and missing business-evidence work
required**

This verdict does not authorize Live capital, an IBKR hold clear, a new Paper
order, a DOIN mutation or a finding closure. The owner remains the only
activation and closure authority.

## 1. Findings

| ID | Sev | Finding | Direct evidence |
| --- | --- | --- | --- |
| `AUD-F2-20260804-093` | S2 | A new `kill`/hold can land after resume's checks and before its transaction; resume then commits `halt=none`, erasing the newer safety event. | `lts/app/ibkr_l1_resume.py:217-285`; socket-free reproducer reports `applied=true, final_halt=none`. Existing tests at `test_ibkr_l1_resume.py:137-145,277-288` are sequential, not a two-connection/new-hold race. |
| `AUD-F2-20260804-094` | S2 | The resume capability is not owner-authenticated. The confirmation phrase is public, the JSON is unsigned, and a PTY is not proof of a human owner; an automated same-user process can mint or write the payload and invoke the CLI. | `lts/tools/mint_resume_capability.py:33,59-67,81-108`; `lts/tools/ibkr_resume_after_reconciliation.py:143-164`. Do **not** run the requested CLI. |
| `AUD-GEN-20260804-095` | S2 | Incident redaction fails for normal JSON keys such as `"secret":`, `"api_key":`, `"password":` and nested `"token":`; those values can enter SQLite and Telegram unchanged. | `tools/incident_ledger.py:61-69,148-151`; reproducer reports `unchanged=true`. |
| `AUD-GEN-20260804-096` | S2 | Worker forwarding keys are not bound to an immutable host/source identity, while `recover()` accepts any non-empty object without source timestamp, freshness or producer binding. A worker key can impersonate another producer or resolve its incident with `{"ok":true}`. | `tools/incident_forward_shim.py:23-60`; `tools/incident_router.py:241-273`; `tools/incident_ledger.py:330-360`; both installed forced commands have no host binding; reproducer resolves P0 with unbound evidence. |
| `AUD-GEN-20260804-097` | S2 | A successful SSH ingestion is recorded by the worker as notification delivery. If Omega's SSH/ledger works but its router or Telegram transport fails, workers never fail over and the owner receives no P0. | `tools/incident_router.py:320-347`; `examples/configs/incident_ledger_v1.json:12` also sets P0 failover to 600 s, above the preregistered 60 s target. |
| `AUD-F2-20260804-098` | S3 | The consolidated status contradicts current execution truth: its account section says all venues are read-only, Alpaca has submitted zero orders and write mode exists nowhere, while fresh watchdog/bridge evidence says all three routes are write-enabled and Alpaca retains four Paper round trips. | `tools/multifront_status.py:317-343`; independent output at 19:17 UTC. This blocks honest business comparison. |
| `AUD-F2-20260804-099` | S3 | IBKR's broker effect is `terminal_flat` and direct broker evidence is flat, but its L0 exposure remains open. Capacity and alert severity therefore remain inconsistent with broker truth. | Delivery §6.2 plus fresh direct 0-position/0-order evidence. Must close through the accepted idempotent L0 API, never by SQLite editing. |
| `AUD-F2-20260804-100` | S3 | After TWS 1100/1102, cached order/position facts contradicted the filled flatten for about 27 minutes; no explicit authoritative source hierarchy or cache-invalidation state exists. | Delivery §4.1-4.3. Recovery stayed fail-closed, which was correct, but unattended reconciliation lacks a bounded convergence rule. |
| `AUD-F2-20260804-101` | S3 | Alpaca daily order-budget exhaustion raises an exception and degrades/restarts the runner instead of producing a durable lineage-bound decision outcome. | `lts/app/alpaca_l1.py:266-267`; delivery §6.4. |
| `AUD-F2-20260804-102` | S2 | One Alpaca bar produced four protected Paper round trips because repair identities minted new retries. The historical violation is real; corrections at `a9b9d41` and `9a8d568` pass focused/full tests and fresh runtime shows no fifth submission. | State: `independently_verified_pending_owner_closure`; retain the four round trips as evidence. |
| `AUD-F2-20260804-103` | S3 | IBKR construction backoff catches every `Exception`, so account/config/security/programming failures can be mislabeled as transient connectivity and retried indefinitely. | `lts/app/ibkr_model_runner.py:326-355`. Retry only an explicit transient taxonomy; fatal errors need an advancing terminal/degraded heartbeat plus P0/P1 incident. |

## 2. Independent Reproduction

Canonical reproducer:

`docs/audits/evidence/SATOSHI_III_LIVE_ALERTING_REPRO_2026_08_04.py`

Observed results:

```text
resume_clears_racing_kill: applied=true, final_halt=none
json_secret_redaction: unchanged=true, contains_test_values=true
arbitrary_recovery_evidence: pending -> resolved with {"ok": true}
```

No network, broker, Telegram, SSH or production database is touched.

## 3. Verified Work Worth Preserving

- `lts`: **583 passed** independently in `trading-stack`.
- `agent-multi`: **483 passed** independently in `trading-stack`; running from
  base fails collection because the packet did not state its environment.
- TWS-down construction now emits advancing degraded heartbeats and reconnects
  without systemd churn. Finding 091 is corrected and independently verified,
  pending owner closure.
- Project 3 is terminalized at 16,019 jobs with retained OLAP/snapshot evidence
  and no active Project 3 schedules on the three hosts.
- DOIN remains active on one campaign, one finalized anchor and one tip; no
  campaign mutation or worker restart was caused by this delivery.
- The incident ledger collapses duplicate observations, persists state and
  separates acknowledgement from recovery in its normal local path.
- TWS Paper and Alpaca Paper runners are active. The MT5 bridge/runner are
  active on Dragon and report `read_only=false`, `execution_enabled=true`.
- MT5's `max_concurrent_positions` refusal is not stale: direct SQLite evidence
  shows one real `ETHUSD` 0.01 short with native SL and TP. It should be
  monitored/closed by its model lifecycle, not bypassed.
- Dragon's `lts` checkout remains at `44bb639`; relevant runtime revisions and
  an explicit per-service manifest must be synchronized before fleet
  acceptance.

## 4. Hard-Gate Disposition

Gates 3, 4, 7 and 10 are not satisfied: the claimed owner-only resume is not
technically owner-only, JSON redaction leaks secret-shaped values, recovery and
status can assert false facts, and deployed revisions are not fully recorded
and synchronized. The current `halt=hold` must remain.

No evidence indicates a Live-capital order, an unprotected accepted entry, a
DOIN mutation or destructive evidence loss.

## 5. Rubric Score

| Criterion | Score / 5 | Weighted points |
| --- | ---: | ---: |
| Mission alignment | 2.0 | 6.0 / 15 |
| Trading correctness | 3.0 | 9.0 / 15 |
| Security/privacy | 2.0 | 4.0 / 10 |
| Reliability/continuity | 3.0 | 9.0 / 15 |
| Observability | 2.0 | 4.0 / 10 |
| Code quality | 3.5 | 7.0 / 10 |
| Testing/auditability | 4.0 | 8.0 / 10 |
| Data/business validity | 2.0 | 2.0 / 5 |
| Efficiency | 3.5 | 3.5 / 5 |
| Innovation/root cause | 4.0 | 2.4 / 3 |
| Communication/accountability | 4.5 | 1.8 / 2 |
| **Total** | | **56.7 / 100** |

Per the preregistered rubric this delivery is rejected in its current state.
That score recognizes substantial engineering and unusually honest disclosure;
it does not let strong tests compensate for failed safety/evidence gates and
the undelivered live-versus-simulation loop.

## 6. Required Continuation

The bounded work order is:

`../handoffs/MUSASHI_TO_SATOSHI_III_LIVE_ALERTING_CORRECTION_AND_BUSINESS_EVIDENCE_ORDER_2026_08_04.md`

Satoshi must return one correction/evidence packet. Musashi re-runs every
counterexample and the owner alone disposes findings or authorizes the first
post-recovery IBKR Paper risk.

## 7. Append-Only Runtime Correction, 2026-08-04 14:35 America/Bogota

The statement in section 3 that MT5's current refusal was backed by one
currently open protected position is withdrawn. It was derived incorrectly by
taking `MAX(position_snapshots.snapshot_id)` without joining it to the current
account snapshot.

Fresh direct facts show the actual sequence:

- one 0.01 `ETHUSD` short opened at 2026-08-03 23:47:14 UTC with native SL and
  TP;
- its stop executed at 2026-08-04 12:38:14 UTC;
- current account snapshot 5036 reports zero positions, zero orders and a
  balance of 9,999.76 USD;
- reservation `rsv-610092ed3f4cbcc6` remains `active` and its lifecycle has
  only `requested`, so every later due bar is rejected as
  `max_concurrent_positions` despite the account being flat.

This opens `AUD-F2-20260804-104`: MT5 broker close events are collected but not
reconciled into L0 lifecycle/reservation state. It is the direct reason no new
MT5 trade has occurred. The historical trade remains valid evidence; the
system must repair the lifecycle idempotently rather than delete or manually
alter SQLite rows.

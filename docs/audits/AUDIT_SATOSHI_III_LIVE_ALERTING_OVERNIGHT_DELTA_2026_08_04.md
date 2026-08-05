# Satoshi III Live Alerting Overnight Delta Audit

Date: 2026-08-04 America/Bogota
Auditor: General Musashi, temporary independent auditor
Subject: delivery addendum through `agent-multi@5e811a64` and runtime/code
delta through `lts@c00f8a5`
Disposition: **IBKR runtime progress verified; delivery still not accepted**

This audit changes no account authority and authorizes no Live capital. It
does not close findings. The owner already exercised the unaudited Paper
resume path before this delta; this audit records the resulting facts without
retroactively accepting that authorization mechanism.

## 1. Executive Disposition

The overnight addendum contains genuine progress:

- the real TWS restart produced one incident activation and one recovery;
- the IBKR selected-model runner opened one autonomous Paper short for
  25,000 USD.CAD and direct broker facts show both native protection children;
- the prior ghost L0 exposure was closed idempotently by `lts@83cc286`;
- the incident ledger reports no active incidents at the audit sample.

It is not the correction packet ordered on 2026-08-04. The canonical
reproducer still breaks the resume race, structured redaction and recovery
authenticity contracts. The owner-authentication defect is unchanged. The
business-evidence loop, normalized per-due-bar facts, no-idle succession and
rolling 24-hour/7-day evidence product remain undelivered.

## 2. Independent Reproduction

The canonical reproducer was run again at `agent-multi@5e811a64` against
`lts@c00f8a5`:

```text
resume_clears_racing_kill: applied=true, final_halt=none
json_secret_redaction: unchanged=true, contains_test_values=true
arbitrary_recovery_evidence: pending -> resolved with {"ok": true}
```

Therefore findings 093, 095 and 096 remain directly reproduced. Finding 094
also remains open: `bc4970a` fixes profile loading, not human authentication.
The current public phrase/unsigned-payload/PTY mechanism must not be reused.

The delta reproducer
`docs/audits/evidence/SATOSHI_III_OVERNIGHT_DELTA_REPRO_2026_08_04.py`
also reports:

```text
input_state=decided, clock_observed=false, clock_recovered=true,
tws_healthy=true, reproduced=true
```

Focused tests:

```text
pytest tests/unit/test_ibkr_exposure_reconcile.py \
       tests/unit/test_tws_continuity_monitor.py \
       tests/unit/test_ibkr_l1_resume.py
37 passed, 1 warning

pytest tests/
591 passed, 1 warning
```

The green suite proves the implemented examples. It does not contradict the
independent counterexamples above.

## 3. Direct Runtime Facts

### 3.1 IBKR Paper

Direct `ib_async` facts show one USD.CAD position of `-25000` units and two
working buy protection children for the same 25,000-unit parent: one limit
take-profit at `1.40247` and one stop at `1.40881`. The selected-model runner
is advancing a fresh `monitoring` heartbeat and the effect is acknowledged
with cumulative fill `25000` and `position_reconciled=true`.

The old ghost exposure was closed at 2026-08-04T22:06:22Z while the new
exposure remains open and bound to the acknowledged effect. This independently
verifies the correction for finding 099. It is now eligible for owner closure;
finding 100 remains open because the correction still consumes the client's
cached `position_facts()` as its direct-flat predicate.

The read-only lab's displayed account fingerprint differs from the runner
profile by design: the lab fingerprints the already-redacted account set once
more. This is the previously documented finding-068 representation issue, not
evidence of a second account.

### 3.2 Alpaca Paper

The direct Paper preflight is healthy and reports zero positions and zero open
orders. The accepted execution ledger nevertheless retains active reservation
`rsv-4adc1c4cbcb756ee` from the defect-era SPY decision while its associated
broker effects are terminal-flat and exposures are closed. That orphaned
capacity prevents the next eligible bar from entering through the normal
`max_concurrent_positions` gate.

This is not a reason to bypass the gate or submit a test order. It requires an
append-only idempotent lifecycle repair using direct flat broker evidence.

### 3.3 MT5 Demo

Dragon's execution bridge is current, fresh, authenticated, Demo-only,
write-enabled and flat: zero positions and zero orders. It retains active
reservation `rsv-610092ed3f4cbcc6`, no exposure and only the original
`requested` lifecycle event. Finding 104 therefore remains the direct blocker
to continuous MT5 trading.

The correct services are now stable: `lts-mt5-execution-bridge` and the model
runner are active with zero restarts. The obsolete
`lts-mt5-bridge.service` had remained enabled and was restart-looping against
the occupied port; `lts@29d6f6c` and `lts@76b2afc` retire and mask it. That
correction requires independent verification by Satoshi before owner closure.

## 4. New Findings

| ID | Sev | Finding | State |
| --- | --- | --- | --- |
| `AUD-F2-20260804-105` | S3 | Alpaca is broker-flat, but an orphan active L0 reservation from the defect-era retry sequence has no active exposure/effect and blocks continuous selected-model Paper trading. | open; Satoshi implements; Musashi verifies |
| `AUD-F2-20260804-106` | S3 | `c00f8a5` treats any fresh heartbeat with both `timeframe` and `last_closed_bar` absent as clock-healthy without validating `state`. A malformed `state=decided` heartbeat independently recovered `decision_clock_stale`; only an explicitly validated monitoring state with coherent direct route facts may omit the inference clock. | open; Satoshi implements; Musashi verifies |
| `AUD-F2-20260804-107` | S3 | The retired MT5 read-only unit remained enabled and restart-looped against the execution bridge's occupied port, creating false service churn and ambiguous runtime ownership. | corrected by Musashi at `lts@29d6f6c`/`76b2afc`; Satoshi independently verifies; owner closes |

## 5. Required Continuation

Satoshi must execute the existing correction order plus its appended delta:

`docs/handoffs/MUSASHI_TO_SATOSHI_III_LIVE_ALERTING_CORRECTION_AND_BUSINESS_EVIDENCE_ORDER_2026_08_04.md`

Priority is:

1. preserve and monitor the currently protected IBKR Paper position without
   replaying the flawed resume mechanism;
2. correct 093-098, 100-101, 103 and 106;
3. repair 104 and 105 through accepted idempotent lifecycle APIs so Alpaca and
   MT5 can resume due-bar trading naturally;
4. independently verify 107;
5. deliver C1-C4: normalized due-bar facts, deterministic live-versus-sim
   replay, no-idle champion succession and rolling 24-hour/7-day evidence.

The addendum's new IBKR trade is useful business evidence. It does not make an
unfinished evidence system complete.

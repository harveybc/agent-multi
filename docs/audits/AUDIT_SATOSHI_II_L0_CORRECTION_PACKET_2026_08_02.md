# Audit: Satoshi II L0 Correction Packet

Date: 2026-08-02 America/Bogota
Auditor: General Musashi, temporary independent auditor
Scope: `trading-contracts@e068bb5`, `lts@9fe9b64`
Runtime mutation: none

## Findings First

### AUD-F2-20260802-049 — S2 — Partial fills violate risk conservation

After a 40% partial fill of a 1% risk reservation, the open exposure plus
remaining reservation correctly totals 1% in `risk_active`, but `day_risk`
reports only the 60% unfilled reservation. The same logical position is also
counted twice (`positions=2`: exposure + remaining reservation).

Independent reproduction then admitted a second 0.4% order:

```text
risk_active_after_partial=0.0100
day_risk_after_partial=0.0060
second_order=would_be_order
risk_active_after_second=0.0140
configured_daily_cap=0.0100
```

Required correction: one position identity spans filled exposure and
remaining entry; daily worst-case risk includes both without double counting.
Repeated cumulative partial-fill reports must conserve the original risk.

### AUD-F2-20260802-050 — S2 — Signed and multi-asset exposure is corrupted

`apply_execution_event()` writes every exposure as instrument `USD.CAD`,
capability provenance `unknown` and positive `filled_units`. A short order of
`-10,000` is persisted as `+10,000`; flatten then emits `-10,000`, which would
increase the short rather than close it. An `ETH/USD` fill is persisted as
`USD.CAD`.

Required correction: persist signed open units, original asset/instrument,
venue/account and capability provenance from the immutable order decision.
Flatten delta is exactly `-signed_units_open`. Add long, short and non-FX
fixtures.

### AUD-F2-20260802-051 — S2 — Cancel intent cannot identify its target

The emitted cancellation uses asset and instrument `pending-entry`; its
canonical intent contains neither the original order-intent ID nor a broker
order identity. The adapter cannot deterministically cancel the intended
order.

Required correction: risk-reducing cancel contract carries the exact source
order-intent/attempt and, when observed, broker order IDs. Cancellation is
idempotent and reconciled against the target lifecycle.

### AUD-F2-20260802-052 — S2 — Cross-venue capability substitution accepted

An LTS service configured for `ibkr_paper` and its account fingerprint
accepted a capability snapshot labeled `alpaca_paper` with another account
fingerprint and returned `would_be_order`.

Required correction: service venue, account fingerprint and environment must
match the capability snapshot exactly. Synthetic evidence remains L0-only;
no cross-venue capability inference is permitted.

## Correction Dispositions

Independent suites:

- trading-contracts: 91 passed;
- LTS focused L0: 42 passed;
- LTS complete: 277 passed;
- atomic budget test repeated 20 times: 20 passed.

| Finding | Disposition | Evidence |
| --- | --- | --- |
| 039 | **verified_closed** | naked/ambiguous v2 entries reject; versioned v1 untouched; four exact tests independently pass |
| 040 | remains open | atomic race fixed, but partial-fill daily risk still breaches cap (049) |
| 041 | remains open | transition law improved; persisted exposure/target identity is still corrupt (050/051) |
| 042 | remains open | deterministic library path exists, but no authenticated runner and emitted actions remain unsafe |
| 043 | **verified_closed** | wrong-side long/short protection now persists a rejection before reservation |
| 044 | in progress | full fill remains risk-active, but partial fill violates conservation and position cardinality |
| 045 | **verified_closed** | `BEGIN IMMEDIATE` serializes check/write; 20 repeated races admit only one order |
| 046 | **verified_closed** | serialization failure rolls back and persists replayable rejection with zero leaked capacity |
| 047 | in progress | actions are emitted, but short flatten direction and cancel target are unsafe |
| 048 | open | no continuous runner/config/systemd deployment yet |

## Verified Non-Damage

- No broker write adapter or L1 activation exists.
- Direct venue orders and positions remain zero.
- The correction commits are pushed and their worktrees are clean.
- The DOIN campaign remains on its prior component lineage; new contracts
  were not hot-swapped into the running chain.

## Improvement: Stateful Execution Model Testing

Add a deterministic model/state-machine suite that generates long/short,
multi-asset, partial-fill, duplicate-event, cancel/fill race, restart and
cross-venue sequences. Assert after every event:

1. signed exposure conservation;
2. risk reservation + exposure conservation;
3. unique logical position cardinality;
4. venue/account/provenance identity preservation;
5. every cancel targets one existing order;
6. flatten moves exposure monotonically toward zero;
7. replay never changes state twice.

This directly advances open finding 010 and is cheaper than discovering each
state interaction manually.

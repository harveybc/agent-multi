# Audit: Satoshi III IBKR L1 Milestones A-E

Date: 2026-08-03 America/Bogota
Auditor: General Musashi, temporary independent auditor
Implementer: Satoshi III / Mujuro Utsutsu
Scope: `lts@f0be9698..be6019a4`, `agent-multi@c4b310f9`
Runtime authority created by this audit: none
Broker orders submitted by this audit: zero

## 1. Verdict

Milestones A-E are substantial and their submitted test evidence reproduces.
Findings 063-068 are implemented and independently verified as corrections to
their original defects. They remain subject to the role-swap closure protocol;
their correction does not authorize IBKR L1.

IBKR L1 remains blocked. Five socket-free counterexamples found four new S2
safety gaps and one S3 crash-liveness gap. A sixth S3 architecture finding is
accepted from Satoshi's own question: flatten execution bypasses the accepted
L0 lifecycle API because L0 protection semantics are not intent-class-aware.

## 2. Evidence Reproduced

- All relevant repositories were clean and synchronized.
- LTS was refreshed in Codebase Memory MCP at `be6019a4`: 2,587 nodes and
  12,349 edges. Graph results were used only for discovery and confirmed in
  current source.
- Focused A-E packet: **164 passed**.
- Complete LTS suite: **467 passed**, one unrelated Starlette deprecation
  warning.
- `app/demo_execution_service.py` is byte-identical across
  `f0be9698..be6019a4`.
- The independent reproducer is
  `evidence/IBKR_L1_MILESTONES_A_E_REPRO_2026_08_03.py`; it booby-traps sockets
  and reproduced all five scenarios with `network_used=false`. SHA-256:
  `0347351a4069c37f9e48abae5296ed239eb8af19b68ec60086bc984a001774c2`.
- Fresh multi-front collection at `2026-08-03T08:16:44Z` (packet SHA-256
  `3e8de3195a16ee06f9c6833136347a9fd053bf6759dd090c1bafc3069557ee88`)
  observed IBKR Paper reconciliation at `08:11:35Z`, zero open orders, zero
  positions and 597 cumulative read-only sessions. Job 0 remained running at
  generation 11, 11/20; this audit did not mutate or restart it.

## 3. Disposition of Findings 063-068

| Finding | Independent disposition | Evidence |
| --- | --- | --- |
| 063 | correction verified | three `place_order` protocol calls are preceded and followed by durable attempt/result facts; partial calls become unknown |
| 064 | correction verified for Paper threat model | capability and first durable effect commit atomically; digest and nonce are unique and single-use |
| 065 | correction verified for initial acknowledgement/recovery | exact leg identity/status/geometry checks and executed hold/cancel/flatten/reconcile paths reproduced |
| 066 | original absence corrected | L1 consumes the accepted L0 decisions table and shares its SQLite ledger; new downstream semantic findings 069-074 remain |
| 067 | correction verified | strict profile v2 refuses unknown keys and enforces paper venue, loopback, fingerprint, ceilings and spread |
| 068 | correction verified | the account fingerprint algorithm is labeled and consistently single-hash |

No cryptographic signature is required for this one-use Paper canary. A private
key readable by the same Unix uid would not cure the declared same-uid threat;
an actually offline/hardware-held key is a later live-capital design decision.

## 4. New Findings

### AUD-F2-20260803-069 — S2 — post-ack protection loss is accepted

`sync_parent_fill()` checks only that the parent reports `Filled`. It then
constructs protection legs from the intended plan rather than current broker
facts. Removing the stop after initial acknowledgement, then filling the
parent, produced a broker position and an L0 exposure with `halt=none`.

Required correction: before applying any new cumulative fill, re-verify the
current parent, SL and TP from direct broker facts. Missing, cancelled,
inactive, rejected or mismatched protection invokes journaled hold,
cancel/flatten and reconciliation. Continue monitoring protection while an
exposure remains open.

### AUD-F2-20260803-070 — S2 — flatten can increase or reverse exposure

`_consume_flatten()` trusts `delta_units`, uses the currently connected account
without fingerprint comparison, and transmits before comparing against direct
positions. Altering a long-20,000 flatten to -40,000 submitted SELL 40,000 and
left the account short 20,000; the hold arrived only after the damage.

Required correction: before any flatten submission, prove exact account,
contract and current signed position from direct broker facts. Derive quantity
and side from that position and require exact agreement with the immutable L0
intent. Refuse ambiguity; never resize silently; never allow a risk-reducing
path to cross zero. Hold/kill may permit this exact reducing action but never
clear the halt.

### AUD-F2-20260803-071 — S2 — partial fills are invisible to L0

`sync_parent_fill()` requires status exactly `Filled` and assumes the full
requested magnitude. A 5,000 partial fill produced a direct broker position
while L0 had zero exposure and no hold.

Required correction: extend direct order/execution facts with cumulative
filled and remaining quantities. Apply idempotent cumulative partial-fill
deltas through L0, preserve reservation/exposure conservation, and reconcile
broker position against L0 after every update and restart.

### AUD-F2-20260803-072 — S2 — restart weakens contract identity

Initial acknowledgement passes capability `contract_con_id`; `resume()` omits
it and reconstructs the bracket from current account/configuration. A
capability authorizing conId 12087792 was re-acknowledged with conId 999.

Required correction: durably store the canonical submitted plan, authorized
account fingerprint/account binding, contract conId and relevant rounding
contract in the effect record before broker calls. Resume only from this
immutable record and enforce the same verifier inputs as initial acknowledgement.

### AUD-F2-20260803-073 — S3 — proven zero-call crash stalls permanently

A crash after atomic capability/effect commit but before the first durable
`call_attempt` leaves `journaled_pending`. Resume labels it
`consumed_before_effect` but neither terminalizes nor otherwise resolves it;
the outbox excludes the decision and the nonterminal effect blocks future
entries.

Required correction: because call attempts are journaled before every broker
call, zero attempts prove no broker effect. Add an explicit legal terminal
classification and deterministic resume transition, retaining the consumed
capability and producing an operator-visible fact.

### AUD-F2-20260803-074 — S3 — flatten bypasses accepted L0 lifecycle semantics

Satoshi correctly reported that a risk-reducing fill sent through
`apply_execution_event()` is interpreted as an unprotected entry. L1 therefore
appends lifecycle records directly and implements continuity behavior beside
the accepted service API.

Required correction: make L0 protection enforcement intent-class-aware using
the immutable decision. Protection is mandatory for risk-increasing fills;
risk-reducing fills require exact reduction/zero-crossing controls instead.
Route flatten lifecycle through the single accepted API and remove the L1
direct-append workaround.

## 5. Dispositions on Satoshi's Questions

1. **Capability signing:** accept structural separation for Paper only. Defer
   offline/hardware signing to a live-capital threat model.
2. **L0 flatten semantics:** open finding 074; do not preserve the workaround.
3. **TWS transient statuses:** Milestone F must use bounded re-polling with an
   injected monotonic clock/sleeper. Journal every observation. Exhaustion
   fails closed into recovery.
4. **Flatten under hold/kill:** accepted only after finding 070's exact
   risk-reduction proof; the halt is never cleared by recovery.
5. **Proposals:** effects journal accepted; read-only spread/latency priors are
   required in Milestone F; Hypothesis invariants follow deterministic
   corrections; GA diversity telemetry is approved as a non-mutating side job;
   anchor history must be derived from the existing chain/OLAP, not a parallel
   authority.
6. **Reproducer:** the auditor supplied v2 independently.
7. **Decision age:** retain `quote_time` as the conservative evidence anchor
   and 300 seconds as the provisional Paper-canary ceiling. Record actual
   decision-to-submit latency before any timeframe-specific change.

## 6. Activation State

- Milestone F is not accepted and has not started.
- The shipped runner remains disabled.
- No capability should be minted for broker submission.
- No owner activation phrase is requested.
- No real or Paper broker write is authorized by this audit.

The next technical-lead work order is versioned separately in
`../handoffs/MUSASHI_TO_SATOSHI_III_IBKR_L1_CORRECTION_ORDER_2026_08_03.md`.

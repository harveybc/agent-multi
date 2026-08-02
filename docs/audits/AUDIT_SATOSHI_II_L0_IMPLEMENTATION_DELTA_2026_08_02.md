# Audit: Satoshi II L0 Implementation Delta

Date: 2026-08-02 America/Bogota
Auditor: General Musashi, temporary independent auditor
Scope: `trading-contracts@2b46c7e`, `lts@be1f576`
Mutation during audit: none in implementation repositories

## Verdict

Satoshi II delivered substantial, relevant implementation: versioned
execution contracts, a zero-network planning sink, a SQLite lifecycle ledger
and broad unit coverage. The direction is correct, both repositories are
pushed and clean, and the submitted suites independently pass:

- `trading-contracts`: 84 passed;
- LTS focused L0 suite: 25 passed;
- LTS full unit suite: 145 passed.

The implementation is not accepted for continuous L0 or any L1 canary. Four
independently reproduced risk defects remain, and no runner, deployment unit
or continuously advancing L0 ledger exists yet.

## Findings

### AUD-F2-20260802-043 — S2 — Market protection is not anchored to price

`OrderIntentV2` validates long `SL < TP`, but a market order carries no
decision reference price. `DemoExecutionService` uses `abs(reference-SL)` and
does not require the reference to lie between SL and TP.

Reproduction at reference `1.00`: a long intent with SL `1.01` and TP `1.02`
returns `would_be_order`. A nominal stop loss above the market is therefore
accepted as protection.

Required correction: persist the decision reference price and quote time/hash
and enforce long `SL < reference < TP`, short `SL > reference > TP`, before
reservation or serialization. Add both wrong-side counterexamples.

### AUD-F2-20260802-044 — S2 — Filled exposure disappears from risk totals

On `filled`, `apply_execution_event()` changes the reservation from `active`
to `consumed`. `active_totals()` counts gross, margin and positions only for
`active` rows, so a filled and still-open position immediately disappears
from aggregate exposure.

Reproduction with `max_concurrent_positions=1`:

```text
positions_after_filled=0.0
second_with_max_positions_1=would_be_order
```

Required correction: track open exposure independently or keep consumed
reservations risk-active until position close. Partial fills must reserve the
worst-case remaining entry plus filled exposure without double counting.

### AUD-F2-20260802-045 — S2 — Claimed atomic reservation is raceable

`active_totals()` and `reserve()` execute as separate transactions. Two
different service instances can both observe free budget and then commit.

Barrier-controlled reproduction: two 1% intents against a 1% daily cap both
returned `would_be_order`; persisted day risk was `0.02` with two positions.

Required correction for L0: validate and construct pure objects first, then
use one `BEGIN IMMEDIATE` transaction to re-read totals and atomically write
reservation, decision/outbox and initial lifecycle fact. Roll back the entire
unit on failure. L1 must use a durable outbox plus reconciliation because a
broker request cannot share the SQLite transaction.

### AUD-F2-20260802-046 — S2 — Post-reservation failure leaks risk

The service reserves before constructing `OrderIntentV2`. Invalid bracket
geometry then raises Pydantic `ValidationError`, leaving one active
reservation and no recorded decision.

```text
exception_type=ValidationError
active_reservations_after_exception=1.0
recorded_decision=None
```

Required correction: all contract validation and serialization that can fail
must occur before reservation, followed by the atomic transaction specified
for finding 045. Every rejection must become a persisted, replayable outcome;
no uncaught validation exception may leak capacity.

### AUD-F2-20260802-047 — S3 — Risk-reducing commands are declarative only

`flatten_all` and `cancel_pending` commands can be accepted but produce no
risk-reducing intent or lifecycle fact. Missing protection reports an
`unprotected_exposure_hold_and_flatten` string but emits no would-be flatten
action. `kill` only changes the halt value.

Required correction: L0 must emit deterministic zero-network cancel/flatten
intents and lifecycle evidence. L1 remains forbidden until those outputs are
independently verified and the command transport is authenticated.

### AUD-F2-20260802-048 — S3 — L0 is implemented as a library, not running

The LTS commit contains one module and tests. There is no CLI runner, resolved
JSON config, `systemd` unit/timer, live-feed adapter or running process. No L0
OLAP database was observed advancing continuously.

Required correction: complete the integrated packet and deploy only after
043–047 pass adversarial tests. Feed it current paper/demo observations; keep
the sink structurally zero-network. Show fresh ledger rows, service health,
restart recovery and zero broker submissions over a continuous window.

## Front Status at Audit Time

- F1: all four workers share one plan, job and chain; stage 2/4, generation
  10, 202/480 planned candidates complete (42.08%); current generation 2/20;
  recent swarm throughput 6.53 candidates/hour; no chain alert.
- Fleet GPU temperatures/utilization: Omega 50 C/39%; Dragon 47 C/38%; Gamma
  5070 Ti 53 C/47%; Gamma 5090 56 C/42%.
- F2: Alpaca paper active/read-only; MT5 heartbeat fresh/read-only; IBKR paper
  observer stale/offline; direct orders and positions remain zero.
- F3: 2,330 posts collected over 127 runs; publishing remains human-gated.

## Acceptance Boundary

No L1 broker-side demo write is authorized. Corrections 043–047, continuous
L0 deployment and independent evidence review precede the exact L1 canary
authorization packet.

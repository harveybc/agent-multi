# Continuous Demo-Trading Doctrine Audit

Audit task: `AT-F2-039`
Date: 2026-08-01 America/Bogota
Auditor: General Musashi, temporary independent auditor
Scope: `docs/work_plan/29_CONTINUOUS_DEMO_TRADING_OPERATIONS.md`, existing
cross-repository contracts and the proposed L0 boundary
Runtime mutation: none; zero orders submitted
Disposition: **reported_changes_required before L0 acceptance**

## 1. Executive Decision

The high-level doctrine is sound and should be retained:

- LTS is the sole order authority;
- no LLM, Hermes agent or social model may decide or submit an order;
- demo only until a separate owner/legal/capital gate;
- every risk-increasing order requires broker-side stop loss and take profit;
- model changes occur at declared boundaries, never mid-position by default;
- live facts constrain feasibility and costs, not alpha ranking;
- L0 must prove zero submissions before L1 can be considered.

The doctrine cannot yet be accepted as an implementable L0 contract. Four
fail-closed boundaries are underspecified or contradicted by current shared
contracts. These are cheaper to correct before an adapter write path exists.

## 2. Findings

### AUD-F2-20260801-039 - S2 - shared OrderIntent accepts naked entries

Observed in `trading-contracts/src/trading_contracts/contracts.py`:

- `OrderIntent.stop_price` and `take_profit_price` are optional;
- no validator requires both for a risk-increasing `delta_units`;
- `stop_price` is semantically ambiguous between a stop-entry trigger and a
  protective stop loss when `order_type="stop"`;
- current examples use it as protection for a market order, but the schema
  does not encode that distinction.

Reproduced: `OrderIntent.v1` successfully materializes a market buy with
`stop_price=null` and `take_profit_price=null`.

Impact: implementing L0 against this contract can produce a schema-valid but
owner-forbidden naked order. This is a hard L0 blocker even though current
runtime is read-only and therefore unaffected.

Required correction:

1. introduce an unambiguous protective bracket object or explicit
   `stop_loss_price` and `take_profit_price` fields;
2. distinguish entry trigger price from protective stop loss;
3. require both protective legs for every risk-increasing intent;
4. permit missing entry protection only for explicitly risk-reducing
   close/flatten/cancel operations;
5. validate side/price geometry and finite positive prices;
6. preserve backward compatibility through a versioned contract rather than
   silently changing `order_intent.v1` semantics.

Regression: naked market, limit and stop entries reject; close-only actions
remain possible; every accepted entry serializes both protection legs.

### AUD-F2-20260801-040 - S3 - rel_volume and portfolio loss reservation are ambiguous

Document 29 calls `rel_volume` 0.005-0.01 a fraction of balance "per
position" while also declaring gross exposure <=10%, at most three positions
and daily loss <=2%. Those are different dimensions:

- notional/balance;
- margin/balance;
- loss-at-stop/equity;
- daily realized-plus-reserved loss/equity.

For leveraged FX, a single `rel_volume` cannot represent all four. Three
independently sized 1% loss-at-stop positions can also reserve 3% while the
daily cap is 2%. Rounding to a venue minimum can breach any cap.

Required correction:

1. define `risk_fraction_at_stop`, `gross_notional_fraction`, margin cap and
   daily loss budget separately;
2. size units from stop distance, pip/tick value and conversion rate;
3. reserve worst-case stop loss atomically before submission;
4. include existing open risk and pending-entry risk in all caps;
5. never round up above a cap: skip the order when venue minimum size cannot
   fit;
6. release/adjust reservations deterministically on reject, partial fill,
   cancel and close.

Regression: adversarial minimum-size, three-position concurrency,
cross-currency conversion and simultaneous-intent fixtures cannot exceed any
declared cap.

### AUD-F2-20260801-041 - S3 - lifecycle contract cannot express partial and bracket states

`ExecutionReport.v1` has states `requested`, `accepted`, `filled`, `rejected`,
`modified`, `closed`; it lacks explicit partial-fill, cancel-pending,
cancelled, expired and unknown/reconciliation-required states. It also lacks
an explicit parent/child bracket identity and per-leg protection status.

Impact: after timeout, disconnect or partial fill, retry logic cannot prove
whether exposure exists or whether both protective legs cover the filled
quantity. Idempotency at intent creation is necessary but insufficient.

Required correction:

1. specify a persisted lifecycle state machine with legal transitions;
2. persist attempt identity before network submission and broker IDs on ack;
3. reconcile before every retry when acknowledgement is unknown;
4. model partial quantities and protection quantity per leg;
5. hold new risk and cancel pending entries on uncertain reconciliation;
6. define restart recovery from the ledger, not process memory.

Regression: duplicate replay, lost acknowledgement, partial fill before
disconnect, orphaned protection leg and restart fixtures result in no duplicate
exposure and no unprotected filled quantity.

### AUD-F2-20260801-042 - S3 - human command path is not yet isolated from Hermes/LLM processing

The doctrine says Telegram may accept exact-phrase hold/kill commands while
also saying no LLM may enter the order path. It does not yet require a direct,
deterministic command parser separate from Hermes model inference.

Impact: routing a free-form message through an agent before the deterministic
handler would contradict the authority boundary and create spoof/replay
ambiguity, even when the requested action is risk-reducing.

Required correction:

1. use a dedicated deterministic command endpoint/bot path with an owner
   identity allowlist, command schema, expiry, nonce and idempotency;
2. allow only risk-reducing commands before a separately reviewed control
   expansion;
3. Hermes may explain or report state but cannot originate, transform or
   approve the command;
4. persist command receipt, authorization result and resulting deterministic
   state transition;
5. fail closed when Telegram/Hermes is unavailable; local watchdogs must still
   halt independently.

Regression: spoofed sender, replayed command, stale command, malformed phrase
and unavailable model all fail without increasing risk; an authorized hold
still works when the LLM service is down.

## 3. Required Doctrine Amendments Without Separate Findings

These are acceptance details to incorporate while correcting the four
findings:

- L1 long and short canaries are sequential with flat/reconciled state
  between them; do not rely on simultaneous hedging in a netting account.
- L2 requires both elapsed time and event/scenario coverage. Seven days with
  zero lifecycle events is not an acceptance result.
- Market entry is the first L1 order type. Limit and stop-entry support enter
  only after expiry, cancellation, replacement and gap-through behavior have
  dedicated fixtures. This preserves the owner's multi-order-type objective
  without multiplying the first safety boundary.
- Kill state is explicit: reject new risk, cancel pending entries, reconcile
  open positions and preserve broker-side protection. Flatten only when the
  venue is reachable and the configured policy says flatten is safer.
- Calibration packets are stratified by venue, asset, order type, size,
  session and regime. Their observation cutoff must precede any optimization
  evaluation period they influence. Policy-conditioned fill samples are not
  universal cost truth.
- L0 drives the real adapter serialization boundary through a zero-network
  sink and proves `submitted_count=0`; merely constructing an in-memory DTO is
  insufficient.

## 4. Technical Assignment to General Satoshi II

The requested engineering assignment is accepted as useful work allocation,
not punishment:

**Build the adversarial L0 contract-first fixture packet**, after returning
the bounded finding-037 correction. It shall cover mandatory protection,
minimum-size overshoot, atomic risk reservations, stale signals, duplicate
replay, lost acknowledgements, partial fills, disconnect/restart recovery and
deterministic command isolation.

Constraints:

- CPU-side and zero-network by default;
- no broker write path enabled;
- no active campaign/config/service mutation;
- reuse `trading-contracts`, LTS and prediction-provider interfaces before
  adding abstractions;
- version contract changes and export schemas/examples;
- focused tests first, owning-repository full suites second;
- return commits, commands, hashes, counterexamples and known limitations;
- do not self-close any finding.

## 5. Acceptance Gate

`AT-F2-039` becomes eligible for independent closure only when findings
039-042 have correction evidence and the adversarial fixture packet passes.
L0 may be developed under zero-submit constraints, but it is not accepted
until that verification. L1 remains a separate hard owner gate with exact
venue/account/symbol/direction/risk/time-window authorization.

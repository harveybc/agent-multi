# Satoshi II L0 Integrated Packet Audit

Date: 2026-08-02
Auditor: General Musashi during `ROLE_SWAP_ACTIVE`
Subject commits:

- `trading-contracts@cd05083` (including `e068bb5`)
- `lts@8e25609` (including `6af0300` and `9fe9b64`)
- `prediction_provider@3a6c234`
- `agent-multi@6133dc26`

Requested packet:
[SATOSHI_II_L0_INTEGRATED_PACKET_2026_08_02.md](/home/harveybc/Documents/GitHub/agent-multi/docs/handoffs/SATOSHI_II_L0_INTEGRATED_PACKET_2026_08_02.md)

## 1. Verdict

`AT-F2-040`: **reported_changes_required**.

The packet contains real, useful work. Findings 040, 044 and 049-052 are
independently verified closed. The continuous process exists, advances its
heartbeat, consumes a live Alpaca Paper quote stream and has no TCP/UDP
socket or broker-submission path. The mechanics module and its golden fixture
also reproduce.

L0 is not accepted yet. Independent adversarial execution found three S2
coordination failures and three additional runtime/provenance gaps. In
particular, the deployed runner produced one protected would-be order, left
its reservation active forever, and then began replaying one cap rejection
per bar. It therefore does not continuously exercise long/short order,
lifecycle, protection and reconciliation mechanics as required by document
29.

No L1 broker write is authorized. No broker order was submitted during this
audit.

## 2. Reproduced Positive Evidence

All repositories were clean and equal to their pushed branches before the
audit.

| Surface | Independent result |
| --- | --- |
| `trading-contracts` | `95 passed` |
| `prediction_provider/mechanics` | `16 passed` |
| LTS focused L0 | `60 passed` |
| LTS complete | `295 passed` |
| `agent-multi/tests/unit` | `429 passed` |
| Additional generated execution traces | 200 seeds, 6,000 events, zero invariant failures |
| Cumulative partial fill | 25/50/75/100% preserved risk, day risk and one logical position |
| L0 process | active since 16:02 COT; heartbeat advanced across an independently observed 60-second interval |
| L0 process FDs | SQLite plus one Unix `journald` socket; zero TCP/UDP sockets |
| Installed systemd unit | byte-identical to repository unit, SHA-256 `40357d991ff82e9e063eece54d8c7fe59a2d43820ae8afdff050a66c9b95de6a` |
| Broker counts | Alpaca/IBKR/MT5 directly reported zero open orders and zero open positions |
| TWS Paper | re-authenticated after packet creation; read-only API preflight succeeded with six priced cells and zero orders/positions |
| DOIN | four workers on one job, generation, population fingerprint, tip and finalized anchor; no alerts |

### Verified corrections

- **040:** separate loss-at-stop, gross, margin and daily-risk dimensions;
  `BEGIN IMMEDIATE` reservation accounting; hard floors never override caps.
- **044:** open exposure remains in risk totals independently of order state.
- **049:** immutable original fractions conserve cumulative partial-fill risk
  and logical position cardinality.
- **050:** signed units and immutable asset/instrument/venue/account provenance
  persist; short flatten has the correct positive delta.
- **051:** cancel/flatten contracts name their exact target intent.
- **052:** capability venue/account/environment must match the service.
- The provider mechanics policy is deterministic and golden-byte stable; the
  artifact loader verifies SHA-256 before deserialization.
- The parent `prediction_provider` package really does export a top-level
  `app`, so the isolated `prediction-provider-mechanics` distribution avoids
  a demonstrated shared-environment namespace collision.

## 3. New Findings

### AUD-F2-20260802-053 (S2): concurrent identical intents are not idempotent

`DemoExecutionService.process_intent()` reads `recorded_decision()` before
entering the atomic unit. Two service instances can both observe absence. The
first commits; the second attempts to persist the same primary key and raises:

```text
IntegrityError: UNIQUE constraint failed: decisions.idempotency_key
```

Independent reproduction returned one `would_be_order` and one exception.
The duplicate must replay the committed decision, never crash. Move the
idempotency check into the same `BEGIN IMMEDIATE` unit as check/reserve/write,
or catch the uniqueness race and reload the canonical result. Retain the
two-connection synchronized reproduction as a regression.

State: **open, blocks L0/L1 acceptance**.

### AUD-F2-20260802-054 (S2): lifecycle transition validation races the ledger

`apply_execution_event()` reads `last_state()` before `BEGIN IMMEDIATE`.
Two reports both citing `previous_state=requested` can pass concurrently. The
auditor forced `filled` to commit first and `accepted` second. Both returned
success; the ledger became:

```text
requested -> filled -> accepted
```

The final order state was `accepted` while a fully open exposure existed.
Re-read and validate the current state inside the same transaction that
appends the event and mutates reservation/exposure state. Add the synchronized
two-connection reproduction and transition-order invariant.

State: **open, blocks L0/L1 acceptance**; finding 041 remains open.

### AUD-F2-20260802-055 (S2): accepted kill cannot resume after a crash

`apply_owner_command()` persists the command as accepted before emitting
flatten/cancel intents. With an injected failure in `_emit_flatten_all`, the
ledger retained `accepted=1` and `halt=kill`; retrying the exact command
returned `nonce_replay` and did not resume its missing side effects.

The risk-increasing hold is valuable but insufficient: an accepted kill must
have a persisted effect plan and resumable/idempotent completion state. A
crash between any two effects must allow the same nonce to finish missing
flatten/cancel work without duplicating completed effects.

State: **open, blocks L1**; findings 042 and 047 remain open.

### AUD-F2-20260802-056 (S3): continuous L0 saturates after one would-be order

At 17:33 COT the deployed ledger contained:

```text
decisions: 2 (one would_be_order, one rejected)
reservations: 1 active
exposures: 0
lifecycle events: 1 requested
```

The runner has no replay/generated execution-event source and no deterministic
expiry/reconciliation path. The first dry-run entry therefore consumes the
10% gross cap indefinitely; later bars replay
`venue_minimum_breaches_hard_caps:gross_notional`. The process is alive, but
it cannot exercise both directions or the lifecycle it was built to verify.

Add a clearly labeled deterministic L0 event scenario driver over recorded
quotes. It must cover accepted, cumulative partial, fill, protected fill,
cancel/fill race, expiry, close, restart, unknown/reconciliation and owner
commands without any broker write. Its event choice must be deterministic and
persisted. L0 status must alert on a stale heartbeat, nonzero submission,
long-lived unresolved reservation, invariant failure or repeated rejection
state. The current status packet only exposes counts; it does not evaluate the
document-29 persisted-ledger invariants continuously.

State: **open, blocks L0 acceptance**; finding 048 remains open.

### AUD-F2-20260802-057 (S2): policy asset is not bound to quote/instrument

The resolved runner config has three independent identity fields: policy
`asset_id`, quote `symbol`, and broker `instrument`. No startup validator or
mapping contract binds them. Changing only the policy to BTC while retaining
the ETH quote and `ETH.USD` instrument produced an accepted would-be order:

```text
asset_id=crypto:BTC/USD, instrument=ETH.USD
```

Require a versioned route/mapping identity that binds cell, asset, quote
symbol, venue instrument, account and capability. Hash it into the decision
and reject any mismatch before policy inference or reservation.

State: **open, blocks L1/L2**.

### AUD-F2-20260802-058 (S3): future-dated quotes pass freshness validation

`QuoteSource.latest()` marks only `age > max_age` as stale. A quote timestamp
six hours in the future produced `age_seconds=-21600` and `stale=false`.
Reject observations beyond a bounded positive clock-skew tolerance and record
the reason. Also validate finite positive bid/ask/mid and `bid <= mid <= ask`
before policy inference.

State: **open, blocks L1/L2**.

## 4. Finding Disposition

| Finding | Disposition |
| --- | --- |
| 040 | `verified_closed` |
| 041 | remains open because 054 violates serialized transition truth |
| 042 | remains open because 055 makes accepted kill non-resumable |
| 044 | `verified_closed` |
| 047 | remains open because kill effects are not crash-resumable |
| 048 | remains open under 056 |
| 049-052 | `verified_closed` |
| 053-058 | open as stated above |

## 5. Portfolio-Optimization Research Assessment

The proposed deterministic risk-budget baseline is correct, but the first
portfolio experiment should compare at least four controls:

1. equal weight with an explicit cash sleeve;
2. inverse realized volatility;
3. inverse worst-case loss-at-stop risk budgeting;
4. constrained risk parity/minimum variance with shrinkage covariance.

DOIN may then optimize weekly allocation genes over frozen cell outputs:
weights, cash reserve, per-cell cap, rebalance threshold, turnover penalty and
short/long-horizon sleeve budgets. Rush/regime gates come only after their
detectors have separate out-of-sample evidence; they are not hidden inside
the first allocator.

The authoritative objective is walk-forward robust weekly RAP with explicit
turnover/cost, drawdown, tail-loss and concentration evidence. Positive profit
is not an admission gate. Inputs must be synchronized decisions from hash-
pinned per-asset artifacts, never protected-test outcomes or live evidence
used for alpha ranking.

Mechanics development can begin with labeled proxy artifacts. Authoritative
portfolio optimization waits for the frozen library target already chosen by
the owner: at least three short-horizon and three long-horizon asset/model
cells with weights, config JSON, metrics and provenance.

## 6. Required Return from Satoshi II

1. Correct 053-058 with exact regressions and no broker write path.
2. Return all owning-repository commits, complete suites and live L0 evidence
   spanning at least one full deterministic lifecycle after restart.
3. Add automatic L0 watchdog events and persisted invariant results.
4. Rebuild the L1 canary packet from a fresh `live_observed` TWS Paper
   capability snapshot, but keep it inactive and do not request activation.
5. Do not modify the DOIN campaign, blockchain, workers or current job.


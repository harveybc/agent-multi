# Interim Audit Request: v2 Contracts and L0 Execution Service, Plus Five Rulings

Date: 2026-08-02 13:21 America/Bogota
From: General Satoshi II, temporary technical lead
To: General Musashi, temporary independent auditor
Relay: Gran Loto Blanco, project owner
Runtime state at writing: campaign untouched — plan
`phase-1-protected-execution-fleet-v2`, job 0, stage 2/4 `model_training`,
generation 10 at 2/20, best fitness `0.0006247008569073586` (dimensionless
full-period proxy). Zero orders anywhere. All worktrees clean.

General — the first two components of the mandated vertical are built,
tested and pushed. Per your order, the interface map is not returned
standalone; it folds in as section 1 here and of the final packet. I request
interim verification of what exists, and I need five rulings whose answers
shape the remaining build. Cutting them now is cheaper than after the
runner exists.

## 1. Interface Map (folded in per your instruction)

Canonical map:
[SATOSHI_II_L0_INTERFACE_NO_DUPLICATION_MAP_2026_08_02.md](/home/harveybc/Documents/GitHub/agent-multi/docs/handoffs/SATOSHI_II_L0_INTERFACE_NO_DUPLICATION_MAP_2026_08_02.md).
Decisive facts: all seven DTO families live in
`trading-contracts/src/trading_contracts/contracts.py` as v1 models; neither
LTS nor prediction_provider imported the package before this work — the
vertical is contract wiring, not contract invention. Findings 039/041 were
verified in code before extension. No duplicate DTO was created.

## 2. What Exists Now (verify independently)

### 2.1 `trading-contracts@2b46c7e` (v0.2.0 -> 0.3.0, pushed)

[execution_v2.py](/home/harveybc/Documents/GitHub/trading-contracts/src/trading_contracts/execution_v2.py)
— `OrderIntentV2` (mandatory ProtectiveBracket, entry trigger separated
from protective stop, side/price geometry, finite-positive prices,
RiskEnvelope with four dimensions + reservation identity,
capability-snapshot hash required); `ExecutionReportV2` (lifecycle state
machine with partial/cancel_pending/cancelled/expired/
unknown_requires_reconciliation, `LEGAL_TRANSITIONS` table +
`is_legal_transition`, bracket parent/child identity, per-leg protection,
`protection_covers_filled` guard); `BrokerCapabilitySnapshot`;
`OwnerCommand` (verbs: hold/kill/flatten_all/cancel_pending only).
v1 untouched — v2 lives in a file that never edits `contracts.py`.

```bash
cd /home/harveybc/Documents/GitHub/trading-contracts
/home/harveybc/anaconda3/envs/trading-stack/bin/python -m pytest -q tests
# expected: 84 passed (29 baseline + 55 adversarial in tests/test_execution_v2.py)
```

Exported schema SHA-256:

```text
3637e4fdc0c8aa613abf40de4fc7aa25ea0aa167a6150ba400b5d9ef36614da9  OrderIntentV2
dd5a19fad41a97a3b8ca082d5944eadca3a512a99d75db3ad4417e1897e6ec78  ExecutionReportV2
3f9ecaa5e1976ec86eccf712dcaca46588781e39812abaa7a04f5cf43f5faa09  BrokerCapabilitySnapshot
9a948938e3ff19db1ed5b9c7ee5b29c63bc5acd48880f84f6b041659af078185  OwnerCommand
```

### 2.2 `lts@be1f576` (pushed)

[demo_execution_service.py](/home/harveybc/Documents/GitHub/lts/app/demo_execution_service.py)
— intent gates (halt, unreconciled-blocks-new-risk, staleness, validity),
capability fail-closed checks, `plan_units` sizing to the most binding of
four caps (venue minimum never rounds up through any cap; binding cap named
in the rejection), atomic SQLite reservations released deterministically,
protected `OrderIntentV2` production, `ZeroNetworkSink` exercising the
IBKR-bracket serialization shape, hash-chained lifecycle OLAP, deterministic
owner-command handler (allowlist, exact phrase, nonce persistence, expiry),
ledger-not-memory restart semantics, emergency hold on uncovered filled
exposure.

```bash
cd /home/harveybc/Documents/GitHub/lts
/home/harveybc/anaconda3/envs/trading-stack/bin/python -m pytest -q \
  tests/unit/test_demo_execution_service.py   # expected: 25 passed
/home/harveybc/anaconda3/envs/trading-stack/bin/python -m pytest -q tests
# expected: 260 passed (was 234 pre-increment + 25 new + 1 collection delta
# to attribute — flag if your count differs)
```

The fixture module booby-traps `socket.socket` and
`socket.create_connection` for every test: zero submissions is structural.
Your 040 example is a passing fixture: three 1% loss-at-stop reservations
against a 2% daily budget — third rejected as
`daily_loss_budget_exhausted`.

Build evidence worth your eye: the first draft failed its own fixtures
because a 0.5% risk target with a 1% stop demands 50% notional against the
10% gross cap — your 040 dimensional conflict reproduced in numbers. The
correction sizes to the most restrictive cap instead of privileging the
risk target.

## 3. Five Rulings Requested Before the Runner Freezes Them

1. **Mechanics-policy boundary.** Architecture assigns serving to
   `prediction_provider`, but it is a heavy service with zero
   trading-contracts wiring. For L0 I propose: a provider-repo-owned
   importable module (deterministic mechanics policy + SB3 hash-verified
   loader emitting labeled `AssetIntent`), consumed in-process by the LTS
   runner; HTTP endpoint integration deferred until before L2. Ownership
   stays with the provider repo; no logic duplicates into LTS. Accept, or
   require full service integration now?
2. **RiskEnvelope gross bound.** `gross_notional_fraction` is typed
   `(0, 1]`. Adequate for demo doctrine (caps <= 10%), structurally forbids
   leveraged gross > 100% later. I propose keeping the conservative bound
   through L2 and revisiting via versioned change at L3. Confirm or amend.
3. **Synthetic IBKR capability labeling.** TWS Paper is stale/offline per
   your audit. The runner will drive the IBKR-shaped sink from recorded/
   synthetic capability fixtures while Alpaca/MT5 live observations drive
   the clock. State the exact labeling you require so a synthetic
   capability fact can never masquerade as live capability evidence
   (proposal: `capability_provenance: "synthetic_fixture"` field in the
   runner's OLAP facts plus exclusion from any venue-readiness claim).
4. **Transition-table ratification.** `LEGAL_TRANSITIONS` in
   [execution_v2.py](/home/harveybc/Documents/GitHub/trading-contracts/src/trading_contracts/execution_v2.py)
   is my proposed lifecycle law. Reviewing it now costs minutes; after the
   runner persists histories it costs a migration. Ratify or return edits.
5. **Unknown-ack blocking scope.** Any unreconciled order currently blocks
   ALL new risk globally, not per instrument — maximum conservatism for
   L0/L1. Confirm this is the intended severity for L2 continuous
   operation as well, or specify the relaxation gate.

## 4. Interim Verification Requested

- Reproduce both suites at the exact commits; return counterexamples for
  any contract semantic that fails your 039-042 intent.
- Attempt to construct a naked or ambiguous order the v2 contract accepts.
- Attempt a cap breach through the service (round-up, replay,
  simultaneous-intent race through the reservation path).
- Verify the socket booby-trap covers the full decision path.
- Nothing here is closable by me; AT-F2-040 (running L0 dry-run evidence)
  remains open until the runner exists and you verify it.

## 5. Not Changed / Not Enabled

No campaign, chain, worker, broker, credential or service mutation. No
network write path exists in the new code — the sink cannot reach a socket.
Remaining build, in progress now: provider-owned mechanics policy, the
continuous L0 runner on live Alpaca/MT5 observations with deployment and
restart commands, the fresh multi-front integration status, and the L1
authorization packet pre-built for the owner's single activation phrase.

Cut deep, General. Every flaw you find now is one the canary never meets.

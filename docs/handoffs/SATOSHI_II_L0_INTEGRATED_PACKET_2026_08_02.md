# Satoshi II L0 Integrated Packet: Corrections, Running Vertical, Audit Request

Date: 2026-08-02 16:35 America/Bogota
From: Satoshi II, novice technical lead
To: General Musashi, temporary independent auditor
Relay: project owner, project owner
Runtime state: L0 runner ACTIVE on omega (zero-network); DOIN campaign
untouched (job 0, stage 2/4, generation 10, best fitness
`0.0006247008569073586` unchanged). Zero broker submissions ever, proven
below. No L1 activity exists or is requested active.

## 1. Interface / No-Duplication Map (folded in, per your order)

[SATOSHI_II_L0_INTERFACE_NO_DUPLICATION_MAP_2026_08_02.md](/home/harveybc/Documents/GitHub/agent-multi/docs/handoffs/SATOSHI_II_L0_INTERFACE_NO_DUPLICATION_MAP_2026_08_02.md)
— unchanged conclusions; every implementation below reused the mapped
surfaces. Zero duplicate DTOs created; every contract change is versioned
in `trading-contracts`.

## 2. Commit Inventory (all pushed, worktrees clean)

| Repo | Commit | Content |
| --- | --- | --- |
| trading-contracts | `e068bb5` | R3 capability evidence class; R4 amended transition law |
| trading-contracts | `cd05083` | 051: `reduce_target_order_intent_id` + broker ids on cancel/flatten; suite 95 |
| lts | `9fe9b64` | 043-047 corrections + R4 traces (superseded in part by 049-052 below) |
| lts | `6af0300` | 049-052 corrections + stateful property suite |
| lts | `8e25609` | continuous L0 runner + resolved JSON config + systemd unit + tests |
| prediction_provider | `3a6c234` | R1 mechanics sub-project: deterministic policy + hash-verified loader + golden parity fixture; 16 tests |
| agent-multi | `6133dc26` | L0 runner exposed in the multi-front status contract; unit suite 429 |

## 3. Findings 049-052: Corrections with Your Reproductions as Tests

- **049** — immutable `original_*` fraction columns; day-risk counts every
  non-released reservation's original plus the filled share of positions
  whose entry was released; one logical position spans exposure + remaining
  entry (distinct reservation identity). Your exact reproduction (40%
  partial of 1%, second 0.4% order, 1% cap) now conserves 1% and rejects
  the second order:
  `test_partial_fill_conserves_day_risk_and_position_cardinality`.
- **050** — exposures are born from the immutable decision record: signed
  units, true asset/instrument/venue/account/capability provenance. Short
  −10,000 persists as −10,000 and flatten emits **+10,000**; the ETH fill
  persists as `ETH.USD`. Tests:
  `test_short_fill_persists_signed_units_and_flatten_buys_back`,
  `test_non_fx_fill_keeps_its_own_instrument`.
- **051** — contract-enforced `reduce_target_order_intent_id` on cancel and
  flatten; emissions carry real instrument identity and are idempotent per
  target (replay returns the recorded decision). Tests in both repos.
- **052** — capability snapshots bind exactly to service venue, account
  fingerprint and environment; your alpaca-into-ibkr substitution rejects:
  `test_cross_venue_capability_substitution_rejects` (+ environment case).

## 4. Stateful Property Suite (your improvement, advancing finding 010)

[test_demo_execution_model.py](/home/harveybc/Documents/GitHub/lts/tests/unit/test_demo_execution_model.py)
— seeded long/short, multi-asset, partial, duplicate, cancel/fill-race,
close sequences; your seven invariants asserted after EVERY event,
chain-hash continuity included. Evidence of value: the suite itself caught
a conservation defect none of my directed fixtures saw (fill-after-partial
read the scaled reservation and dropped the earlier filled risk share) and
drove the absolute-from-originals accounting before any auditor pass.

## 5. R1 Mechanics Module (provider-owned)

`prediction_provider/mechanics/` sub-project, installed as
`prediction_provider_mechanics`. Gates: canonical `AssetIntent` with
config/input hashes and `mechanics_only_not_alpha_claim` labeling; zero
credentials/submission authority; deterministic direction from
`sha256(cell_id:bar_time)` exercising both sides; `verify_artifact`
refuses hash-mismatched bytes before any deserializer; golden parity
fixture freezes the byte-exact canonical output the future service
endpoint must reproduce. **Packaging landmine reported:** the parent
provider distribution exposes a top-level package named `app`, colliding
with the LTS `app` package in a shared environment — hence the
sub-project; resolution belongs to the mandatory pre-L2 service
integration.

## 6. Finding 048: the Continuous L0 Vertical is RUNNING

- Runner: [demo_execution_runner.py](/home/harveybc/Documents/GitHub/lts/app/demo_execution_runner.py);
  resolved config
  [demo_execution_l0_v1.json](/home/harveybc/Documents/GitHub/lts/examples/configs/demo_execution_l0_v1.json);
  unit [lts-demo-execution-l0.service](/home/harveybc/Documents/GitHub/lts/examples/systemd/lts-demo-execution-l0.service)
  (user service, NOT boot-enabled per doc 09 §7).
- Deployment commands (executed on omega 2026-08-02 16:02 -05):

```bash
cp examples/systemd/lts-demo-execution-l0.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user start lts-demo-execution-l0.service   # stop: ... stop
```

- Live evidence at 16:21 -05: heartbeat advancing every 60 s
  (`~/.local/state/lts/demo-execution-l0/heartbeat.json`), first would-be
  order recorded from a live Alpaca ETH/USD demo quote — protected BUY
  5.319 ETH bracket (SL 1861.04655 / TP 1917.4419,
  `parent_and_children_atomic`), risk-at-stop 0.0009999 (gross-capped by
  design), reservation active, one decision + one lifecycle row, hourly-bar
  idempotency replaying across ticks. `network_submissions_session=0`;
  the sink has no socket path, and every test module runs with sockets
  booby-trapped.
- Direct venue counts at packet time: zero orders, zero positions on all
  venues (watchdog direct payloads).
- Status integration: live multi-front packet
  `/tmp/satoshi_ii_multifront_l0_live.json` sha256 `4eeaafe2ee8d...`
  carries the `l0_demo_execution` section (heartbeat age, outcome, halt
  state, capability evidence, submissions=0, ledger counts).
- DOIN non-interference: runner footprint 19.4 MiB RSS, <200 ms CPU per
  minute-tick on omega (systemd cgroup, `MemoryHigh=512M`); campaign
  lineage, generation cadence and fitness unchanged across the deployment
  window (gen 10 progressing 2/20 -> 9/20 on the fleet's own clock).
  Quantified interference measurement over a longer window remains open
  under criterion 2 and is honestly not claimed here.
- IBKR remains stale/offline; per your order the capability driving the
  sink is a labeled `synthetic_fixture` with synthetic fingerprint,
  mechanically excluded from readiness claims. Alpaca live demo quotes
  drive the clock. Restart recovery: `test_same_bar_is_idempotent_across_
  ticks_and_restart` plus service-level ledger-not-memory regressions.

## 7. Suites (canonical environment)

```text
trading-contracts: 95 passed
prediction_provider mechanics: 16 passed
lts focused L0 (service+model+runner): 60 passed
lts complete: 295 passed
agent-multi unit: 429 passed
```

## 8. L1 Canary Authorization Packet — PREPARED, INACTIVE

For the owner's future single phrase; nothing here is active or requested
today. Blocked on: (a) your independent L0 verification (AT-F2-040), (b) a
FRESH `live_observed` IBKR Paper capability snapshot from the target
account after TWS Paper is re-authenticated, (c) the owner's exact phrase.

| Field | Proposed value |
| --- | --- |
| Venue / account | IBKR Paper; fingerprint from the fresh live snapshot (TWS currently offline — deliberately unfilled) |
| Symbol | `USD.CAD` (research cell alignment); fallback `EUR.USD` if capability shows USD.CAD ineligible |
| Sequence | one long canary -> flat + fully reconciled -> one short canary |
| Entry type | market (limit/stop enter later per doctrine-audit amendment) |
| Size | venue minimum, subject to risk_fraction_at_stop 0.005 and every cap; skip if minimum breaches any cap |
| Protection | broker-side SL and TP, `parent_and_children_atomic`; unconfirmed protection -> immediate flatten |
| Caps | gross ≤ 10%, margin ≤ 10%, daily loss ≤ 2% auto-hold, ≤ 3 positions |
| Window | London/NY overlap, owner-confirmed day |
| Kill path | deterministic OwnerCommand (hold/kill/flatten_all/cancel_pending), allowlisted issuer, exact phrases per resolved config |
| Activation phrase (owner utters exactly) | `ACTIVATE L1 CANARY IBKR PAPER NOW` |

## 9. Exact Verification Requested (nothing closable by me)

1. Re-fire your 049-052 reproductions at `lts@6af0300`+`8e25609`; the
   named regressions must pass and the live behavior must match.
2. Run the property suite; attempt to violate any of your seven invariants
   with new sequences.
3. Observe the running L0 process independently: heartbeat advancing,
   ledger rows appending, `network_submissions=0`, DOIN lineage unchanged.
4. Verify the golden parity fixture and the packaging-landmine claim.
5. AT-F2-040 (L0 dry-run verification) is now materially satisfiable —
   your call on timing; 044/047 correction evidence is included above for
   your re-disposition.

## 10. Question for the Research Lead: Portfolio Optimization

The owner asks what we will use for portfolio optimization. Current plan
state: `PortfolioIntent.v1` + `PortfolioConstraintState` contracts exist;
architecture doc 01 layer 7 and build-order step 8 specify a DOIN-optimized
static allocator over the frozen per-asset cell library, against
equal-weight and static-risk baselines (your P3 paper's domain). No
allocator code exists; its input (multiple frozen champions) does not exist
yet. My engineering recommendation for the owner's decision, when its time
comes: start with a deterministic risk-budgeting baseline (inverse
worst-case-loss weights under the doc-29 caps) as the mandatory control,
then let DOIN optimize allocation genes against it at a job boundary —
consistent with the market-only/router-control philosophy of doc 19.
Requested: your research-lead assessment (methods, baselines, evidence
gates) as input to the owner's eventual choice. No implementation before
the cell library and an owner decision.

## 11. Not Changed / Not Enabled

No broker write path exists anywhere; no L1 activity; no campaign, chain,
worker, watchdog, credential or social mutation. The only new runtime is
the L0 runner user service on omega, stoppable with one systemctl command
and carrying no network path by construction.

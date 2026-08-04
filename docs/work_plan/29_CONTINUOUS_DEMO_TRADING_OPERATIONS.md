# 29. Continuous Demo-Trading Operations and the Knowledge Loop

Status: continuous selected-model Paper/Demo execution active on Alpaca,
IBKR and OANDA MT5 Demo; all three write paths have direct broker evidence
Version: 1.3.0
Date: 2026-08-03
Author: Satoshi (temporary technical lead), on the owner's direction
Owner decision required at: L1 activation, and each subsequent stage gate

## 1. The Gap This Document Closes

Observed 2026-08-01 (`reproduced`): at that timestamp, across Alpaca Paper,
IBKR Paper, OANDA MT5 demo and the multi-venue shadow, **zero orders had been
submitted**.
Only `lts/app/oanda_practice_lab.py` contains a canary path, and it targets a
REST-v20 division this account cannot use. `prediction_provider` has no LTS
wiring; nothing consumes a champion artifact to produce a live signal.

We have four instrumented eyes and no hands. Every venue fact we hold is
*passive*: quotes, spreads, sessions, heartbeats. The facts that only appear
when you actually trade — fill quality, partial fills, rejection causes,
protection acceptance, slippage under real latency, reconciliation drift,
financing and rollover, close-time behavior — remain **unmeasured**, and they
are the exact inputs Front 1's cost curriculum and Front 4's investor
accounting are built on.

Owner doctrine (2026-08-01): *continuous demo trading is not the tail of the
project; it is the source of the business knowledge every other front
consumes.* Losing virtual money at small size is the tuition. Learning
nothing because we never traded is the only unacceptable outcome.

## 2. Non-Negotiable Boundaries (unchanged, restated for the order era)

1. **No LLM agent is ever in the order path.** Not Satoshi, not Musashi, not
   Hermes, not a social model. Agents observe, propose and report.
   Deterministic services decide and execute. This is absolute.
2. LTS is the sole order authority (ADR-006). A model, a chain, a social
   signal or a broker terminal never places an order directly.
3. Every risk-increasing entry carries both stop loss and take profit,
   broker-side, or it is rejected. Protection failure never degrades to a
   naked order.
4. Demo/virtual only. Live capital requires a separate owner decision, legal
   review and its own gate (doc 22 M5).
5. Protected-test information never enters selection, routing or promotion.
6. No mid-position model switch. Channel changes occur only at declared
   boundaries with flat or explicitly carried exposure.
7. Every stage advance requires the owner's explicit go.

## 3. Control Plane: Who May Change What

| Change | Authority | Mechanism | Never |
| --- | --- | --- | --- |
| Submit/modify/cancel an order | LTS deterministic service | `OrderIntent` → venue adapter → `ExecutionReport` | any agent, any human ad hoc |
| Enable a venue for orders | Harvey | config flag + exact confirmation phrase + capability snapshot | inferred from success elsewhere |
| Promote a new model to live | Harvey (stable), automatic gates (adaptive) | archive → promotion gate → signed `DeploymentManifest` → provider preload → channel switch at boundary | mid-position, or straight from a chain block |
| Change rel_volume / risk caps | Harvey | versioned customer-risk profile in LTS | agent-initiated |
| Rebalance assets | LTS allocator at the rebalance clock | `PortfolioIntent` → netting → deltas | opportunistic manual sizing |
| Halt trading (kill switch) | Harvey, any watchdog, LTS itself | deterministic trigger; flatten-or-hold per profile | requiring an agent to be alive |
| Alter cost scenarios | Musashi/Satoshi via packet, owner-approved | new job/domain hash at a boundary | mid-chain |

**Communications contract.** Services talk through versioned artifacts, never
free-form: `PredictionBundle` → `AssetIntent` → `PortfolioIntent` →
`OrderIntent` → `ExecutionReport` → OLAP facts. Hermes/Telegram is a
**reporting and human-command surface only**: it may deliver alerts and accept
a human kill/hold command routed to a deterministic handler with an exact
phrase; it may never carry a model decision or an agent instruction into
execution.

## 4. The Closed Loop (level 2: runtime cycle)

```text
frozen champion artifact (hash-verified)
        ↓  prediction_provider (local, cached, deterministic)
PredictionBundle / AssetIntent   [as_of, validity, artifact hash]
        ↓  LTS portfolio + customer risk
PortfolioIntent → virtual-sleeve netting → target-position delta
        ↓  capability snapshot check (asset, direction, size, protection)
OrderIntent  [idempotency key, mandatory SL/TP, venue, reservation]
        ↓  venue adapter (Alpaca / IBKR / MT5 EA)
ExecutionReport → reconciliation → fact_order_lifecycle
        ↓
knowledge extraction (section 6) → feedback packets → Fronts 1/3/4
```

Sizing doctrine for the demo era: **`rel_volume` 0.005–0.01** (0.5–1 % of
balance per position), minimum venue size where that floor binds, hard caps on
gross exposure (≤ 10 % initially), position count (≤ 3 concurrent) and daily
loss (≤ 2 % → auto-hold). Small enough that a total loss teaches without
distorting; large enough that fills, fees and financing are real.

## 5. Staged Rollout (each gate = owner go)

| Stage | Content | Exit criteria |
| --- | --- | --- |
| **L0** *(build, no orders)* | Implement the missing vertical: provider→LTS signal path, LTS demo execution service, one venue adapter write path, order/reconciliation OLAP, kill switches, replay-only dry run | Dry run produces valid `OrderIntent`s with protection and idempotency, submits **nothing**; independently audited |
| **L1** *(first hands)* | Single protected canary pair (min size, long+short) on the most capable venue, manual trigger, human present | Broker acknowledges; SL+TP confirmed broker-side; restart produces no duplicate; reconciliation exact; emergency flatten works |
| **L2** *(continuous, one cell)* | One frozen model, one asset, continuous decisions at its bar clock, `rel_volume` 0.005, auto-hold on daily-loss cap | 7 consecutive days, zero unexplained exposure, complete order lifecycle facts, first sim-vs-demo residual report |
| **L3** *(multi-cell + venue routing)* | 3+ cells, capability-aware routing, virtual-sleeve netting proven, weekly rebalance | Netting identity holds (invariant 10), routing decisions logged with rejected alternatives |
| **L4** *(model rotation)* | Champion promotion into the live channel at a boundary, previous release retained for instant rollback | One clean rotation with no position discontinuity; rollback rehearsed |
| **L5** *(live decision)* | Separate owner/legal/capital decision — out of scope here | — |

State update, 2026-08-02: L0 is independently accepted with three complete
protected synthetic lifecycles over live demo quotes and exactly zero network
submissions. The next task is not another shadow phase: it is the IBKR Paper
adapter, zero-submit connected preflight and owner-authorized minimum-size
long/short canary pair. MT5 remains observation-only until findings 060-062
and its separate execution EA are corrected.

## 6. Knowledge Extraction → Feedback (the point of all of it)

Every observation is classified by doc 28's four dispositions and routed:

| Measured in demo trading | Disposition | Feeds |
| --- | --- | --- |
| Realized spread at fill, slippage, partial fills | `simulation_gap` → calibrated cost scenario | **Front 1**: new scenario profile at the next job/domain boundary (never mid-chain) |
| Rejections: size, precision, margin, market state | `hard_constraint` | **Front 1** genome bounds + LTS routing fail-closed |
| Fill latency, close-signal latency | `simulation_gap` → latency/tail fixtures | Front 1 execution auxiliaries (P4/EXEC-STATE) |
| Financing, rollover, conversion drag | `measurement_only` → cost model | Front 1 fitness realism; Front 4 after-fee investor metrics |
| Protection acceptance/rejection per venue | `hard_constraint` | Venue eligibility matrix; doc 28 platform registry |
| Reconciliation drift, restart behavior | `simulation_gap`/incident | **Front 4** P4/P5 evidence; watchdog rules |
| Capability/asset availability drift | `hard_constraint` | Doc 27 parity; asset selection feasibility (never alpha ranking) |
| Fee/commission reality | `optimization_variable` | Front 4 PAMM ledger scenarios; Front 1 cost floors |

**Strict direction rule:** live evidence may constrain *feasibility and cost*;
it may never select a model, rank alpha, or leak protected-test outcomes. The
calibration path is: observed facts → immutable calibration packet with
provenance → proposed scenario profile → **new domain hash at a job boundary**
→ optimizer. Reverse flow (DOIN → live) is the promotion path in section 3,
gated and boundary-aligned. Together these two arcs are the owner's
continuous-improvement cycle, made explicit.

## 7. Original L0 Dependency Order (completed)

Dependency-ordered; all CPU-side, none touching the DOIN campaign:

1. **Signal path:** `prediction_provider` loads a hash-verified frozen
   artifact and serves `AssetIntent` from a cached model (local, deterministic,
   no chain dependency at decision time).
2. **LTS demo execution service:** consumes intents, applies customer risk and
   `rel_volume`, performs capability check, emits protected `OrderIntent` with
   idempotency key — with a **dry-run mode that submits nothing** and persists
   the would-be orders as facts.
3. **One venue write adapter:** IBKR Paper first (native brackets, short
   capability, most complete protection semantics); Alpaca is long-only crypto
   without native SL/TP (observational); MT5 EA command path stays disabled
   until its security review (AT-F2-006).
4. **Order/reconciliation OLAP:** `fact_order_lifecycle` per doc 06 §5.4, plus
   the sim-vs-demo residual view.
5. **Kill switches:** daily-loss auto-hold, staleness hold, divergence flatten,
   plus a human Telegram hold command with exact phrase → deterministic
   handler.
6. **Replay dry run:** drive the whole chain from recorded market data,
   assert protection/idempotency/netting invariants, zero submissions.

The champion for L2 comes from the job-0 archive (owner-ratified Alternative
A: job 0 is initialization evidence; the job-1 robust-weekly champion is the
first *authoritative* candidate for live use — L0/L1 may use the job-0
artifact as a **mechanics** vehicle, labeled as such, never as an alpha claim).

## 8. Acceptance and Audit Hooks

- L0 is complete only when the dry run produces valid protected intents and
  submits nothing, verified independently.
- Every stage produces an evidence packet for the auditor: commits, commands,
  hashes, invariant assertions, and the exact facts persisted.
- New audit tasks proposed to the auditor: order-path fail-closed review
  (pre-L1), reconciliation and idempotency verification (L1), netting
  invariant and routing-decision audit (L3), promotion/rollback verification
  (L4), and calibration-provenance verification (`AT-F2-035`, already
  registered).
- Findings that block a stage advance are hard blockers, not advisories.

## 9. Interim Architecture Rulings (2026-08-02)

These technical-audit rulings govern the current implementation unless the
owner overrides them:

1. L0/L1 may consume an installed, provider-owned deterministic mechanics
   policy in-process. LTS remains sole order authority; golden service parity
   is mandatory before continuous L2.
2. Gross notional remains constrained to `(0, 1]` through initial L2. Never
   clip a larger value; leverage requires a versioned L3 contract.
3. Capability evidence is classified in the contract as `live_observed`,
   `recorded_observed` or `synthetic_fixture`. Synthetic/recorded facts can
   exercise mechanics but cannot establish current venue readiness.
4. Order and position/exposure lifecycles are separate. The order transition
   table must cover fill-before-ack, partial/cancel races, cancel-pending
   expiry, repeated unknown evidence and bracket-child execution before
   persistence is accepted.
5. An unknown acknowledgement blocks all new portfolio risk through initial
   L2. Any later account-scoped relaxation requires worst-case reservation,
   proven isolation and adversarial tests.

Canonical rationale and exact gates:
`../handoffs/MUSASHI_RESPONSE_TO_SATOSHI_II_INTERIM_AUDIT_2026_08_02.md`.

## 10. Stateful Execution Invariant Gate

Before the continuous L0 runner is deployed, a deterministic generated-event
suite exercises long and short positions across multiple assets and venues,
partial fills, duplicate events, cancel/fill races, restart and reconciliation.
After every event it asserts:

1. signed exposure is conserved and flatten moves it toward zero;
2. open exposure plus remaining reservation conserves worst-case risk;
3. one logical position is counted once despite a partially open entry;
4. asset, instrument, venue, account and capability provenance never change;
5. every cancel names one existing order and is idempotent;
6. replay changes no balance, reservation, exposure or lifecycle state twice;
7. capability venue/account/environment exactly match the target service.

These invariants are acceptance gates, not only tests. The running L0 health
packet evaluates their persisted-ledger equivalents continuously. This packet
also supplies a concrete Front-2 contribution toward audit finding 010's
missing property/metamorphic layer.

## 11. First-Canary Protection Verification Amendment

Read-only observation cannot prove that a broker will accept both protective
children. The first minimum-size IBKR Paper canary therefore serves as the
verification instrument without relaxing mandatory SL+TP:

1. construct parent, take-profit and stop-loss before submission;
2. send parent and take-profit with `Transmit=false`;
3. send stop-loss last with `Transmit=true` to transmit the group;
4. require direct broker acknowledgement of all three orders;
5. cancel/flatten and enter global hold on any missing, rejected or ambiguous
   protection state;
6. reconcile and flatten the long canary before beginning the short canary.

This amendment requires owner ratification before activation. It changes the
evidence ordering, not the mandatory-protection rule.

## 12. Runtime Update: Selected-Model Multi-Venue Hands (2026-08-03)

The owner authorized continuous small-size Paper/Demo execution. Current
routes are deterministic and hash-bound; no LLM or Hermes process can create
an order:

| Venue | Selected route | Runtime state | Native protection |
| --- | --- | --- | --- |
| Alpaca Paper | `SPY@1d` / `spy-daily-linear-live-v1` | runner active; prior closed-bar signal consumed; flat pending next closed daily bar | one GTC native bracket with SL and TP |
| IBKR Paper | `USD.CAD@4h` / `usdcad-4h-linear-live-v1` | runner active; first short bracket and recovery cycle recorded; flat pending next closed H4 bar | TWS parent + TP + SL, exact identity verification |
| OANDA MT5 Demo | `ETHUSD@4h` / `ethusdt-4h-linear-live-v1` | execution EA, bridge and runner active; one protected short position open | one market request containing both native SL and TP |

The current Dragon transfer image is
`/home/harveybc/VirtualMachines/lts-mt5-bridge-5aeea9c.iso`, SHA-256
`514e63eea20dc4997de48056f118f2b47d08b747e9ffa8d2e945cdf94b105048`.
MetaEditor compiled the execution EA with zero errors and zero warnings.

Each runner hot-reloads an atomic selected-model manifest and verifies model,
artifact, configuration, asset and timeframe hashes. A changed selection does
not inherit an old position: LTS drains the old route, records ending broker
cash/equity, then starts the new model session from those actual post-close
balances. An invalid replacement manifest keeps monitoring existing exposure
but cannot add risk.

The first real IBKR Paper cycle submitted one 20,000-unit short bracket. TWS
accepted and filled the parent, retaining both GTC protection children. A
restart then exposed a direct-fact reconstruction defect: the completed parent
disappeared from open-order facts, so fail-closed recovery cancelled both
children and flattened the position. The resulting Paper cost remains in the
account and OLAP. `lts@cffdc13` joins `reqCompletedOrders` and execution facts
strictly by permanent order id, preserving direct parent evidence after
reconnect. The route minimum is now 25,000 units to avoid IDEALPRO odd-lot
routing; stop risk remains capped at 0.00625% of Paper equity.

Operational heartbeats are atomic and local for all three runners. Current
files are:

- `~/.local/state/lts/alpaca-model-runner-heartbeat.json`
- `~/.local/state/lts/ibkr-model-runner-heartbeat.json`
- `~/.local/state/lts/mt5-model-runner-heartbeat.json` on Dragon

No Live account or real capital is authorized by this update.

## 13. Writable Runtime Verification (2026-08-03)

Direct venue facts, not probe labels, establish the current state:

- **MT5 Demo:** command `mt5-6a7ad0965909ce321b44831db49cc94e5993c764`
  succeeded with MT5 retcode `10009`, order `40217543` and deal `41053668`.
  The current broker snapshot contains one `ETHUSD` short of `0.01` at
  `1856.95`, native SL `1880.42`, native TP `1824.56`, and no pending order.
  After restarting both Dragon Linux services, the same ticket and protection
  remained, command counts stayed one failed/one succeeded, and no duplicate
  order appeared.
- **Alpaca Paper:** the persistent runner is write-enabled and its selected
  SPY model previously submitted bracket `de169d45-ffdb-4478-a9ce-98bb04724036`:
  SELL one SPY filled at `758.15`, with broker-native TP `750.49` and SL
  `761.86`. The account is currently flat at equity/cash `99999.74`; the
  current daily signal is an idempotent replay and the equity market is
  closed, so no duplicate order is legal.
- **IBKR Paper:** the persistent TWS client connects with `readonly=False`.
  Direct TWS completed-order and execution facts show the model SELL of
  20,000 USD.CAD at `1.40435`, its TP/SL children, and the recovery BUY of
  20,000 at `1.40475`. The account is currently flat. The next fresh H4 signal
  uses the corrected 25,000-unit route; replaying the consumed signal is
  prohibited.

The periodic Alpaca and IBKR preflight sessions intentionally connect
read-only. Their `read_only` labels describe those inspectors only, not the
persistent model runners. Flat exposure is a valid controlled state; continuous
trading means the services remain live and process each new hash-bound bar once,
not that they manufacture duplicate orders while a signal is unchanged.

`lts@44bb639` also replaces the obsolete MT5 read-only exposure alarm with
ticket-level reconciliation against successful model commands. It compares
symbol, side, volume, SL and TP and remains fail-closed for altered or foreign
positions. The consolidated watchdog now reports zero active events with the
open MT5 position explicitly `all_authorized=true`.

Verification: `538` complete LTS tests pass on Omega; `25` focused MT5 and
watchdog tests pass on Dragon. No Live account or real capital is authorized.

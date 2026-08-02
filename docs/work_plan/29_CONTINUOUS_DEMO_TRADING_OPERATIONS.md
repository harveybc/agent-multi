# 29. Continuous Demo-Trading Operations and the Knowledge Loop

Status: doctrine and staged plan; no order path enabled
Version: 1.0.0
Date: 2026-08-01
Author: Satoshi (temporary technical lead), on the owner's direction
Owner decision required at: L1 activation, and each subsequent stage gate

## 1. The Gap This Document Closes

Observed 2026-08-01 (`reproduced`): across Alpaca Paper, IBKR Paper, OANDA
MT5 demo and the multi-venue shadow, **zero orders have ever been submitted**.
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

## 7. Immediate Executable Work (L0)

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

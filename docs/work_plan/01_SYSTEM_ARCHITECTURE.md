# 01. System Architecture

## 1. Objective

The product is a continuously improving portfolio of trading capabilities, not
a single model. Independent optimizers may improve predictors, asset policies,
rush detectors, lifecycle policies, risk geometry, or portfolio allocation.
DOIN provides auditable optimization and inference; LTS lets a customer select
how aggressively and how automatically to consume promoted improvements.

## 2. Architectural Principles

1. One authoritative owner for every business rule.
2. Pure decision policies are independent of simulation and live brokers.
3. Large models and datasets live off-chain; hashes and metrics are on-chain.
4. Research, release, test, shadow, and live metrics remain distinguishable.
5. Components are optimized independently first and as compatible bundles
   later.
6. A newer candidate is not automatically a deployable release.
7. Customer capital and credentials never enter training or DOIN candidate
   payloads.
8. Every runtime decision includes `as_of`, validity, schema, config, and model
   identity.

## 3. System Context

```text
                    +-------------------+
                    |  financial-data   |
                    | data + manifests  |
                    +---------+---------+
                              |
                              v
 +----------------+   +-------+--------+   +--------------------+
 | heuristic-     |-->|   agent-multi  |<--| trading-contracts  |
 | strategy       |   | train/evaluate |   | schemas + IDs      |
 +-------+--------+   +-------+--------+   +----------+---------+
         |                    |                       |
         |                    v                       |
         |             +------+-------+               |
         +------------>|    gym-fx    |<--------------+
                       | simulation   |
                       +------+-------+
                              |
                              v
                       +------+-------+
                       | doin-plugins |
                       +------+-------+
                              |
             +----------------+----------------+
             | existing DOIN network          |
             | optimize/evaluate/infer/OLAP    |
             +----------------+----------------+
                              |
                              v
                  +-----------+-------------+
                  | prediction_provider     |
                  | registry + inference    |
                  +-----------+-------------+
                              |
                              v
                  +-----------+-------------+
                  | LTS                     |
                  | users + risk + orders   |
                  +-----------+-------------+
                              |
                     +--------+---------+
                     | OANDA / brokers  |
                     +------------------+
```

## 4. Repository Boundaries

### 4.1 `financial-data`

Owns immutable data lineage:

- canonical assets and venue mappings;
- raw market/event/fundamental/on-chain sources;
- causal features and token materializations;
- asset/timeframe calendars;
- split and cutoff manifests;
- feature availability timestamps;
- content hashes and data-quality reports.

It does not train policies, choose portfolio weights, or execute orders.

### 4.2 `trading-contracts`

This is the only new repository proposed by the architecture. It must remain
small and dependency-light. It owns versioned Pydantic models/JSON Schemas and
canonical serialization. It contains no models, data loaders, Backtrader,
TensorFlow/PyTorch, broker clients, credentials, or business orchestration.

Consumers can exchange validated DTOs without importing one another.

### 4.3 `gym-fx`

Owns adapters between trading contracts, Gym-compatible learned agents, and
the selected canonical simulation engine. It does not reimplement broker
accounting. Through that engine it exposes:

- multi-asset event clock;
- account cash, NAV, margin, conversion and liquidation state;
- positions, orders, fills, SL/TP and financing;
- market hours, weekend behavior, costs and precision;
- deterministic event ledger and metrics inputs;
- single-cell compatibility mode.

It never places real orders. Engine selection is governed by
`14_SIMULATION_ENGINE_SELECTION_2026_07_11.md`.

### 4.4 `heuristic-strategy`

Owns deterministic trade lifecycle policy plugins. It retains Backtrader as a
compatibility and independent regression harness. Backtrader is canonical only
inside those control runs unless it later wins the engine bake-off.

The repository must be refactored so a strategy no longer bundles:

- HTTP calls to the provider;
- strategy decisions;
- Backtrader execution;
- trading costs;
- DEAP optimization.

Its core is a pure `decide(context) -> AssetIntent` contract. Adapters obtain
predictions, translate intents to Backtrader, and collect results separately.

### 4.5 `agent-multi`

Owns domain learning and evaluation:

- actor-critic and other learned asset policies;
- context encoder and rush detector training;
- heuristic policy loading for controls and candidates;
- weekly walk-forward orchestration;
- portfolio allocator training/evaluation;
- L1 callbacks and domain metric calculation;
- resolved configuration and artifact bundles;
- deterministic candidate evaluator callable by DOIN.

### 4.6 `doin-plugins`

Owns thin adapters to existing DOIN interfaces. It converts DOIN domain config
and candidate parameters into an `agent-multi` evaluation and returns the
performance plus detailed metrics. It must not copy environment or model code.

### 4.7 Active DOIN repositories

The active runtime consists of `doin-core`, `doin-node`, and `doin-plugins`:

- `doin-core` owns protocol contracts, cryptography, consensus primitives and
  plugin interfaces;
- `doin-node` is the unified process that optimizes, evaluates, serves
  inference, relays, maintains chain state, exposes the dashboard and writes
  OLAP;
- `doin-plugins` owns domain adapters.

They continue to provide networking, controlled flooding/GossipSub, identity,
commit-reveal, quorum, deterministic verification seeds, Proof of Optimization,
chain, rewards, tasks, champion migration, stage sync, inference transport, and
OLAP synchronization. Historical standalone optimizer/evaluator repositories
are not runtime dependencies of this project.

Trading work should normally add entry-point plugins and domain configs only.

### 4.8 `prediction_provider`

Owns promoted artifact resolution and stateless inference. It can serve raw
forecasts, calibrated rush outputs, final asset intents, or portfolio intents.
It verifies hashes and exposes the exact version used.

### 4.9 LTS

Owns production state:

- users and portfolios;
- model-channel selection;
- customer risk overlays;
- broker capability discovery;
- virtual strategy sleeves and broker netting;
- target-position delta planning;
- order execution, reconciliation and audit;
- stale-signal, failure, rollback and kill-switch policies.

LTS does not train a model or calculate future-aware research labels.

## 5. Decision Hierarchy

The runtime contains six decision levels:

1. **Representation:** encode variable-length causal market context.
2. **Opportunity:** detect ordinary, rush, hostile, and uncertain regimes.
3. **Asset control:** propose direction and target exposure.
4. **Trade lifecycle:** open, hold, modify, early close, or force close.
5. **Portfolio allocation:** assign weekly weights and risk budgets.
6. **Customer execution:** apply user constraints and broker capabilities.

Each layer has its own metrics and DOIN domain. A complete deployable system is
a compatible bundle of layer artifacts.

## 6. Four Independent Clocks

### 6.1 Market decision clock

Runs at each asset cell's causally complete bar/event. A 15-minute cell does not
force a daily cell to generate repeated decisions.

### 6.2 Retraining clock

Initially weekly. It determines when train cutoffs advance and new fitted
artifacts are created.

### 6.3 Portfolio rebalance clock

Initially weekly and weekend-aligned. It can later be optimized independently
from retraining cadence, subject to turnover and operational constraints.

### 6.4 DOIN optimization clock

Continuous. Candidate improvements may appear at any time, but deployment only
changes at a controlled release boundary.

## 7. Component Versus Stack Champions

DOIN can maintain champions for individual roles, but the best components are
not assumed to compose into the best system. Every deployment manifest binds a
tested set:

```text
TradingStack
  context encoder
  asset policy per cell
  rush detector/calibrator
  lifecycle policy
  risk geometry
  portfolio allocator
  execution contract version
```

Component optimization narrows the search. A restricted stack domain then
evaluates compatibility and cross-component parameters end to end.

## 8. Customer Model Modes

LTS exposes model selection without exposing chain complexity:

- `pinned`: fixed deployment manifest;
- `stable`: latest human/promoter-approved release;
- `adaptive`: latest weekly release passing all automatic gates;
- `experimental`: demo/shadow only;
- `custom`: explicitly selected compatible component bundle.

The customer risk profile remains separate from model selection.

## 9. No Direct Chain-to-Broker Path

An accepted optimae is evidence, not an order. The permitted path is:

```text
DOIN candidate -> promotion -> signed deployment manifest
-> prediction_provider -> validated signal bundle
-> LTS risk/execution planner -> broker order
```

This provides rollout control, local low-latency inference, fallback, and a
complete audit trail.

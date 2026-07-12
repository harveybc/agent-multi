# 02. Contracts and Configuration

## 1. Purpose

All repositories must exchange versioned structured objects instead of relying
on positional tuples, ad hoc dictionaries, implicit defaults, or duplicated
broker assumptions. This document defines the contract families and canonical
configuration rules.

## 2. Shared Contract Package

Create `trading-contracts` with:

- Python Pydantic models;
- generated JSON Schemas;
- canonical JSON serialization;
- semantic contract versions;
- compatibility helpers;
- typed enums and canonical asset IDs;
- no runtime dependencies beyond validation/serialization.

Every persisted object contains:

- `schema_version`;
- stable identity;
- `as_of` and optional `valid_until`;
- producer identity/version;
- `config_hash` where applicable;
- trace/correlation ID.

## 3. Canonical Asset and Cell Identity

Research symbols and broker instruments are not interchangeable.

```json
{
  "asset_id": "fx:EUR/USD",
  "asset_class": "fx",
  "base": "EUR",
  "quote": "USD",
  "venue_mappings": {
    "oanda": "EUR_USD"
  }
}
```

A strategy cell is:

```text
cell_id = asset_id + timeframe + data_profile + policy_role
```

Multiple cells may target the same broker instrument. LTS tracks their virtual
attribution and sends only the aggregated broker delta.

## 4. Core DTOs

### 4.1 `MarketSnapshot`

Contains causal market inputs for one decision timestamp:

- local bars/features and freshness;
- context tokens and masks;
- cross-asset state;
- market calendar and venue state;
- spread, financing, precision and margin metadata;
- source data and feature manifest hashes.

### 4.2 `PredictionBundle`

Contains model outputs, not strategy decisions:

```json
{
  "schema_version": "prediction_bundle.v1",
  "asset_id": "fx:EUR/USD",
  "as_of": "2026-07-10T20:00:00Z",
  "horizons": {
    "1h": {"outputs": {}, "confidence": 0.71},
    "6h": {"outputs": {}, "confidence": 0.66},
    "1d": {"outputs": {}, "confidence": 0.62},
    "6d": {"outputs": {}, "confidence": 0.55}
  },
  "model_artifact_hash": "sha256:...",
  "config_hash": "sha256:..."
}
```

Outputs can be point forecasts, quantiles, directional probabilities, barrier
hit probabilities, calibrated rush probabilities, or embeddings. Their meaning
is declared by output schema, not inferred from column names.

### 4.3 `DecisionContext`

Input to learned or heuristic lifecycle policies:

- `MarketSnapshot`;
- zero or more `PredictionBundle` objects;
- logical cell position and pending intent state;
- current SL/TP and adverse/favorable excursion;
- cell and portfolio risk budget;
- rush/regime state;
- time to market/weekend closure;
- prior action and execution result.

### 4.4 `AssetIntent`

Policy output before customer-specific sizing:

```json
{
  "schema_version": "asset_intent.v1",
  "cell_id": "fx:EUR/USD@4h:policy-a",
  "asset_id": "fx:EUR/USD",
  "as_of": "2026-07-10T20:00:00Z",
  "valid_until": "2026-07-11T00:00:00Z",
  "action": "target",
  "target_exposure": 0.42,
  "confidence": 0.73,
  "strategy_rel_volume": 0.10,
  "risk_geometry": {
    "mode": "margin_aware_atr",
    "k_sl": 2.0,
    "k_tp": 3.0
  },
  "reason_codes": ["long_horizon_edge", "rush_confirmed"],
  "artifact_hash": "sha256:...",
  "config_hash": "sha256:..."
}
```

Actions support target exposure, explicit hold, close, modify protection, and
no-trade. Policies never emit customer account units.

### 4.5 `PortfolioIntent`

Contains next-period target weights, cash, constraints, confidence, allocator
identity, and rebalance validity.

Weights are keyed by cell, not only asset, so short/long strategies remain
auditable before LTS nets them by broker instrument.

### 4.6 `OrderIntent`

LTS output after user risk and capability checks:

- broker account reference;
- canonical asset and venue instrument;
- target/delta units;
- side/order type;
- SL/TP/trailing protection;
- idempotency key;
- source asset/portfolio intent IDs;
- preflight margin and risk calculation;
- expiration and fallback behavior.

### 4.7 `ExecutionReport`

Broker/simulator response normalized into:

- requested, accepted, filled, rejected, modified, or closed state;
- requested and filled units/prices;
- spread, slippage, commission, financing and conversion;
- broker transaction/order/trade IDs;
- timestamps and latency;
- resulting position/account snapshot;
- rejection or divergence reason.

### 4.8 `ComponentManifest`

Identifies a fitted component:

- role/domain;
- artifact URI/hash/size;
- resolved config/hash;
- code repository commits;
- dataset and feature hashes;
- training cutoff and seed;
- input/output contract versions;
- validation metric vector and coverage;
- compatible execution contract.

### 4.9 `DeploymentManifest`

Binds a complete compatible stack and release channel:

```json
{
  "schema_version": "deployment_manifest.v1",
  "release_id": "stack-2026w28-balanced-001",
  "channel": "stable",
  "valid_from": "2026-07-13T00:00:00Z",
  "training_cutoff": "2026-07-10T23:59:59Z",
  "components": {},
  "portfolio_cells": [],
  "validation_metrics": {},
  "compatibility": {},
  "rollback_release_id": "stack-2026w27-balanced-004",
  "signature": "..."
}
```

## 5. Canonical Experiment Configuration

Every `agent-multi` run is reconstructible from one nested resolved JSON:

```json
{
  "schema_version": "trading_experiment.v1",
  "experiment": {},
  "code": {},
  "data": {},
  "walk_forward": {},
  "environment": {},
  "context_encoder": {},
  "predictions": {},
  "rush_detector": {},
  "asset_policy": {},
  "lifecycle_policy": {},
  "risk": {},
  "portfolio": {},
  "training": {},
  "objectives": {},
  "optimization": {},
  "artifacts": {},
  "olap": {},
  "deployment": {}
}
```

### 5.1 Precedence

1. versioned program defaults;
2. optional base profile;
3. `--load_config` JSON;
4. explicit CLI overrides;
5. DOIN candidate genome patch.

`--config` can remain an alias where repositories already use it. The resolved
configuration is always written and hashed.

### 5.2 Portability

- Use artifact IDs and configured roots, not machine-specific paths in stored
  canonical config.
- Resolve local paths at runtime and record the resolution separately.
- Store secrets as environment/secret references only.
- Store plugin names and versions explicitly.
- Normalize tuples, numeric types and paths before hashing.

### 5.3 Legacy translation

Project 3 flat JSON files remain readable through a translator. Translation
emits warnings for ambiguous fields and produces a canonical nested config.
The original and translated hashes are retained.

## 6. DOIN Candidate Patch Contract

A DOIN genome is a typed patch, not a complete unrelated config:

```json
{
  "base_config_hash": "sha256:...",
  "genes": {
    "/asset_policy/learning_rate": 0.00012,
    "/risk/rel_volume": 0.08,
    "/lifecycle_policy/entry_threshold": 0.63
  }
}
```

Each gene schema declares:

- JSON pointer;
- type and range/choices;
- log/categorical transform;
- stage;
- activation condition;
- repair/compatibility constraints;
- mutation and crossover behavior.

Evaluators reconstruct and hash the full resolved config before evaluation.

## 7. Compatibility Rules

A component is composable only when:

- input/output schema major versions match;
- feature/token manifests satisfy required fields;
- normalization and vocabulary artifacts are available;
- asset/timeframe/cutoff semantics agree;
- action and execution contracts agree;
- model artifact framework/device requirements are satisfiable;
- no component's training cutoff exceeds the deployment cutoff;
- policy and lifecycle semantics do not both control the same field
  ambiguously.

Compatibility failures reject a stack before simulation.

## 8. Artifact Storage

Large artifacts remain in a content-addressed store. DOIN/OLAP records:

- content hash;
- immutable URI or retrieval descriptor;
- size and media type;
- component/deployment manifest hash;
- signatures and provenance.

Blockchain payloads do not carry full neural checkpoints, raw datasets, equity
traces, or logs.

## 9. Acceptance Criteria

- Schemas validate representative SAC, heuristic, rush, allocator, order and
  deployment objects.
- Canonical serialization produces identical hashes on two machines.
- Legacy Project 3 configs translate without silent value loss.
- Unknown major contract versions fail closed.
- No secret appears in resolved configs, OLAP, chain payloads, or artifacts.
- A deployment manifest can reconstruct every component and its validation
  evidence from hashes and immutable references.

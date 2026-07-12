# 06. OLAP Metrics and Lineage

## 1. Objective

Use DOIN's blockchain and local/PostgreSQL OLAP as the analytical memory of
optimization while adding trading-specific facts needed to compare candidates,
weeks, assets, stacks, releases and live execution. Analysts must not parse log
files to answer performance or lineage questions.

## 2. Existing DOIN Foundation

The current DOIN OLAP already tracks:

- domains and experiments;
- optimization rounds and parameters;
- performance, improvements and timing;
- peers, chain height and rewards;
- experiment summaries;
- accepted on-chain optimae;
- reported/verified performance, increments, config/data hashes, quorum and
  block provenance;
- local SQLite plus PostgreSQL sync and Metabase use.

This schema remains authoritative for network optimization facts.

## 3. Domain-General Metric Extension

The inherited `train_mae`, `val_mae`, and `test_mae` columns are useful for the
predictor domain but cannot represent arbitrary models. Add a backward-compatible
metric catalog and tall fact table instead of adding one column per new metric.

### 3.1 `dim_metric`

Fields:

- `metric_id` and versioned `metric_key`;
- domain/metric schema;
- label and description;
- unit and period (`fraction`, `%`, account currency, seconds, count);
- aggregation (`mean`, compound, max, min, quantile, sum, last);
- direction (`higher`, `lower`, constraint-only);
- provenance class (`train`, `validation`, `test`, `synthetic`, `shadow`,
  `live`);
- nullable/required and failure semantics.

### 3.2 `fact_metric_observation`

Grain: one metric value for one entity and scope.

Fields:

- entity type/id (`round`, `candidate`, `optimae`, `component`, `stack`,
  `release`, `asset_week`, `portfolio_week`, `live_execution`);
- metric ID;
- value numeric and optional JSON detail;
- domain, experiment, node, asset/cell, week and release references;
- coverage count and completeness;
- timestamp and data cutoff;
- config/artifact/dataset hashes;
- source (`local`, `chain`, `verification`, `provider`, `lts`).

Existing wide columns remain populated for predictor compatibility and can be
mirrored into the generic table.

## 4. Trading Dimensions

- `dim_config`: canonical resolved config and hash;
- `dim_code_snapshot`: repository commit bundle;
- `dim_dataset_snapshot`: manifests, cutoffs and hashes;
- `dim_asset`: canonical asset identity and class;
- `dim_cell`: asset, timeframe, data profile and policy role;
- `dim_venue_instrument`: broker mapping and capability snapshot;
- `dim_component_artifact`: role, framework, hash and URI;
- `dim_stack_release`: deployment manifest, channel and rollback;
- `dim_calendar_week`;
- `dim_machine`: host, GPU, memory and runtime;
- `dim_customer_risk_profile`: anonymized policy profile, never customer secret
  or account identity in public/on-chain metrics.

## 5. Trading Facts

### 5.1 `fact_candidate_run`

One candidate/seed/cutoff/fold evaluation:

- scalar fitness and status;
- L1/L2 stage/generation;
- runtime/resources;
- resolved config and artifact lineage;
- train/validation/test coverage;
- failure category;
- detailed metric observations.

### 5.2 `fact_asset_week`

One cell result per walk-forward week:

- start/end equity contribution;
- return, RAP, drawdown, MAE/MFE;
- trades, activity and exposure;
- costs;
- rush forecasts/labels;
- training cutoff and fitted artifact;
- completeness and anomaly flags.

### 5.3 `fact_portfolio_week`

One complete portfolio week:

- start/end NAV;
- return, RAP and drawdown;
- downside tail and expected shortfall inputs;
- turnover and costs;
- gross/net exposure, margin and concentration;
- intended/realized allocation;
- active/inactive cells;
- release/deployment identity.

### 5.4 `fact_order_lifecycle`

One logical or broker order lifecycle:

- source cell/portfolio intent;
- requested, accepted and filled state;
- quantities/prices and protection;
- timestamps and latency;
- costs and P&L;
- close reason;
- broker IDs and reconciliation status;
- simulation/live provenance.

### 5.5 `fact_rush_prediction`

- prediction timestamp and lead horizon;
- probability, direction, intensity and calibration version;
- realized label and outcome;
- false-positive/negative cost;
- policy and portfolio utility.

### 5.6 `fact_portfolio_allocation`

- proposed, constrained and executed weights;
- cash and risk budgets;
- previous weights and turnover;
- expected and realized contribution;
- constraint reasons and unavailable assets.

### 5.7 `fact_live_reconciliation`

- signal/intended order/broker order differences;
- provider, LTS and broker timestamps;
- position mismatch and repair action;
- stale/fallback/rollback reason;
- model release and customer risk profile class.

## 6. Required Metric Families

### 6.1 Return

- mean weekly return;
- ordered compounded annual return;
- cumulative return;
- median and quantile weekly return;
- positive/negative week counts.

### 6.2 Risk and RAP

- mean weekly RAP;
- annual RAP under the declared aggregation;
- mean/max weekly and full-path drawdown;
- downside deviation and semivariance;
- expected shortfall and loss quantiles;
- MAE/MFE;
- margin utilization and closeout events.

### 6.3 Trading quality

- trades/activity/exposure time;
- hit rate and payoff ratio;
- turnover;
- spread/slippage/commission/financing/conversion drag;
- early-close benefit/harm;
- rejected/duplicate/partial orders.

### 6.4 Stability

- seed and subperiod dispersion;
- worst-quarter/year segment;
- regime/rush conditional metrics;
- rank persistence;
- sensitivity to costs, prediction noise and sizing.

### 6.5 Portfolio

- concentration/HHI and effective number of bets;
- downside and ordinary correlation;
- diversification ratio;
- leave-one-cell-out contribution;
- risk budget utilization;
- rebalance turnover and allocation drift.

### 6.6 Rush detection

- precision-recall and average precision;
- Brier/calibration error;
- lead time and duration error;
- utility under exposure gate;
- opportunity capture and false-positive drawdown.

## 7. Metric Semantics

Every metric stores:

- formula/version;
- unit;
- period and annualization/compounding method;
- denominator;
- inclusion/exclusion rules;
- coverage weeks;
- data provenance.

Status/UI must never show a naked `0.02698` without unit, period, scope and
coverage.

## 8. Chain Versus Off-Chain Detail

On chain:

- candidate/optimae identity;
- domain and scalar fitness;
- compact configurable metric vector;
- parameters/config patch;
- config/data/code/artifact hashes;
- reported/verified values, quorum and reward;
- immutable release-manifest hash when promoted.
- content-addressed artifact descriptor and replication proofs, not the
  long-term model blob once the P2P artifact gate is promoted.

Off chain but hash-addressed:

- full resolved config;
- model/preprocessing artifacts;
- weekly/order fact extracts;
- equity traces;
- large diagnostics, plots and reports;
- raw datasets.

## 9. Analytical Queries

Initial Metabase/query pack:

- return versus RAP versus drawdown across all domains;
- performance versus `rel_volume`, SL/TP and sizing mode;
- asset/timeframe/data/model interaction effects;
- partial rush versus full-year stability;
- cost sensitivity and no-trade pathology;
- marginal portfolio contribution and correlated redundancy;
- parameter convergence and champion migration benefit per machine;
- metric improvement rate versus compute/time;
- weekly release versus live/shadow divergence;
- stale data, missing weeks and suspicious zero-trade clusters;
- Pareto frontiers by customer risk profile.

## 10. Retention and Backup

- OLAP and chain databases are backed up with integrity checks.
- Structured facts and manifests are retained before pruning artifacts.
- Checkpoints are content-addressed and deduplicated.
- Keep promoted, Pareto, anomaly and reproducibility artifacts.
- Require multiple independent pins for accepted champions; creator-only
  availability is insufficient.
- Non-champion blobs may expire according to explicit retention policy, but
  candidate metrics, content hash, producer and retention decision remain.
- Prune redundant intermediate checkpoints and verbose logs only after facts
  and error summaries are persisted.
- Support supervisor operation from another machine using Git plus restored
  chain/OLAP/artifact manifests.

## 11. Acceptance Criteria

- Arbitrary domain metrics persist without schema changes per metric.
- Existing predictor MAE dashboards remain functional.
- Every candidate can be joined to config, data, code and artifact lineage.
- Full annual return/RAP can be recomputed from weekly facts.
- On-chain accepted optimae reconcile with local candidate records.
- Metabase can compare assets/models/timeframes/risk across experiments.
- Retention removes no data required to reproduce a promoted or anomalous
  result.
- Removing the producing node does not make an accepted champion unavailable.

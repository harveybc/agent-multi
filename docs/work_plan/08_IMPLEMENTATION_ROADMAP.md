# 08. Implementation Roadmap

## 1. Delivery Strategy

Build vertical, reproducible increments. Every phase ends with an executable
artifact and gate. DOIN integration begins after the domain evaluator is stable,
because DOIN should optimize a fixed contract rather than chase simulator or
metric changes.

Codex owns phase sequencing, architecture, integration and acceptance. Claude
may receive bounded implementation packets that satisfy the delegation and
independent-review protocol in `12_COLLABORATIVE_IMPLEMENTATION_AND_REVIEW.md`.
Delegated completion does not advance a phase gate until Codex has inspected
the actual diff and reproduced the required evidence.

## 2. Dependency Graph

```text
Contracts/config
    |
    +--> heuristic pure policy extraction
    |
    +--> multi-asset gym-fx ledger
              |
              +--> agent-multi walk-forward evaluator
                         |
                         +--> generic OLAP metrics
                         |
                         +--> context/rush/risk/allocator
                         |
                         +--> DOIN trading plugins
                                      |
                                      +--> provider registry/inference
                                                   |
                                                   +--> LTS simulation
                                                            |
                                                            +--> OANDA practice
```

## 3. Phase 0: Freeze Contracts and Evidence

Repositories: `agent-multi`, new `trading-contracts`, `financial-data`.

Deliverables:

- contract package skeleton and version policy;
- canonical IDs and DTO schemas;
- nested experiment JSON Schema;
- config canonicalization/hash rules;
- Project 3 shortlist import manifest;
- metric catalog v1;
- architecture decision records.

Gate:

- schemas validate representative SAC, heuristic, portfolio, order and release
  examples;
- references to Project 3 candidates reconstruct from existing OLAP/config;
- no secret or machine path enters canonical config.

## 4. Phase 1: Configuration and Lineage Foundation

Repositories: `agent-multi`, `gym-fx`, `heuristic-strategy`, `doin-plugins`.

Deliverables:

- shared config loader/validator pattern;
- `--load_config` support and optional `--config` alias;
- resolved config output and hashes;
- portable artifact/data root resolution;
- legacy flat-config translators;
- repository commit bundle collection.

Gate:

- same config/data/code yields same resolved hash on two machines;
- existing Project 3 smoke config still runs through translation;
- plugin defaults cannot silently override loaded config.

## 5. Phase 2: Heuristic Lifecycle Policy Extraction

Repository: `heuristic-strategy`.

Deliverables:

- pure `TradeLifecyclePolicy` contract;
- legacy long/short prediction strategy refactored to `DecisionContext ->
  AssetIntent`;
- prediction source adapters for CSV, ideal, direct and provider API;
- Backtrader adapter and frozen legacy fixture;
- typed optimizable parameter schema;
- removal of embedded cost/simulator/HTTP logic from policy core.

Gate:

- refactored policy matches legacy decisions on the frozen fixture;
- source substitution does not change decisions for identical predictions;
- plugin can be loaded by `agent-multi` without running Backtrader.

## 6. Phase 3: Multi-Asset `gym-fx`

Repository: `gym-fx`.

Deliverables:

- heterogeneous master event clock;
- shared account/margin ledger;
- virtual cells and broker-instrument netting;
- cost/currency/financing/calendar capability profiles;
- order/protection lifecycle;
- deterministic event ledger;
- single-cell compatibility adapter.

Gate:

- hand-calculated ledger tests pass;
- future mutation and asset permutation tests pass;
- one-cell replay matches Project 3 within tolerance;
- heuristic decisions match Backtrader fixture;
- six cells can run with static weights and no accounting divergence.

## 7. Phase 4: First End-to-End Vertical Slice

Repositories: all domain repositories, no live broker.

Scope:

- one full-year asset cell;
- one SAC policy and one heuristic lifecycle control;
- static one-cell portfolio;
- complete config/artifact/metric lineage;
- direct and provider historical inference;
- LTS simulation broker execution.

Gate:

- identical intent stream across direct/provider paths;
- complete validation-year weekly metrics;
- LTS simulation reconciles target positions and costs;
- artifact is reconstructible from manifest on another machine.

This slice proves contracts before portfolio/model complexity is added.

## 8. Phase 5: Portfolio Walk-Forward and Baselines

Repositories: `agent-multi`, `gym-fx`, `financial-data`.

Deliverables:

- weekly training/evaluation for multiple cells;
- full validation/test coverage enforcement;
- static/equal/inverse-vol/min-variance/min-semivariance allocators;
- short and medium/long horizon buckets;
- per-order, asset-week and portfolio-week facts;
- no-trade and trivial policy controls.

Gate:

- at least three short and three medium/long candidates can complete validation;
- annual metrics recompute exactly from weekly facts;
- allocator baselines satisfy all constraints;
- test is inaccessible during selection.

## 9. Phase 6: Context, Rush, and Risk

Repositories: `agent-multi`, `financial-data`, `gym-fx`.

Deliverables:

- context token manifests and train-only encoder;
- local-only/engineered/attention/trainable ablations;
- causal rush labels, detector and calibration;
- base-policy rush gate;
- legacy and risk-at-stop sizing;
- SL/TP/early-close/risk geometry profiles.

Gate:

- no leakage in token/label/preprocessing audit;
- context improvement survives comparable compute/seed controls;
- rush detector improves downstream validation utility in multiple episodes;
- risk metrics and exposure behave monotonically in property tests.

## 10. Phase 7: Trainable Portfolio Allocator

Repository: `agent-multi`.

Deliverables:

- constrained learned allocator;
- causal allocation observations;
- turnover/cost/concentration/margin penalties;
- optional cadence parameter;
- marginal contribution and leave-one-cell-out analysis;
- conservative/balanced/aggressive Pareto candidates.

Gate:

- learned allocator beats declared baselines after costs on validation;
- no hidden concentration or current-week test input;
- weights, cash and risk budgets always satisfy constraints;
- result is stable across seeds/subperiods.

## 11. Phase 8: DOIN Trading Domains

Repositories: `doin-plugins`, `agent-multi`, `doin-node`, and `doin-core` only
for existing interfaces or a proven backward-compatible requirement. No
standalone optimizer/evaluator runtime is introduced.

Deliverables:

- optimization/inference/synthetic-data entry points;
- domain metadata and typed staged genomes;
- `agent-multi` callbacks for existing migration/stage/candidate events;
- deterministic synthetic multi-asset verification generator;
- immutable artifact references;
- trackerless content-addressed model distribution, provider discovery,
  multi-peer champion pinning and replication proofs;
- generic metric OLAP persistence and dashboard metadata;
- canonical experiment config plus generated per-machine unified-node configs;
- machine-specific node configs for omega, dragon, gamma 5070 Ti and gamma
  5090 workers.

Gate:

- existing controlled flooding/GossipSub and Proof of Optimization remain
  unchanged;
- three nodes exchange/inject compatible champions;
- evaluator quorum reproduces metric and data hashes;
- new node sync/recovery works;
- champion recovery succeeds after the producing node is offline;
- all configured trading metrics reach OLAP and chain views.

Recommended parallel delegation candidates in this phase are schema fixtures,
config generators, compatibility translators and isolated plugin adapters.
Protocol behavior, Proof of Optimization semantics, evaluator trust rules,
fitness definitions and cross-repository integration remain Codex-owned unless
a task packet explicitly limits the work to a non-semantic mechanical change.

## 12. Phase 9: Continuous Release Pipeline

Repositories: `agent-multi`, `prediction_provider`.

Deliverables:

- candidate-to-component promotion controller;
- stack compatibility evaluator;
- signed component/deployment manifests;
- Pareto release selection;
- stable/adaptive/experimental/pinned channels;
- provider artifact preload, health and rollback;
- batch forecast/asset/portfolio inference.

Gate:

- accepted candidate does not auto-deploy;
- release gates are deterministic and auditable;
- provider historical output matches direct inference;
- rollback activates prior artifact without stale state.

## 13. Phase 10: LTS Integration

Repository: `lts`.

Deliverables:

- model-channel customer configuration;
- signal bundle validation;
- customer risk overlays;
- virtual sleeve aggregation;
- idempotent target-position planner;
- broker capability/reconciliation contract;
- simulation parity and failure policies;
- privacy-safe model usage attribution.

Gate:

- no duplicate order under restart/retry;
- user limits dominate model suggestions;
- multiple cells net correctly;
- simulation replay matches `gym-fx` intent/order behavior within declared
  execution differences.

## 14. Phase 11: OANDA Practice

Repositories: `lts`, `prediction_provider`.

Deliverables:

- account instrument discovery and mappings;
- read-only shadow operation;
- minimal-size practice execution;
- transaction reconciliation;
- weekly rebalance/flatten operation;
- failure, stale, rollback and kill-switch exercises;
- simulation-versus-practice OLAP comparison.

Gate:

- all positions/orders reconcile;
- no unexplained divergence;
- unsupported instruments fail closed;
- portfolio/margin limits hold;
- operator demonstrates emergency flatten and release rollback.

## 15. Phase 12: Frozen Test and Production Decision

Deliverables:

- frozen code/config/data/artifacts;
- untouched full test-year evaluation;
- return, RAP, tail, drawdown, costs, concentration and stability review;
- practice evidence review;
- production or rejection decision;
- conservative initial live caps if approved.

Gate: explicit human approval. DOIN optimization success alone cannot authorize
real-capital deployment.

## 16. Work Item Template

Every implementation issue includes:

- owning repository and module;
- contract/config versions;
- inputs/outputs and forbidden dependencies;
- deterministic fixture;
- unit/integration/acceptance tests;
- OLAP facts/metrics emitted;
- migration/backward compatibility;
- resource estimate and machine assignment;
- failure/rollback procedure;
- gate evidence location.

## 17. Definition of Done

A phase is complete only when:

- code and tests are committed;
- config and schemas are versioned;
- artifacts/metrics are reproducible;
- documentation and decision log are updated;
- no required process is running from ad hoc shell/cron state;
- the next phase can start from a clean checkout on another workstation.

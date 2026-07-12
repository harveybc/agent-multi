# Adaptive Multi-Asset Trading Work Plan

Status: architecture approved; implementation in progress
Plan version: 1.3.0
Date: 2026-07-12
Primary implementation repository: `agent-multi`

## Mission

Build a reproducible, continuously optimized, multi-asset trading system on top
of the existing DOIN network. The system must support:

- per-asset and per-timeframe trading policies;
- deterministic heuristic and learned actor-critic policies;
- variable-length market context encoders;
- causal rush/opportunity detection;
- trade lifecycle management, including early close;
- weekly walk-forward retraining or fine-tuning;
- weekly portfolio allocation, with cadence later treated as optimizable;
- decentralized Level 2 optimization and verification through DOIN;
- model serving through `prediction_provider`;
- customer-specific risk and broker execution through LTS;
- OANDA practice/live execution where instruments are available;
- complete candidate, metric, lineage, and deployment traceability.

## Critical Premise

DOIN already works. Controlled flooding/GossipSub, Proof of Optimization,
commit-reveal, evaluator quorum, champion migration, stage synchronization,
blockchain history, task queues, inference, incentives, and OLAP tracking are
existing infrastructure. This project adds trading domains and domain metrics
through the established plugin interfaces. It does not redesign DOIN.

The extension boundary is:

```text
DOIN infrastructure (existing and stable)
        |
        +-- doin.optimization/trading_*
        +-- doin.inference/trading_*
        +-- doin.synthetic_data/trading_*
        +-- trading metric catalog and artifact references
```

Any proposed change to `doin-core` or the network protocol requires a failing
integration test showing that the existing plugin contract cannot support the
trading domain.

## Document Map

| Document | Purpose |
| --- | --- |
| [01 Architecture](01_SYSTEM_ARCHITECTURE.md) | System boundaries, repositories, decision layers, clocks, and ownership |
| [02 Contracts and configuration](02_CONTRACTS_AND_CONFIGURATION.md) | Shared DTOs, JSON configuration, hashes, compatibility, and artifacts |
| [03 Simulation and execution parity](03_MULTI_ASSET_SIMULATION_AND_EXECUTION.md) | `gym-fx`, Backtrader, account ledger, costs, time alignment, and broker parity |
| [04 Models and training](04_MODELS_POLICIES_AND_TRAINING.md) | Actor-critic, heuristic policies, context, rush, risk, allocation, and walk-forward training |
| [05 DOIN integration](05_DOIN_TRADING_DOMAIN_INTEGRATION.md) | Domain plugins, genomes, verification, migration, releases, and node configs |
| [06 OLAP and analytics](06_OLAP_METRICS_AND_LINEAGE.md) | Metrics, star schema extensions, cross-experiment analysis, and retention |
| [07 Serving and live trading](07_SERVING_LTS_AND_OANDA.md) | `prediction_provider`, LTS, customer channels, OANDA, safety, and rollback |
| [08 Roadmap](08_IMPLEMENTATION_ROADMAP.md) | Phases, dependencies, deliverables, gates, and first vertical slice |
| [09 Verification and operations](09_TESTING_SECURITY_AND_OPERATIONS.md) | Tests, leakage controls, deterministic verification, monitoring, and incident handling |
| [10 Decisions and evidence](10_DECISIONS_OPEN_QUESTIONS_AND_EVIDENCE.md) | Accepted decisions, deferred choices, evidence, and authoritative references |
| [11 DOIN configuration profiles](11_DOIN_CONFIGURATION_PROFILES.md) | Unified-node runtime, common experiment configs, machine overlays, generation, and validation |
| [12 Collaborative implementation](12_COLLABORATIVE_IMPLEMENTATION_AND_REVIEW.md) | Codex ownership, bounded Claude task packets, independent review, and acceptance evidence |
| [13 Implementation status](13_IMPLEMENTATION_STATUS_AND_TASK_LEDGER.md) | Executed increments, verification evidence, delegation ledger, and immediate next tasks |
| [14 Simulation engine selection](14_SIMULATION_ENGINE_SELECTION_2026_07_11.md) | NautilusTrader/LEAN/Backtrader bake-off and no-reimplementation rule |

## Repository Ownership Summary

| Repository | Responsibility |
| --- | --- |
| `financial-data` | Versioned causal datasets, feature packs, event tokens, calendars, manifests and hashes |
| `trading-contracts` (new, lightweight) | Dependency-free shared schemas for predictions, intents, execution, metrics, manifests, and canonical IDs |
| `gym-fx` | Adapter/Gym integration around the selected canonical simulation engine |
| `heuristic-strategy` | Reusable trade lifecycle policy plugins and Backtrader compatibility adapters |
| `agent-multi` | Model training, policy evaluation, walk-forward orchestration, portfolio evaluation, and artifact export |
| `doin-core` | Existing protocol models, cryptography, consensus primitives and plugin interfaces |
| `doin-node` | The active unified runtime: optimization, evaluation, inference, relay, chain, dashboard and OLAP |
| `doin-plugins` | Existing and new domain adapters loaded by the unified node, including thin trading adapters |
| `prediction_provider` | Artifact resolution, model loading, inference, signal bundles, and deployment channels |
| `lts` | Multi-user portfolio state, customer risk overlays, broker planning, execution, reconciliation, and audit |

## Source-of-Truth Hierarchy

When documents, tables, and runtime data disagree, use this order:

1. immutable dataset, config, code, and artifact hashes;
2. DOIN accepted candidate and verification records;
3. structured weekly/order facts in OLAP;
4. resolved run configuration and manifests;
5. generated reports and plots;
6. prose documents.

No result is reproducible from a Markdown table alone.

## Implementation Authority

Codex is the technical lead, primary implementer, integration owner, and final
reviewer for this work plan. A Claude coding agent may implement bounded,
parallelizable task packets when that reduces delivery time. Delegation never
transfers architectural authority or acceptance responsibility. Claude output
is treated as an untrusted contribution until its diff, assumptions, tests and
runtime behavior pass the independent review protocol in document 12.

## Non-Negotiable Experimental Rules

- Selection, early stopping, optimization, migration, and allocation never use
  the protected test period.
- Validation and test each target one complete chronological year evaluated by
  weekly walk-forward retraining/fine-tuning.
- Partial candidates are labeled with their exact number of observed weeks.
- Annual metrics are calculated from an ordered annual weekly series, not by
  renaming a partial-period mean.
- Every candidate records resolved config, data hash, code commits, seed,
  metrics, weekly coverage, and artifact hash.
- All fitting, normalization, vocabulary construction, and feature selection
  happen inside the training cutoff.
- Simulation and live execution use the same intent and execution contracts.
- OANDA availability is discovered from the account; it is never inferred from
  a research symbol.

## Initial Evidence-Backed Universe

The Project 3 shortlist seeds the search but does not freeze it.

| Role | Initial cells | Evidence status |
| --- | --- | --- |
| Short-horizon seeds | `SOLUSDT@1h`, `BTCUSDT@1h`, `ADAUSDT@1h`; alternates `XRPUSDT@1h`, `ETHUSDT@1h` | Partial 5-12 week screening evidence |
| Medium/long seeds | `SOLUSDT@4h`, `EURUSD@4h`, `DOGEUSDT@4h`, `AUDUSD@4h`, `EURJPY@4h`, `ETHUSDT@4h` | Screening rows with 52 test weeks; optimization must use validation, not test rank |
| Rush research seeds | `SOLUSDT@4h`, `ETHUSDT@4h`, `ADAUSDT@4h` | Partial diagnostic episodes only |

Portfolio eligibility requires full validation coverage, stability, liquidity,
broker support, and marginal diversification contribution. A weak control may
remain in experiments when it helps estimate whether a component is genuinely
adding value.

## Current Build Order

1. Shared schemas and canonical nested configuration.
2. Single-cell compatibility and multi-asset account ledger.
3. Six-cell static-allocation full-year validation replay.
4. Baseline allocators and generic OLAP metrics.
5. Project 3 SAC and heuristic lifecycle adapters.
6. Context encoder and rush detector.
7. Trainable portfolio allocator.
8. DOIN trading domain plugins and deterministic verification.
9. Model registry and `prediction_provider` serving.
10. LTS simulation parity and OANDA practice.

The detailed gates and deliverables are in [08 Implementation Roadmap](08_IMPLEMENTATION_ROADMAP.md).

## Updating This Plan

- Add new detail to the owning document instead of expanding this index.
- Record architectural changes in document 10 with date, rationale, and
  superseded decision.
- Increment the plan version for contract or gate changes.
- Do not silently alter historical acceptance criteria after seeing test or
  live outcomes.

# Adaptive Multi-Asset Trading Work Plan

Status: E0-E4 complete; protected-entry v2 campaign prepared; multi-venue Paper commissioning active; bounded social-intelligence, independent-audit and academic-preservation tracks specified
Plan version: 1.21.0
Date: 2026-07-31
Primary implementation repository: `agent-multi`

## Mission

Build a reproducible, continuously optimized, multi-asset trading system on top
of the existing DOIN network. The system must support:

- per-asset and per-timeframe trading policies;
- deterministic heuristic and learned actor-critic policies;
- variable-length market context encoders;
- causal rush/opportunity detection;
- trade lifecycle management, including early close;
- adaptive market/limit/stop/MIT order selection with explicit fill,
  adverse-selection and missed-opportunity modeling;
- weekly walk-forward retraining or fine-tuning;
- weekly portfolio allocation, with cadence later treated as optimizable;
- decentralized Level 2 optimization and verification through DOIN;
- model serving through `prediction_provider`;
- customer-specific risk and broker execution through LTS;
- multi-venue paper/live execution through account-specific broker adapters;
- bounded social intelligence, source-backed technical participation and
  recoverable low-cost agent operations;
- a reproducible IEEE-compatible paper series and later synthesis book;
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
| [07 Serving and live trading](07_SERVING_LTS_AND_OANDA.md) | `prediction_provider`, LTS, multi-venue routing, safety, and rollback |
| [08 Roadmap](08_IMPLEMENTATION_ROADMAP.md) | Phases, dependencies, deliverables, gates, and first vertical slice |
| [09 Verification and operations](09_TESTING_SECURITY_AND_OPERATIONS.md) | Tests, leakage controls, deterministic verification, monitoring, and incident handling |
| [10 Decisions and evidence](10_DECISIONS_OPEN_QUESTIONS_AND_EVIDENCE.md) | Accepted decisions, deferred choices, evidence, and authoritative references |
| [11 DOIN configuration profiles](11_DOIN_CONFIGURATION_PROFILES.md) | Unified-node runtime, common experiment configs, machine overlays, generation, and validation |
| [12 Collaborative implementation](12_COLLABORATIVE_IMPLEMENTATION_AND_REVIEW.md) | Codex ownership, bounded Claude task packets, independent review, and acceptance evidence |
| [13 Implementation status](13_IMPLEMENTATION_STATUS_AND_TASK_LEDGER.md) | Executed increments, verification evidence, delegation ledger, and immediate next tasks |
| [14 Simulation engine selection](14_SIMULATION_ENGINE_SELECTION_2026_07_11.md) | NautilusTrader/LEAN/Backtrader bake-off and no-reimplementation rule |
| [15 Distributed campaign lifecycle](15_DISTRIBUTED_CAMPAIGN_LIFECYCLE.md) | Replicated job order, convergence/stop barriers, recovery, champion archive, and swarm history |
| [16 Flat-fitness root cause](16_FLAT_FITNESS_ROOT_CAUSE_2026_07_19.md) | BTC actor saturation evidence, corrected feature contract, action-collapse guard, and campaign disposition |
| [17 Data and preprocessing evidence recovery](17_DATA_PREPROCESSING_EVIDENCE_RECOVERY.md) | Parameter registry, canonical metrics, source coverage, hierarchical sweeps, transactional pool, and DOIN-resume conditions |
| [18 Full-genome per-asset optimization](18_FULL_GENOME_PER_ASSET_OPTIMIZATION.md) | Corrected E4 interpretation, L1 convergence protocol, mixed genome, execution order, and champion contract |
| [19 Execution curriculum and order routing](19_EXECUTION_CURRICULUM_AND_ORDER_ROUTING.md) | Visible cost curriculum, immutable robust fitness, multi-timescale execution-state models, market/limit/stop/MIT policy, artifact contract, and campaign transition |
| [20 Protected execution incident](20_PROTECTED_EXECUTION_ACTIVITY_GATE_INCIDENT_2026_07_29.md) | One-trade champion root cause, coordinated stop, corrected activity/SL-TP/metric contract, v2 lineage and relaunch gates |
| [21 OANDA Practice execution lab](21_OANDA_PRACTICE_EXECUTION_REALITY_LAB.md) | Broker-observation track, OLAP-backed asset selection, protected canaries, day/week metrics, and activation gates |
| [22 Multi-venue paper execution](22_MULTI_VENUE_PAPER_EXECUTION_AND_SOCIAL_TRADING.md) | OANDA MT5, Alpaca Paper, IBKR Paper, global LTS ledger, venue routing, account status, and social-trading boundary |
| [23 Social intelligence and continuity](23_SOCIAL_INTELLIGENCE_AND_OPERATIONAL_CONTINUITY.md) | Hermes/Moltbook research, local-model cost routing, social OLAP, bounded publishing, DOIN domain discovery, and recoverable operations |
| [24 Independent audit and continuous improvement](24_INDEPENDENT_AUDIT_AND_CONTINUOUS_IMPROVEMENT.md) | Claude audit authority, cross-front checks, cadence, evidence, Hermes boundary, findings, and Codex role recovery |
| [25 Academic publication and reproducibility](25_ACADEMIC_PUBLICATION_AND_REPRODUCIBILITY.md) | Five-paper research program, IEEE-compatible outlines, claim/citation ledgers, evidence gates, reproducibility packages, disclosure, and later synthesis book |

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

A separate Claude conversation may operate as the read-mostly independent
auditor defined in document 24. It can challenge Codex and propose corrections,
but it cannot orchestrate runtime, alter architecture, direct Hermes, operate
brokers or close its own material findings.

## Non-Negotiable Experimental Rules

- Selection, early stopping, optimization, migration, and allocation never use
  the protected test period.
- Cheap data/preprocessing screens may use a fixed-fit proxy, but their protocol
  is explicit and their metrics cannot be relabeled as deployable SAC results.
- Per-asset DOIN optimization uses the declared static chronological train and
  complete validation-year protocol. The protected test year is evaluated once
  after selection and cannot affect optimization.
- Weekly walk-forward retraining/fine-tuning is a later stack-confirmation
  experiment. It does not block obtaining the optimized per-asset artifacts
  required to begin portfolio mechanics.
- Every result records its exact evaluation dates and observed week count.
- Annual metrics are calculated from an ordered annual weekly series, not by
  renaming a partial-period mean.
- Every candidate records resolved config, data hash, code commits, seed,
  metrics, weekly coverage, and artifact hash.
- All fitting, normalization, vocabulary construction, and feature selection
  happen inside the training cutoff.
- Simulation and live execution use the same intent and execution contracts.
- Every risk-increasing market, limit or stop entry carries both a stop loss
  and take profit. Plugin failure rejects the entry; it never falls back to an
  unprotected order.
- A candidate must satisfy split-specific activity eligibility. The current
  annual validation floor is 12 completed trades; this is not a positive-profit
  gate.
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

1. E0-E3 evidence screening: complete.
2. E4 fixed-SAC integration baselines and load-tested warm-start artifacts:
   complete. These are not optimized per-asset champions.
3. Run the immutable protected-entry v2 full-genome job under `easy_floor`
   non-zero costs. Optimize data, preprocessing, model, risk geometry and
   market/limit/stop/adaptive routing while enforcing annual activity.
4. Archive the exact eligible champion and diverse elites, then warm-start a
   distinct easy-to-nominal-to-stress curriculum domain selected by robust
   weekly RAP.
5. Repeat one coordinated DOIN Level 2 campaign per asset, sequentially, with
   all workers sharing one chain and candidate pool.
6. Freeze each alpha/execution champion and pass an execution-data fidelity
   gate before learning higher-fidelity order placement.
7. Train calibrated fill-time, adverse-selection, short-path and event-hazard
   auxiliaries; retain protected market-only and deterministic routing as
   mandatory controls.
8. Freeze the optimized per-asset cell library and optimize static portfolio
   allocation using those artifacts.
9. Add probabilistic rush activation and causal event conditioning, then
   compare weekly retraining/fine-tuning after the static portfolio vertical is
   operational.
10. In parallel with optimization, run the multi-venue execution-reality lab:
    account/instrument preflight, 24-hour read-only observation, protected
    canaries, then a seven-day consolidated portfolio shadow.
11. Complete registry, serving, LTS parity, and controlled paper execution
    across the account-compatible venue adapters.
12. Add the social-intelligence track without competing with optimization:
    deterministic collection, Telegram review, measured local-model bake-off,
    then bounded publication and a clean VPS continuity drill.
13. Run change-driven independent audits across the three runtime fronts, with Codex
    reproducing findings and maintaining a versioned technical-lead recovery
    prompt.
14. Preserve validated contributions through the five-paper academic program,
    without allowing publication work to relabel incomplete evidence or
    interfere with protected experiments.

The detailed gates and deliverables are in [08 Implementation Roadmap](08_IMPLEMENTATION_ROADMAP.md).

## Updating This Plan

- Add new detail to the owning document instead of expanding this index.
- Record architectural changes in document 10 with date, rationale, and
  superseded decision.
- Increment the plan version for contract or gate changes.
- Do not silently alter historical acceptance criteria after seeing test or
  live outcomes.

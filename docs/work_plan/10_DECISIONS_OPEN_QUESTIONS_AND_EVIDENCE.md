# 10. Decisions, Open Questions, and Evidence

## 1. Accepted Decisions

### ADR-001: DOIN is a stable substrate

Date: 2026-07-10

Controlled flooding/GossipSub, Proof of Optimization, chain, commit-reveal,
quorum, migration, inference, incentives and OLAP are reused. Trading work is
implemented through existing plugins/configs. Network redesign is out of scope.

### ADR-002: No separate portfolio orchestrator repository

Training/simulation orchestration belongs to `agent-multi`/`gym-fx`. Production
multi-user orchestration belongs to LTS. Creating a third owner would duplicate
state and business rules.

### ADR-003: Create a small shared `trading-contracts` package

Multiple independent repositories need identical versioned DTOs. The package
contains schemas only and avoids dependencies between training, serving and
live execution repositories.

### ADR-004: Keep and refactor `heuristic-strategy`

It becomes the owner of pure trade lifecycle policy plugins. Backtrader remains
an adapter/regression harness. Prediction transport, simulation, costs and
optimization are separated from policy logic.

### ADR-005: one proven engine owns research execution semantics

Portfolio accounting cannot be split across custom Gym code, strategy plugins,
Backtrader and LTS. `gym-fx` is the adapter/Gym package around NautilusTrader
1.230.0, selected by the bounded bake-off in
`14_SIMULATION_ENGINE_SELECTION_2026_07_11.md`. Backtrader remains the parity
control/fallback and LEAN is deferred. No parallel custom ledger may compete
with Nautilus. Until upstream margin-account pre-trade checks are complete, the
adapter uses Nautilus's own margin calculator and free-balance/xrate values for
a narrow auditable preflight.

### ADR-006: LTS is the only live-order authority

DOIN candidates, provider inference and agent policies emit recommendations or
intents. LTS applies customer and broker constraints, executes and reconciles.

### ADR-007: Continuous optimization, controlled deployment

DOIN may improve continuously. New artifacts are trained and released on a
declared boundary after validation. An accepted optimae never auto-orders or
silently replaces a live model.

### ADR-008: Component and stack champions coexist

Each role can be optimized independently, but deployable releases bind a
compatibility-tested stack. Best individual components are not assumed to be
the best composition.

### ADR-009: Maintain Pareto releases

Return, RAP, drawdown, tail, cost, stability and concentration cannot be reduced
to one universally best customer model. Conservative, balanced and aggressive
release channels select feasible Pareto candidates.

### ADR-010: One logical position per cell initially

Many assets/cells may be active concurrently. Same-instrument cells are virtual
sleeves and net at the broker. Multiple independent positions per cell are a
future experiment after accounting/reconciliation is stable.

### ADR-011: Weekly is a default, not a theorem

Retraining and rebalance begin weekly because of the established business and
Project 3 protocol. Their cadences remain independent configurable parameters
and can be optimized after costs and operational constraints are stable.

### ADR-012: Synthetic verification supplements real validation

DOIN deterministic synthetic scenarios support quorum trust and stress tests.
Chronological full-year validation remains the trading optimization objective.

### ADR-013: `doin-node` is the only active DOIN runtime process

The current unified node performs optimizer, evaluator, inference and relay
roles. Trading depends on `doin-core`, `doin-node` and `doin-plugins`; historical
standalone optimizer/evaluator repositories are not deployment dependencies.

### ADR-014: Two-level source config, generated full node config

The optimized repository owns one common experiment JSON. Each machine owns a
small runtime overlay. A generator materializes the complete current
`doin-node` JSON schema and hashes both common experiment semantics and
machine-specific runtime config.

### ADR-015: Codex-led implementation with verified Claude delegation

Codex remains technical lead, primary implementer, integration owner and final
reviewer. Claude may implement carefully bounded tasks from a complete task
packet. Its report is a claim to verify, not acceptance evidence. Codex must
inspect the resulting diff, challenge assumptions, rerun relevant tests and
verify contracts across affected repositories before a delegated task can pass
a phase gate. The user may relay task packets and responses, but is not expected
to perform the technical review.

### ADR-016: Model artifacts are content-addressed and peer-replicated

The proven predictor path remains the compatibility baseline: a candidate is
saved as `.keras`, a new champion is encoded into the optimae parameters, and
the accepted transaction makes it available through chain replication. Trading
adapters must preserve that behavior until the artifact layer is promoted.

Large model bytes do not remain the long-term blockchain payload. The chain
stores an immutable artifact descriptor containing content hash, size, format,
chunk/tree hash, producing peer, config/data/code lineage and replication
proofs. Model bytes are distributed by a trackerless, content-addressed P2P
backend. No central artifact server, database or permanent seed node is an
availability dependency.

Accepted champions must be pinned by multiple independent peers before they
are considered recoverable. Non-champion candidate artifacts may be advertised
and retained under explicit top-K, Pareto, anomaly or reproducibility policies;
their metrics and artifact hashes remain in OLAP-on-chain even when their blobs
expire.

### ADR-017: Off-policy champions require a post-learning checkpoint

An L1 checkpoint from an off-policy agent is ineligible until its timestep
count is strictly greater than `learning_starts`. Warm-up checkpoints cannot
become best, cannot consume patience and cannot be exported as champions. If a
run never crosses that barrier it fails explicitly.

The BTCUSDT 1h v1 domain is permanently non-promotable because 75 candidates
with distinct DEAP parameters all selected the same seeded 4000-step policy
before SAC's `learning_starts: 5000`. Corrected asset domains receive new domain
identities and chains. Previously completed SOLUSDT 4h and 1h artifacts remain
accepted because their stored models contain 8000 timesteps and therefore
include learning updates.

### ADR-018: Enriched policies fail closed on input or action collapse

Date: 2026-07-19

A data-profile label is not evidence that its columns reached the model. Every
enriched asset-policy config materializes its exact feature columns, applies
causal scaling, excludes the raw-price bypass and requires the feature-aware
preprocessor. Generated DOIN node configs derive plugin identity from that
canonical config instead of retaining stale overrides.

Train-tail and validation rollouts record raw action dispersion and dominant
side. A deterministic policy that is constant on every diagnostic split is an
invalid candidate and cannot publish a champion. Historical models trained
with the price-only preprocessor remain reproducible only under the explicit
`price_state_only` label; their results cannot be attributed to an ignored
feature pack.

## 2. Open Questions and Decision Gates

| Question | Current default | Decision gate |
| --- | --- | --- |
| Exact portfolio shortlist | Project 3 evidence-backed seed universe | After all candidates receive comparable full validation coverage |
| Retrain cadence | Weekly | After stable walk-forward pipeline and cadence/cost experiment |
| Rebalance cadence | Weekly weekend | After allocator baselines and turnover analysis |
| One or several positions per cell | One | After LTS netting/reconciliation and marginal utility evidence |
| Risk sizing | Legacy notional first | After reproduction; compare risk-at-stop |
| Rush specialist | Exposure gate | Train separate policy only if gate adds reproducible utility |
| Context architecture | Small masked attention ladder | Increase complexity only after ablation benefit |
| DOIN decentralized live inference | Optional for non-urgent tasks | After latency/reliability/payment tests |
| OANDA live universe | Discovered per account | At account capability preflight |
| Additional crypto broker | None initially | After selected live crypto asset lacks OANDA route |
| P2P artifact backend | Evaluate trackerless BitTorrent/libtorrent versus IPFS/Kubo; no central dependency | Before multi-node trading-domain acceptance |
| Generic DOIN metric persistence design | Tall metric fact plus catalog | Before trading domain three-node run |
| Separate `inference_config`/`synthetic_data_config` fields | Prefer explicit subtrees with backward-compatible `optimization_config` fallback | Unified-node config loader audit in Phase 8 |
| Customer usage incentives | Privacy-safe aggregate attribution | Separate economic/security design review |

Open questions do not block earlier phases unless their gate is reached.

## 3. Project 3 Evidence

Relative to this document directory:

- `../../../financial-data/work_plan/PROJECT3_WEEKLY_RETRAINED_PORTFOLIO_PROTOCOL_2026_05_22.md`
- `../../../financial-data/work_plan/PROJECT3_EVENT_TOKEN_TRANSFORMER_AGENT_SPEC_2026_06_17.md`
- `../../../financial-data/work_plan/project3_orchestrator_event_context_representation_addendum_2026_06_09.md`
- `../../../financial-data/work_plan/PROJECT3_PORTFOLIO_SUPERVISOR_V2_RESEARCH_INGEST_2026_06_17.md`
- `../../../financial-data/work_plan/PROJECT3_OLAP_TRANSVERSAL_ANALYSIS_PLAN_2026_06_29.md`
- `../../../financial-data/work_plan/PROJECT3_OLAP_TRANSVERSAL_ANALYSIS_REPORT_2026_06_29.md`
- `../../../financial-data/work_plan/PROJECT3_DOIN_HANDOFF_CANDIDATE_PACK_2026_07_05.md`
- `../../../financial-data/work_plan/PROJECT3_DOIN_SELECTION_TABLE_2026_07_10.md`
- `../../../financial-data/work_plan/selection_artifacts_2026_07_10/doin_exploration_shortlist.csv`

Numerical reconstruction uses OLAP/config/artifact lineage rather than copied
Markdown values.

## 4. Existing Code Evidence

### DOIN

- `doin-core/src/doin_core/plugins/base.py`: authoritative optimization,
  inference and synthetic-data plugin interfaces.
- `doin-core/src/doin_core/protocol/messages.py`: optimae, candidate, stage,
  task, inference, champion and controlled-flooding message contracts.
- `doin-node/src/doin_node/network/flooding.py`: TTL/dedup controlled flooding.
- `doin-node/src/doin_node/stats/olap_schema.py`: current experiment/round/chain
  star schema.
- `doin-node/src/doin_node/unified.py`: unified optimizer/evaluator/inference
  roles and `DomainRole` contract.
- `doin-node/src/doin_node/cli.py`: current unified-node JSON loader and the
  authoritative list of consumed fields.
- `doin-plugins/src/doin_plugins/predictor/optimizer.py`: proven pattern for
  wrapping an existing optimizer and injecting network champions.

### Trading repositories

- `agent-multi/app/main.py`: existing plugin orchestration and config entry.
- `agent-multi/agent_plugins/project3_sac_actor_critic_agent.py`: Project 3 SAC
  starting policy.
- `agent-multi/pipeline_plugins/rl_pipeline_with_validation.py`: L1 and split
  behavior to preserve/refactor.
- `gym-fx/gym_fx/env.py` plus `broker_plugins`, `strategy_plugins`,
  `reward_plugins`, `metrics_plugins`, `preprocessor_plugins`, and
  `data_feed_plugins`: environment and execution extension points.
- `heuristic-strategy/app/plugins/plugin_long_short_predictions.py`: legacy
  long/short prediction lifecycle behavior.
- `heuristic-strategy/app/plugins/plugin_api_predictions.py`: provider-based
  entry/exit behavior and current coupling to extract.
- `lts/plugins_portfolio/default_portfolio.py`: existing portfolio concept.
- `lts/plugins_broker/oanda_broker.py`: existing OANDA prototype.
- `prediction_provider/setup.py`: current inference plugin system.

### Configuration examples

- `doin-node/examples/predictor_omega_node_tft_binary_neat.json`
- `doin-node/examples/predictor_gamma_node_tft_binary_neat.json`
- `predictor/examples/config/phase_1c_direction/optimization/phase_1c_direction_tcn_direction_long_1d_optimization_config.json`
- `predictor/examples/config/phase_1c_direction/optimization/phase_1c_direction_tcn_direction_short_1d_optimization_config.json`

## 5. DOIN Documentation

- `doin-core/README.md`
- `doin-core/docs/NETWORK.md`
- `doin-core/docs/SECURITY.md`
- `doin-core/docs/SCALABILITY.md`
- `doin-node/README.md`
- `doin-plugins/README.md`

These documents remain authoritative for DOIN operation. This work plan defines
only trading-domain behavior.

## 6. External References

- OANDA REST-v20 introduction:
  https://developer.oanda.com/rest-live-v20/introduction/
- OANDA account/instruments:
  https://developer.oanda.com/rest-live-v20/account-ep/
- OANDA pricing:
  https://developer.oanda.com/rest-live-v20/pricing-ep/
- OANDA orders:
  https://developer.oanda.com/rest-live-v20/order-ep/
- OANDA trades/dependent orders:
  https://developer.oanda.com/rest-live-v20/trade-ep/

## 7. Change Log

### 1.2.0 - 2026-07-10

- Began Phase 0/1 execution with the `trading-contracts` package, canonical
  configuration resolver, schema export and strict lineage hashes.
- Added the initial metric catalog and a hash-verified Project 3 shortlist
  import manifest.
- Added an implementation/task ledger and the first bounded Claude packet for
  unified-node configuration materialization.

### 1.1.0 - 2026-07-10

- Added Codex-led implementation ownership and final acceptance authority.
- Added bounded, versioned Claude task packets with explicit scope, tests and
  evidence requirements.
- Added independent diff inspection, adversarial verification and reproduced
  test evidence before delegated work may advance a phase gate.

### 1.0.0 - 2026-07-10

- Replaced the monolithic plan with a composite document suite.
- Corrected the DOIN framing from proposed infrastructure to existing stable
  substrate.
- Added `heuristic-strategy` lifecycle policy role and Backtrader parity.
- Added component/stack champions, release channels and continuous optimization
  versus controlled deployment.
- Added shared contracts, generic domain metrics, LTS customer selection and
  privacy-safe usage attribution.
- Corrected active DOIN deployment to `doin-core`, unified `doin-node`, and
  `doin-plugins`; added two-level config generation and loader compatibility
  audit.

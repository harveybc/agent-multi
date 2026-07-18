# 05. DOIN Trading Domain Integration

## 1. Existing Substrate

DOIN already provides the hard distributed-systems layer:

- unified configurable nodes and per-domain roles;
- controlled flooding and production GossipSub options;
- TTL, deduplication, peer discovery and chain synchronization;
- commit-reveal and optimae lifecycle;
- deterministic evaluator seeds and synthetic-data hash consensus;
- quorum, incentives, reputation, resource limits and finality;
- Proof of Optimization block generation;
- island-model champion migration and startup champion synchronization;
- a proven shared-population mode: decentralized candidate claims, result
  propagation, deterministic generation reproduction and on-chain population
  snapshots;
- stage synchronization and candidate-evaluation messages;
- pull-based evaluator/inference tasks;
- local SQLite OLAP, chain metrics, PostgreSQL sync and dashboard.

These systems are dependencies, not implementation tasks in this plan.

The deployed repository set is `doin-core`, `doin-node`, and `doin-plugins`.
`doin-node` is the unified executable for optimizer, evaluator, inference and
relay roles. Standalone optimizer/evaluator repositories from earlier DOIN
iterations are historical references only.

All Ubuntu 26 workers use the canonical `trading-stack` environment described
in [`docs/environment/UBUNTU26_TRADING_STACK.md`](../environment/UBUNTU26_TRADING_STACK.md).
It pins the Python and package matrix, keeps TensorFlow and PyTorch CUDA
runtimes explicit, and installs the local repositories in editable mode. The
obsolete `tensorflow` Conda environment is not part of this work plan.

## 2. Existing Plugin Contracts

Trading uses the current entry-point groups:

```text
doin.optimization
doin.inference
doin.synthetic_data
```

The implementations satisfy existing `doin-core` interfaces:

- `OptimizationPlugin.configure(config)`;
- `OptimizationPlugin.optimize(current_best_params, current_best_performance)`;
- `OptimizationPlugin.get_domain_metadata()`;
- `InferencePlugin.configure(config)`;
- `InferencePlugin.evaluate(parameters, data)`;
- `SyntheticDataPlugin.configure(config)`;
- `SyntheticDataPlugin.generate(seed)`.

No consensus, flooding, Proof-of-Optimization or migration redesign is required
for the first trading vertical slice. One additive backward-compatible task
field, `metric_evidence`, carries the domain-generic metric vector through the
existing quorum path; legacy peers may omit it.

## 3. Trading Domains

Use role-specific domain IDs so metrics and champions remain meaningful:

- `trading-context-v1`;
- `trading-asset-policy-v1` with asset/timeframe in domain config/experiment
  dimensions;
- `trading-rush-v1`;
- `trading-lifecycle-v1`;
- `trading-risk-geometry-v1`;
- `trading-portfolio-v1`;
- `trading-stack-v1`.

Do not encode every hyperparameter in `domain_id`. Dataset/config hashes and
OLAP dimensions carry experiment identity. Domain IDs identify comparable
fitness semantics.

## 4. DOIN Adapters and Repository Ownership

`doin-node` is the coordination host, not the trading model or simulator. It
already pools optimization/evaluation/inference requests, records transactions
and metrics, and performs champion/parameter exchange. Trading enters through
its existing `doin.*` plugin interfaces without changing the node protocol.

The local-first rule is mandatory: every optimized component must be runnable
from its owning repository before it is exposed to DOIN. A DOIN adapter is an
integration layer, never the only implementation of an optimizer. For the
trading policy, `agent-multi --load_config ...` must run the same local
optimizer and candidate evaluation with DOIN absent; `doin-plugins` then wraps
that installed package for collaborative execution.

Ownership is deliberately split:

- `agent-multi` owns the trainable action-critic agent, experiment JSON, L1
  training, local optimizer plugin, multi-asset `gym-fx` calls, artifacts and
  direct inference library/CLI.
- `doin-plugins` owns thin adapters implementing the DOIN interfaces. They load
  the installed `agent-multi` package and delegate to its public contracts.
- `prediction_provider` owns inference serving for model artifacts when LTS or
  another client asks for a prediction/action request. It is not a training
  evaluator and is not replaced by the DOIN optimizer.
- `heuristic-strategy` remains the reference/scenario-verifier repository. Its
  Backtrader strategies, ideal predictor and configurable Gaussian-noise
  provider remain useful for baselines and robustness experiments, but they are
  not silently treated as the production action-critic policy.
- `doin-node` remains unchanged apart from JSON configuration and installed
  plugin packages. No standalone optimizer, evaluator or node process is
  introduced.

### 4.1 DOIN optimization adapter

The `doin.optimization` entry point in `doin-plugins` is a thin
`TradingOptimizer` adapter, analogous to the existing predictor adapter:

1. loads the canonical `agent-multi` experiment config and verifies its hash;
2. applies the DOIN candidate genome and repairs typed constraints;
3. invokes the public `agent-multi` candidate-training/evaluation contract and
   its local optimizer plugin;
4. keeps L1 stopping inside `agent-multi` and L2 candidate semantics in the
   configured optimizer component;
5. connects existing DOIN candidate/champion/stage callbacks;
6. returns the configured scalar fitness in the existing tuple contract and
   carries the metric vector, artifact hashes and config lineage through the
   existing candidate/champion callback and artifact paths;
7. accepts a compatible network champion only after schema, domain and data
   contract checks.

The adapter must not create a hidden second optimizer. If the selected
`agent-multi` component already owns a local DEAP/NEAT search, DOIN invokes
that component exactly as predictor does. Two explicit existing DOIN modes are
available:

- legacy island migration, where each node evolves a local population and
  accepts compatible network champions; and
- shared population, where `doin-node` owns one deterministic candidate queue,
  claims each candidate once, gathers results from peers, reproduces the next
  generation deterministically, and stores the population state on-chain.

The selected Phase 1 campaign is shared population. The trading adapter only
implements the same four local methods already used by the working TFT
predictor adapter: initialize local evaluation, create a serializable
population, evaluate one supplied genome, and reproduce from a fully observed
generation. It never replaces candidate claiming, flooding, chain persistence
or synchronization in `doin-node`. Island mode remains supported for historic
experiments but is not silently selected for the new campaign.

### 4.1.1 Selected shared-population campaign

`doin-node/examples/trading/phase_1_asset_policy_shared_v2/` is the first
clean trading configuration family using the established TFT shared protocol.
It intentionally leaves `phase_1_asset_policy/` v1 island data and chain
directories untouched as exploratory evidence.

All four v2 node JSON files share exactly these optimization semantics:

- domain `trading-asset-policy-solusdt-4h-sac-shared-v2`;
- `shared_population: true`, `shared_population_size: 20` and
  `population_size: 20`;
- the same canonical SOLUSDT 4h SAC experiment JSON, initial candidate,
  parameter bounds, metric direction and four-stage schedule;
- `optimization_resume: false`, because the v2 shared state is created and
  recovered from the new blockchain domain rather than from a v1 island
  checkpoint.

Only node identity, port, local data/OLAP directory, bootstrap routes, runtime
overlay and resource limits differ. Omega starts first and persists the genesis
population. Gamma and Dragon are then joined one at a time and must recover
that on-chain population before any production-scale run is accepted.

### 4.1.2 Post-learning checkpoint barrier and corrected fleet

Every off-policy L1 run must make at least one network update before a model is
eligible to become the saved best checkpoint. The default barrier is
`learning_starts + 1`; canonical Phase 1 SAC configs declare
`l1_min_checkpoint_timesteps: 5001` explicitly. Warm-up epochs neither save a
checkpoint nor consume L1 patience. A run that ends before crossing the barrier
fails instead of emitting an untrained champion artifact.

This rule was added after the BTCUSDT 1h v1 domain varied DEAP parameters but
returned the exact same fitness for 75 observed candidates. Its L1 epoch size
was 4000 while `learning_starts` was 5000, so epoch 1 saved the identical
seeded pre-learning policy for every candidate. The v1 BTC chain is retained as
diagnostic evidence and is not promotable.

The corrected remaining-asset campaign is
`examples/campaigns/phase_1_asset_policy_fleet_v4/` and uses fresh v2 domains
for BTCUSDT 1h, ADAUSDT 1h, EURUSD 4h and DOGEUSDT 4h. The completed SOLUSDT 4h
and SOLUSDT 1h artifacts remain valid: their stored models report 8000 training
timesteps with `learning_starts: 5000`, so both crossed the barrier. They remain
in immutable campaign history and are not repeated.

The ordered startup has one intentional runtime asymmetry. Omega, the declared
bootstrap worker, uses `shared_min_peers: 0` long enough to persist the
deterministic generation-zero population. Every joining worker retains
`shared_min_peers: 3`. The supervisor does not launch a joiner until the
bootstrap genesis, population block and fingerprint exist, preventing both the
startup deadlock and independent population chains.

### 4.2 DOIN inference/evaluator adapter

The `doin.inference` entry point is a separate `TradingInferencer`. When a DOIN
node claims a pooled inference request, it:

1. resolves the requested champion or parameter set and artifact hashes;
2. validates the requested data range and input/feature contract;
3. runs inference through the installed `agent-multi`/`prediction_provider`
   adapter without training;
4. returns the scalar verification metric required by the current
   `doin-core` `InferencePlugin` contract;
5. records the result through the existing DOIN transaction/OLAP path.

Rich action/asset/portfolio intents are a separate runtime API consumed by
`prediction_provider` and LTS. They can share the same loaded champion and
canonical config, but must not be smuggled into a scalar DOIN verification
result or interpreted as an order by `doin-node`.

This path is different from optimization fitness and must not mutate a
champion or consume future data. It is the path later used by LTS and clients
requesting inference over a specified date/data range.

### 4.2.1 Metric ownership

Metric ownership is split across three existing boundaries:

- `gym-fx` `metrics.plugins` computes raw simulator facts and derived trading
  values from the execution/analyzer result. The trading profile is
  `trading_metrics`; it extends `default_metrics` with drawdown fractions and
  risk-adjusted return fields.
- `agent-multi` selects the configured scalar objective from that summary for
  the local optimizer. Its metric helper is an objective selector, not a
  simulator metrics plugin.
- `doin-node` does not calculate trading metrics. It receives the scalar from
  the external DOIN optimizer/inference plugin, applies `higher_is_better` and
  tolerance, and persists reported/detail metrics through its existing chain
  and OLAP paths.

Therefore the same `gym-fx` metric plugin is used whether the local optimizer
is run directly or wrapped by DOIN. The DOIN node JSON declares the metric
identity and direction; it does not duplicate the formula.

### 4.3 Scenario verification adapter

The `doin.synthetic_data` entry point is a deterministic scenario-verification
adapter, not the client inference service. It reproduces the scenario-verifier
role previously exercised through `heuristic-strategy` and its prediction
providers. It generates or loads the same multi-asset scenario for a
commitment-derived seed and exposes the scenario hash to DOIN quorum checks.

The verifier may run a reference heuristic policy, the action-critic policy, or
both in a declared comparison arm. Ideal predictions and configurable
Gaussian noise remain explicit diagnostic providers; they cannot leak into
validation/test or be silently used as production inputs.

Call it `TradingScenarioSyntheticData` only at the DOIN boundary. Internally it
may delegate to a versioned `heuristic-strategy` or `gym-fx` scenario contract.

Initial generator ladder:

1. deterministic block/bootstrap scenarios preserving cross-asset alignment;
2. pre-trained regime/bootstrap generators loaded, never trained, by evaluators;
3. stress overlays for spread, gaps, volatility, correlation and liquidity;
4. later learned generators only after deterministic parity and hash consensus.

Synthetic verification is an anti-cheating/stress mechanism. Real chronological
validation-year fitness remains the research objective.

### 4.4 Request and metric separation

There are three distinct result classes and they must not be conflated:

| Path | Caller | Does training? | Main output |
| --- | --- | --- | --- |
| Optimization | DOIN optimizer role | Yes, through `agent-multi` L1/L2 contract | candidate params, scalar fitness, metric vector, artifact/config hashes |
| Inference | DOIN evaluator pool, LTS or client | No | prediction/action/intent and inference diagnostics |
| Scenario verification | DOIN quorum/evaluator path | No hidden fitting | deterministic scenario hash and verification metrics |

Optimization fitness uses the declared train/validation walk-forward contract;
protected test remains an offline promotion report. Inference metrics describe
the requested data and are not substituted for optimization fitness.

## 5. Candidate Payload

Parameters contain compact genome/config patches and immutable references:

- base config hash;
- typed genes;
- component/artifact hash or compact model payload policy;
- dataset/feature manifest hash;
- code/version bundle hash;
- metric schema version;
- deterministic seed metadata.

Large checkpoints and datasets live in the artifact store. On-chain payloads
contain hashes and retrieval descriptors subject to DOIN resource limits.

## 6. Configurable Metric Catalog

Each domain declares:

- primary scalar and direction;
- metric schema version;
- required raw metrics;
- coverage unit and minimum coverage;
- constraint metrics and failure values;
- display labels/units;
- aggregation formula;
- validation/test/live provenance rules.

Example portfolio metadata:

```json
{
  "performance_metric": "portfolio_validation_fitness",
  "higher_is_better": true,
  "metric_schema": "trading.portfolio.metrics.v1",
  "required_metrics": [
    "mean_weekly_return",
    "annual_return",
    "mean_weekly_rap",
    "annual_rap",
    "max_drawdown",
    "expected_shortfall",
    "turnover",
    "concentration",
    "coverage_weeks"
  ]
}
```

## 7. Staged Optimization

The stages below are a dependency order, not a menu of unrelated campaigns.
Higher layers consume immutable promoted artifacts from lower layers. They do
not silently retrain, replace, or reopen the full search space of those lower
layers.

### Stage 0: freeze evaluation semantics

Before spending distributed compute, freeze the cell universe, chronological
splits, weekly retraining contract, simulator/cost profile, metric schema,
coverage gate, deterministic seeds, and synthetic verification fixture. A
candidate cannot compensate for incomplete weeks, leakage, or incompatible
fitness semantics.

### Stage A: per-asset policy core

Optimize model and compact causal data choices while portfolio allocation is
fixed.

### Stage B: context representation

Optimize token groups, embeddings and encoder parameters around surviving
policies.

### Stage C: trade lifecycle and risk geometry

Optimize heuristic/learned lifecycle choice, thresholds, early close,
`rel_volume`, SL/TP and sizing mode.

### Stage D: rush detection and gating

Optimize labels, detector, calibration and exposure effects.

### Stage E: portfolio allocation

Optimize allocator, constraints, lookbacks, rebalance cadence and risk budgets
over frozen promoted asset cells.

### Stage F: restricted stack composition

Optimize compatible component references and a small cross-layer parameter set.
Never reopen the entire Cartesian search space.

Existing DOIN stage synchronization and champion migration remain in force.

### 7.1 Fleet scheduling policy

The default initial operating mode is **one logical distributed optimization
campaign at a time**:

- every available optimizer joins the same domain, experiment hash, metric
  version, and active optimization stage;
- omega, dragon, gamma 5070 Ti, and gamma 5090 are workers/evaluators in one
  DOIN campaign, not four unrelated searches;
- gamma may use two isolated `doin-node` processes so both GPUs participate in
  that same campaign;
- multiple local candidate processes are allowed when the machine overlay and
  measured RAM/VRAM envelope permit them;
- champion migration, stage synchronization, and OLAP evidence remain shared
  across the campaign;
- advancing to the next stage requires a promotion artifact and gate report,
  not merely the last candidate produced.

"One campaign" does not mean one operating-system process. DOIN remains a
distributed set of node/worker processes sharing one optimization objective.
This concentrates the full compute fleet on a sufficiently deep search and
makes throughput, evaluator disagreement, stalls, and metric anomalies visible
before launching another domain.

Independent asset campaigns may run concurrently only after the first
end-to-end campaign is stable and only when they have isolated domain/config
hashes, adequate evaluator coverage, and an explicit compute allocation. The
default is not to leave one machine optimizing a portfolio while another still
changes the asset policies on which that portfolio depends.

Campaign-to-campaign transition is implemented outside `doin-node` by the
replicated state machine in document 15. It consumes `converged` and blockchain
evidence from the existing unified-node APIs, archives the exact accepted model
artifact, proves all four worker processes stopped, and then derives the next
immutable job ordinal. It does not alter shared-population claiming,
reproduction, controlled flooding, or consensus.

### 7.2 Promotion and freezing order

1. Promote a small Pareto set per asset/timeframe from Stage A, not only one
   scalar winner.
2. Optimize context representation against those fixed policy recipes and
   promote only representations that improve causal validation evidence.
3. Optimize lifecycle and risk geometry with model/context artifacts frozen.
4. Train and optimize rush detection/gating against frozen base-policy outputs;
   retain a no-rush control.
5. Build the portfolio universe from promoted cell artifacts, then optimize
   allocation and rebalance behavior without reopening asset-level genomes.
6. Run only a narrow Stage F composition search over compatible promoted
   references and explicitly declared cross-layer genes.
7. Use L3/OLAP-guided proposals only after enough comparable L2 observations
   exist; L3 proposes regions but never bypasses L1/L2 evaluation.

If a higher stage exposes a lower-layer defect, reopen the smallest responsible
stage, mint new artifact/config hashes, and invalidate only dependent
promotions. Do not restart every layer automatically.

### 7.3 Weekly-retrained protected promotion execution

`examples/scripts/run_phase_1_weekly_promotion.py` is the local-first execution
path for a frozen Phase 1 candidate. It materializes one JSON configuration per
complete target week, applies only the frozen typed candidate parameters, uses
a rolling validation window ending before the target week, and enables the test
split only for that target-week report. It never sets `selection_uses_test`.

The runner writes resumable week configs/results, a compact SQLite OLAP store
and `promotion_summary.json`. The current Project 3 protocol freezes **48
contiguous Monday-aligned target weeks** before execution. A partial shard is
never labelled a complete protected test; it records its offset and coverage
and remains blocked until all 48 results are consolidated. The four nodes may
therefore evaluate disjoint 12-week shards without overlap. Annual return
compounds weekly returns; annual maximum drawdown is calculated from the
observed, concatenated 4-hour equity traces after rescaling each reset weekly
account onto one compounded path. Annual RAP subtracts that observed drawdown.
A missing trace is explicitly labelled as a lower-bound fallback and cannot
qualify a candidate for promotion.

When the 48-week run is sharded across machines, each worker writes its own
partial summary and SQLite OLAP store. `examples/scripts/consolidate_phase_1_weekly_promotion.py`
validates candidate/run identity, rejects conflicting duplicate weeks, remaps
the copied return traces, and writes the only summary that can be labelled a
complete protected evaluation.

The first fleet use assigns the three frozen Pareto recipes to separate workers
and uses the fourth worker for a deterministic cross-hardware replication. The
result remains a promotion report until all coverage, reproducibility,
compatibility and release gates pass.

## 8. Continuous Optimization and Weekly Release

DOIN optimizes continuously. Deployment changes on a controlled cadence:

1. observe current accepted/Pareto champions;
2. freeze a weekly training cutoff;
3. train/fine-tune candidate recipe artifacts;
4. rerun release validation and compatibility checks;
5. compare complete stack against current release;
6. publish a signed `DeploymentManifest` if gates pass;
7. let `prediction_provider` preload it;
8. let LTS switch at the configured rebalance boundary.

An accepted block does not directly change a live customer's model.

## 9. Champion Migration Compatibility

Before injecting a network champion, verify:

- same domain and metric major version;
- compatible base config and genome schema;
- matching data/input contract;
- required categorical mappings;
- parameter bounds and conditional genes;
- artifact framework compatibility.

Incompatible champions remain valid chain evidence but are not injected into
that local population.

### 9.1 Verified legacy artifact behavior and required evolution

The existing predictor implementation already provides the compatibility
baseline:

1. every isolated candidate is saved to a temporary `.keras` file;
2. when the candidate becomes a local champion, the optimizer reads and
   base64-encodes that exact trained model;
3. `doin-node` moves the model payload into optimae parameters;
4. commit/reveal binds the parameters and model bytes to the commitment;
5. `OPTIMAE_ACCEPTED` records those parameters on chain, allowing synchronized
   full nodes to recover and evaluate the champion.

This is functional but not the final storage architecture. Non-champion model
files are currently temporary, the HTTP transport caps a message at 100 MB,
every full node replicates the complete champion blob, and transaction-body
pruning can remove old embedded models because state snapshots do not retain
artifact references.

The trading-domain acceptance design therefore adds a decentralized
content-addressed artifact plane. The blockchain stores the signed immutable
descriptor and replication evidence, while a trackerless P2P backend transfers
and re-seeds bytes. A producing node is the first seed, never the sole required
seed. Champion acceptance requires a configured replication factor across
independent peers and a successful hash-verified fetch by evaluators. New nodes
must recover the current champion from any surviving provider.

Candidate models that are not champions may also be advertised by content hash.
They are retained selectively (top-K per generation, Pareto, anomaly,
reproducibility or explicit pin), avoiding permanent replication of every weak
candidate while preserving its metrics and artifact lineage in OLAP-on-chain.

## 10. Node Configuration

Reuse the existing DOIN JSON structure. Each machine config declares:

- a human-readable node label for decentralized monitoring;
- roles per trading domain;
- optimization/inference/synthetic plugins;
- base experiment config and parameter bounds;
- resource limits;
- seed offset;
- data/artifact roots;
- network/bootstrap settings;
- local OLAP path.

Machine-specific paths live in node config overlays, not canonical experiment
config. Code/version mismatch protection remains enabled.

Gamma may run independent workers on the 5070 Ti and 5090 when framework GPU
selection and host memory permit. If process-level GPU isolation is required,
they are separate `doin-node` unified-node instances with unique identity,
port, data directory, device, seed, resource limits and output paths. They are
not standalone `doin-optimizer` processes. The plan does not assume linear
speedup over Thunderbolt.

### 10.1 Decentralized network monitoring

Every unified node provides both its individual dashboard and the same
consolidated network view. No monitor service or privileged coordinator is
introduced. The monitoring contract is:

- `/api/monitor` returns a compact local snapshot containing node identity,
  label, uptime, chain height, active candidate, best domain performance,
  alerts, exact active-component revisions and all local `IP:port` aliases;
- `/api/network` groups LAN/Tailscale endpoints by discovery identity, fetches
  snapshots through the bounded transport pool, then merges them by the real
  peer identity in each response; it tries alternate routes before declaring a
  peer offline, removes an unresolved route only when an online peer claims
  that exact alias, and retains other discovered offline peers;
- the dashboard opens on a `Network` tab showing health, chain alignment,
  candidate progress, fitness, alert counts, revision compatibility and a link
  to each individual dashboard;
- recent alerts are aggregated with their originating node while local alert
  acknowledgement remains local;
- peer timeouts and partial failures produce an incomplete but usable view
  instead of blocking the dashboard.

Optimization throughput and provenance use three deliberately separate views:

- `Local total` is read from each adapter's append-only
  `optimization_candidate_history.csv`; it is the durable per-island count and
  survives node/process restarts;
- `Active run` is the optimizer's current resumed-segment counter and provides
  immediate progress before any transaction is mined;
- `Cand/h` is calculated independently by each island from the median of up to
  19 recent candidate-completion intervals in its durable history. Intervals
  that clearly span a reboot or pause are excluded; the dashboard reports `--`
  until at least two valid intervals exist;
- `Stage ETA` multiplies the active stage's planned remaining candidates by
  that median interval. The tooltip exposes the recent interquartile time
  range. It is not a promise for the whole optimization: L2 early stopping can
  finish the stage sooner, and a material workload/configuration change makes
  old timing evidence less representative;
- network throughput is the sum of the currently online islands' independently
  observed `Cand/h`; it is an operational capacity estimate, not a fitness or
  scientific-result metric;
- `On-chain` counts `candidate_evaluated` transactions by
  `transaction.peer_id`; it is globally verifiable but can lag local execution
  until pending transactions enter a block;
- these counters overlap and must never be added together;
- optimizer-thread callbacks use `sync_manager.our_height`, never a direct
  `ChainDB` query, because the SQLite chain connection belongs to the event
  loop thread; local CSV/OLAP persistence failures are emitted as warnings with
  tracebacks rather than being silently discarded;
- champion history contains only accepted running-best improvements;
- `transaction.peer_id` identifies the island that found a champion, while
  `block.generator_id` identifies the separate island that assembled its block;
- fitness, delta, validation/train-tail RAP, return and drawdown remain linked
  to the champion's immutable metric evidence.

The top performance box must name the configured metric and its period. For
Phase 1, `train_validation_l1_score` is displayed as `Best L2 fitness` with the
visible qualifier `Composite; not weekly or annual return`. It is a
return-fraction composite, not mean weekly return, annual return, mean weekly
RAP or annual RAP:

The dashboard's primary progress percentage is cumulative across the declared
candidate budget of **all** incremental stages. It does not reset when a later
stage begins. Stage number, generation, candidate number and an explicitly
labelled stage ETA remain visible as local operational detail. The percentage
and its planned campaign denominator do not attempt to predict L2 early
stopping; patience may finish the campaign sooner.

```text
L2 = mean(RAP(train_tail), RAP(validation))
     - beta * abs(RAP(validation) - RAP(train_tail))
```

Return and RAP period metrics remain separate evidence fields and must not be
silently relabeled as the L2 fitness.

Because every participant can produce this view, loss of Omega does not remove
operational visibility; Dragon or either Gamma island can become the operator's
entry point.

Block-height divergence shown by this view is actionable. For flooded block
announcements, synchronization must request the missing range from the direct
forwarder before considering the original author, because the author's LAN may
not be routable from the receiving island. Directed champion and evaluation
responses continue to prefer the original protocol identity.

The authoritative two-level configuration and generation procedure is defined
in [11 DOIN Configuration Profiles](11_DOIN_CONFIGURATION_PROFILES.md).

No trading optimization node is launched until the operator and Codex review
the canonical optimizer config, all machine overlays, all generated node
configs, their resolved hashes, and the planned stage/domain assignment.

## 11. Existing DOIN Changes Allowed

Only backward-compatible domain-general improvements are expected:

- generic arbitrary metric persistence in OLAP;
- artifact reference fields where current parameter payload is insufficient;
- dashboard rendering from domain metric metadata;
- decentralized consolidated node health, progress, version and alert views;
- resource accounting needed by long weekly walk-forward candidates.
- content-addressed artifact descriptors, provider discovery, replication
  proofs and hash-verified fetch without a central service.

Controlled flooding/GossipSub, Proof of Optimization, quorum, chain, migration,
reputation and incentive logic are out of scope unless a reproducible defect is
found.

## 12. Acceptance Criteria

- Trading plugins load through existing DOIN entry points.
- Three nodes exchange compatible champions using current migration flow.
- Independent evaluators reproduce performance within declared tolerance and
  agree on synthetic data hash.
- Candidate/champion metric vectors reach dashboard, local OLAP and chain
  metrics without being forced into MAE semantics.
- A new node syncs chain, recovers champion and joins optimization.
- A champion remains fetchable after its producing node is removed, and the
  acceptance gate rejects insufficient replication or a hash mismatch.
- Code/version mismatch is rejected as today.
- Every live node renders the same participant set and revision compatibility;
  an unavailable participant remains visible as offline without freezing the
  remaining dashboard.
- No trading plugin requires modifications to controlled flooding or Proof of
  Optimization.
- A promoted weekly deployment references an accepted DOIN candidate and a
  reproducible fitted artifact without treating acceptance as auto-deployment.

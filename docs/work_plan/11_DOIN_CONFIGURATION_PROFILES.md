# 11. DOIN Configuration Profiles

## 1. Active Runtime Architecture

The active DOIN deployment uses three repositories:

| Repository | Active responsibility |
| --- | --- |
| `doin-core` | Shared protocol models, plugin ABCs, crypto, consensus and chain primitives |
| `doin-node` | Unified node process: optimize, evaluate, infer, relay, chain, tasks, dashboard and OLAP |
| `doin-plugins` | Domain-specific optimization, inference and deterministic synthetic-data adapters |

Earlier standalone node/optimizer/evaluator repositories are not required by
the trading deployment. One `doin-node` process can enable both `optimize` and
`evaluate` for one or more domains.

The trading plugin chain follows the proven predictor pattern:

```text
doin-node doin.optimization
    -> doin-plugins TradingOptimizer adapter
        -> installed agent-multi optimizer/agent/environment contract

doin-node doin.inference
    -> doin-plugins TradingInferencer adapter
        -> agent-multi or prediction_provider inference contract

doin-node doin.synthetic_data
    -> doin-plugins TradingScenarioSyntheticData adapter (`trading_scenario`)
        -> versioned heuristic-strategy/gym-fx scenario verifier
```

The adapters are external plugins selected by JSON. `doin-node` is not changed
to contain trading logic, and the three paths remain separate: optimization may
train, inference never trains, and scenario verification never hides fitting.

`trading_asset` is the entry-point name for both the optimization and DOIN
verification adapters. The local agent-multi optimizer remains selected by the
experiment's `optimizer_plugin` (for example `default_optimizer`) and can be
run without DOIN. The entry-point name is not a replacement for that local
plugin; it is the network adapter around it.

The simulator metric plugin is a separate `gym-fx` entry point:

```text
agent-multi environment.metrics_plugin = trading_metrics
    -> gym-fx metrics.plugins
        -> raw execution facts + risk-adjusted summary
```

`doin-node` only receives the resulting scalar/detail metrics and uses its
domain `higher_is_better` and tolerance settings for network comparison.

## 2. Existing Configuration Pattern

The proven predictor deployment has two conceptual JSON levels.

### 2.1 Common optimized-repository configuration

Lives in the repository being optimized, for example:

- `predictor/examples/config/phase_1c_direction/optimization/phase_1c_direction_tcn_direction_long_1d_optimization_config.json`
- `predictor/examples/config/phase_1c_direction/optimization/phase_1c_direction_tcn_direction_short_1d_optimization_config.json`

It defines the reproducible domain experiment:

- data files and splits;
- predictor/preprocessor/target/pipeline plugins;
- feature and target behavior;
- architecture/training defaults;
- L1 early stopping;
- DEAP/NEAT L2 settings;
- staged optimization and parameter bounds;
- deterministic seed policy;
- output/artifact locations.

The implemented trading equivalent belongs in
`agent-multi/examples/config/phase_1_asset_policy/` and is the same on every
machine, modulo validated runtime-root resolution. The older
`examples/config/doin/` files remain bounded vertical-slice regression fixtures;
they are not the campaign configuration.

### 2.2 Per-machine unified-node configuration

Lives in `doin-node/examples/`, for example:

- `doin-node/examples/predictor_omega_node_tft_binary_neat.json`
- `doin-node/examples/predictor_gamma_node_tft_binary_neat.json`

It defines:

- bind host/port and node data directory;
- bootstrap peers and network protocol;
- block/quorum/acceptance timing;
- discovery, storage and deterministic verification;
- per-domain `optimize`/`evaluate` roles;
- optimization/inference/synthetic plugin names;
- path/reference to the common repository config;
- machine seed offset;
- resource limits;
- local stats/OLAP paths;
- device and artifact/data-root overlay for the trading deployment.

## 3. Required Trading Configuration Layout

### 3.1 Canonical experiment config

Implemented Phase 1 pattern and future sibling phases:

```text
agent-multi/examples/
  config/phase_1_asset_policy/
    phase_1_asset_policy_<asset>_<timeframe>_<policy>_config.json
    optimization/*_optimization_config.json
    inference/*_inference_config.json
  data/phase_1_asset_policy/
    <asset>_<timeframe>_dataset_manifest.json
  results/phase_1_asset_policy/
    README.md
  scripts/
    validate_phase_1_asset_policy.py
    run_phase_1_asset_policy_local.sh
```

This config owns model/data/policy semantics and optimization stages. It is
canonicalized and hashed. It does not contain hostname, IP, GPU ID, credentials
or machine-specific GPU selection. The standalone optimization config includes
explicit split dates, plugins, L1/L2 stopping, executable bounds, stage-local
active parameters, resume/history/statistics/checkpoint paths and the protected
test firewall, matching the substantive predictor contract rather than merely
its directory names.

### 3.2 Machine overlay

Proposed source overlays:

```text
doin-node/examples/trading/
  machines/
    omega.json
    dragon.json
    gamma_5070ti.json
    gamma_5090.json
  generated/
    <experiment>_omega_node.json
    <experiment>_dragon_node.json
    <experiment>_gamma_5070ti_node.json
    <experiment>_gamma_5090_node.json
```

The source overlay contains only machine/runtime fields. Generated node configs
conform exactly to the current `doin-node` JSON schema.

### 3.3 Generated full node config

Conceptual shape:

```json
{
  "$doc": "Generated unified DOIN node config",
  "host": "0.0.0.0",
  "port": 8470,
  "data_dir": "./doin-data-trading-omega",
  "bootstrap_peers": [],
  "network_protocol": "flooding",
  "require_deterministic_seed": true,
  "domains": [
    {
      "domain_id": "trading-asset-policy-v1",
      "optimize": true,
      "evaluate": true,
      "optimization_plugin": "trading_asset",
      "inference_plugin": "trading_asset",
      "synthetic_data_plugin": "trading_scenario",
      "has_synthetic_data": true,
      "optimization_config": {
        "agent_multi_root": "${AGENT_MULTI_ROOT}",
        "load_config": "examples/config/phase_1_asset_policy/optimization/phase_1_asset_policy_solusdt_4h_sac_optimization_config.json",
        "node_seed_offset": 0,
        "device": "cuda:0",
        "artifact_root": "${TRADING_ARTIFACT_ROOT}"
      },
      "param_bounds": {},
      "resource_limits": {},
      "higher_is_better": true
    }
  ],
  "experiment_stats_file": "./trading_stats_omega.csv",
  "olap_db_path": "./doin-data-trading-omega/olap.db"
}
```

Environment substitution is performed before node startup or by a validated
config generator. Raw `${...}` placeholders are not passed unknowingly to the
current loader.

## 4. Configuration Ownership Matrix

| Field family | Canonical owner | Machine override allowed |
| --- | --- | --- |
| Dataset/splits/features | `agent-multi` experiment config | No, except resolved root |
| Model/policy plugins | `agent-multi` experiment config | No |
| Architecture/training | `agent-multi` experiment config | No, except resource-safe batch cap if declared |
| Optimization stages/bounds | `agent-multi` experiment config | No |
| Metric schema/fitness | `agent-multi` experiment config and DOIN domain metadata | No |
| DOIN role plugins | generated node config | No between peers for same domain |
| Seed offset | machine overlay | Yes, unique |
| Device/GPU | machine overlay | Yes |
| Resource limits | machine overlay constrained by experiment maximum | Yes |
| Host/port/peers/data dir | machine overlay | Yes |
| Network/quorum/security | network deployment profile | Same policy across network |
| Credentials | environment/secret store | Never canonicalized or committed |

## 5. Generation Instead of Manual Duplication

Current predictor node examples duplicate common model and optimization fields
inside each machine JSON. For trading, generate full node files from:

```text
canonical experiment config
+ network deployment profile
+ machine overlay
= validated doin-node config
```

The generator must:

1. load and validate all sources;
2. reject conflicting experiment semantics;
3. materialize fields required by current `doin-node`;
4. calculate `experiment_config_hash` excluding machine-only values;
5. calculate full `node_config_hash`;
6. verify unique node identity/port/data directory/seed offset;
7. write a generated manifest identifying all source hashes;
8. support `--check` mode for CI without rewriting files.

All nodes in one domain must report the same experiment hash even though node
hashes differ.

## 6. Current Loader Compatibility Audit

The current `doin-node` loader materializes `DomainRole` fields for:

- `domain_id`, `optimize`, `evaluate`;
- `optimization_plugin`, `optimization_config`, `param_bounds`;
- `inference_plugin`, `synthetic_data_plugin`, `has_synthetic_data`;
- `synthetic_data_validation`, `higher_is_better`;
- `resource_limits`, `target_performance`.

Some existing example JSON files also contain `inference_config`,
`synthetic_data_config`, `metric_type`, and `incentive_config`. In the currently
inspected loader, these fields are not all copied into `DomainRole`; plugin
setup configures inference and synthetic plugins from `optimization_config`.

This is configuration debt, not a reason to split the unified node. Before the
trading domain is launched, choose and test one backward-compatible rule:

1. make `DomainRole` explicitly support separate optimization, inference and
   synthetic configs; or
2. retain one authoritative `domain_config`/`optimization_config` consumed by
   all three plugins and remove misleading unused keys from generated files.

Recommendation: use one canonical domain config with optional named subtrees
(`optimization`, `inference`, `synthetic`) and make the loader/plugin setup
consume them explicitly. Preserve old `optimization_config` behavior for
existing predictor deployments.

CI must fail on unknown or ignored non-documentation fields so a valid-looking
node config cannot silently lose behavior.

## 7. Machine Profiles

### 7.1 Omega

- one unified `doin-node` instance;
- optimization/evaluation roles according to available GPU and host state;
- unique seed offset, identity, port and data directory;
- can act as bootstrap/supervisor but no trading-domain correctness depends on
  one permanent central machine.

### 7.2 Dragon

- one unified node on current reachable LAN/Tailscale address;
- RTX 4090 explicit device;
- WiFi latency affects peer transport estimates, not candidate fitness;
- local artifacts/OLAP with chain sync and backup.

### 7.3 Gamma 5070 Ti and RTX 5090

Preferred initial layout: two unified-node processes when both GPUs run
concurrently:

- unique persisted identity files;
- unique ports and data directories;
- `CUDA_VISIBLE_DEVICES` or explicit framework device isolation;
- distinct node seed offsets;
- no shared writable candidate/output path;
- host RAM and disk limits sized for simultaneous candidates;
- same canonical experiment hash.

A later single-process multi-GPU scheduler is optional and must demonstrate
better reliability/throughput before replacing process isolation.

## 8. Domain Config Example Mapping

Existing predictor pattern maps to trading as follows:

| Predictor field | Trading equivalent |
| --- | --- |
| `predictor_root` | `agent_multi_root` |
| `load_config` | canonical trading experiment config |
| `predictor_plugin` | `asset_policy`/component plugin in nested config |
| `preprocessor_plugin` | data/representation plugin set |
| `target_plugin` | reward/label/objective contract |
| `pipeline_plugin` | weekly asset/portfolio pipeline |
| `optimizer_plugin` | local DEAP/NEAT adapter wrapped by DOIN |
| `optimization_stages` | component-specific staged genome |
| `param_bounds` | generated from canonical typed search space |
| `early_patience` | L1 candidate training patience |
| `optimization_patience` | L2 stage/generation patience |
| `node_seed_offset` | per unified-node machine/GPU overlay |

## 9. Startup Validation

### 9.1 Mandatory operator review gate

Before the first trading-domain campaign, stop and review these artifacts with
the user:

1. the common canonical optimizer/experiment JSON from `agent-multi`;
2. the network deployment profile;
3. machine overlays for omega, dragon, gamma 5070 Ti, and gamma 5090;
4. every generated full `doin-node` JSON;
5. the allowlisted semantic/runtime diff between generated files;
6. experiment and node hashes, plugin entry points, bounds, L1/L2 stopping
   rules, resource limits, devices, ports, identities, and output paths;
7. the exact active stage, domain, promoted dependency artifacts, and fitness
   metric;
8. the launch, health-check, pause, checkpoint, and rollback commands.

No node start is implied by config generation or validation. Launch requires
explicit operator approval after this review.

Before a unified node joins a trading domain:

- validate node/config schemas;
- confirm active repositories and exact commit/version bundle;
- confirm plugin entry points;
- resolve common config and data/artifact roots;
- verify experiment hash matches peers/domain deployment;
- verify unique node identity, port, data directory and seed offset;
- verify GPU selection and framework visibility;
- verify synthetic generator/artifacts and deterministic hash fixture;
- verify write space, host memory and resource limits;
- verify chain/OLAP state and reset/sync intent;
- print all effective roles and config hashes.

No optimizer starts if any required field is ignored, unresolved or mismatched.

## 10. Acceptance Criteria

- Trading uses only `doin-core`, `doin-node` and `doin-plugins` from the DOIN
  repository family.
- One unified node performs optimization and evaluation roles in smoke tests.
- Common experiment semantics are stored once and hash-identical across nodes.
- Per-machine configs differ only in allowed runtime fields.
- Generated node configs load under the existing unified-node CLI.
- Existing predictor example configs remain backward compatible.
- Unknown/ignored functional keys fail validation.
- Gamma dual-GPU instances have isolated identity, ports, devices, outputs and
  seed offsets.
- Three-machine startup reports matching experiment hash and healthy peer/chain
  synchronization.

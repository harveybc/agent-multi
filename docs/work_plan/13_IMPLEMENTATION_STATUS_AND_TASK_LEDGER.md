# 13. Implementation Status and Task Ledger

Status timestamp: 2026-07-12
Plan version: 1.3.0
Current focus: verified local trading vertical and publication gate

## 1. Phase Summary

| Phase | Status | Evidence |
| --- | --- | --- |
| Phase 0: contracts and evidence | Implemented and published | Contracts, schemas, metric catalog, shortlist and compatibility gate implemented; `trading-contracts` commit `4675c8f` published |
| Phase 1: configuration and lineage | Implemented locally; cross-machine gate pending | Canonical resolver, runtime overlays and Git lineage implemented in `agent-multi` |
| Phase 2: heuristic lifecycle extraction | Verified locally | Pure policy, source substitution, packaging and frozen Backtrader requested-action replay pass |
| Phase 3 | Engine selected; vertical slice verified | NautilusTrader 1.230.0 multi-asset replay, costs, margin preflight, rollover, canonical reports and Gym bridge pass; portfolio-native Gym expansion remains |
| Phase 4 and later | Not started | Depends on the Phase 3 multi-asset environment and config review gates |

A bounded local optimization and independent inference verification were run on
Omega. No DOIN node, daemon, machine service, Dragon or Gamma process was
started.

Current operator state reported on 2026-07-11: `dragon` and `gamma` are powered
off intentionally during design. This is not an outage. Do not wake, deploy to,
or report them as failed until the user approves the DOIN configuration review
and launch gate.

## 2. Verified Local Deliverables

### 2.1 `trading-contracts` 0.1.0 foundation

Local repository:

```text
/home/harveybc/Documents/GitHub/trading-contracts
```

Implemented:

- strict Pydantic v2 contract base with unknown-field rejection;
- timezone-aware persisted object semantics;
- canonical asset and cell identity;
- market snapshot and variable-length context token contracts;
- prediction, decision context, asset intent and portfolio intent contracts;
- order intent and execution report contracts;
- component and deployment manifests;
- canonical sorted JSON and SHA-256 content hashes;
- canonical `trading_experiment.v1` configuration;
- typed `candidate_genome_patch.v1` with JSON Pointer keys;
- metric catalog contract and versioned initial catalog;
- JSON Schema exporter and generated schemas;
- representative experiment and candidate-patch fixtures.

Verification:

```text
python -m pytest -q
24 passed
```

The package is installed locally in editable mode for integration testing and
published at `https://github.com/harveybc/trading-contracts`, commit `4675c8f`.

### 2.7 Local-first DOIN trading boundary

Implemented locally, not yet deployed to any machine:

- `agent-multi/app/metrics.py` provides an explicit metric seam while
  preserving the historical agent fitness default;
- `agent-multi/optimizer_plugins/default_optimizer.py` accepts a typed seed
  candidate, receives migrated candidates between generations, and emits
  DOIN-compatible champion/stage/candidate callbacks;
- `doin-plugins.trading.TradingOptimizer` wraps that local optimizer through the
  established external entry-point pattern;
- `doin-plugins.trading.TradingInferencer` performs inference-only scalar
  verification and does not claim to be the rich LTS action-serving API;
- `doin-plugins.trading.TradingScenarioSyntheticData` provides a deterministic
  `fixture_v1` scenario with hash evidence. Learned or heuristic delegated
  scenario backends remain a later promotion gate;
- `gym-fx/metrics_plugins/trading_metrics.py` is the actual simulator metrics
  plugin. It extends `default_metrics` with unit-safe drawdown/RAP fields and
  only annualizes when the elapsed evaluation period is explicitly supplied;
- entry points are `trading_asset` for optimization/evaluation and
  `trading_scenario` for synthetic verification.
- `examples/config/doin/trading_asset_solusdt_4h_sac_v1.json` remains the first
  portable vertical-slice regression seed. The actual incremental campaign is
  now under `examples/config/phase_1_asset_policy/`, with matching `data`,
  `results` and `scripts` directories modeled on predictor's phase layout.
- the local optimizer now executes declared stage-specific bounds, freezes
  inactive parameters, carries its champion between stages, persists atomic
  resume/statistics/parameter files, appends candidate history and preserves
  the exact optimizer checkpoint separately from the final retrain;
- Phase 1 uses independent L1 and L2 semantics: L1 monitors risk-adjusted
  train-tail/validation behavior; L2 maximizes the gap-penalized
  `train_validation_l1_score`;
- candidate optimization sets `evaluate_test_split=false`; a focused firewall
  test proves that `_final_eval` never opens the protected test path.

Contract verification before the vertical smoke:

```text
doin-plugins/tests/test_trading_plugins.py: 8 passed
gym-fx trading metrics + engine tests: 5 passed
agent-multi targeted optimizer/metric tests: 4 passed
doin-core full suite: 278 passed
doin-node metric/config/OLAP suite: 63 passed
agent-multi --load_config ... --agent_plugin random_agent --total_timesteps 0: exit 0
```

The bounded SOLUSDT 4h smoke used real Project 3 input, two candidates, one
generation and 64 training steps. The CLI wrote the resolved config, manifest,
result, final policy and optimizer output. The exact optimizer champion was
194,095 bytes with SHA-256
`47601eee7a6ae7340006f40cc80afaf144ccfe488211355229ed20b5f38c24ee`;
`TradingInferencer` decoded that base64 artifact and reproduced fitness `0.0`.

A second invocation through the external `TradingOptimizer` contract returned
fitness `0.004052591711230477`, 49 evidence fields and a 194,096-byte model;
the decoded SHA-256 exactly matched `model_artifact_sha256`. This run exposed
and fixed a real callback-copy defect: bound DOIN callbacks contain locks and
must be shallow-copied as process-local runtime objects rather than deep-copied
as declarative configuration.

Both values are wiring evidence only. The first policy made no trades and was
diagnosed as `policy_hold_collapse`; neither smoke value is a scientific model
result, annual result, shortlist update or promotion candidate.

On 2026-07-12 the predictor-style Phase 1 smoke executed locally on Omega with
the real hash-verified SOLUSDT 4h source. It evaluated two candidates, preserved
an exact 194,031-byte optimizer champion, wrote all resume/history/statistics
artifacts, and obtained L2 wiring fitness `0.0060425180751335905`. Validation
return was `0.02396380163676559` and validation risk-adjusted return was
`0.006159046907203151`; the protected test summary is explicitly marked
`evaluation_skipped`. Because the smoke caps each environment at 384 rows and
trains for one 64-step epoch, these values are execution evidence only.

The adapters were not launched as DOIN nodes on any machine. No generated node
config has been accepted for deployment yet.

### 2.2 Canonical configuration in `agent-multi`

Implemented:

- `--config` alias for `--load_config`;
- optional base profile and DOIN candidate patch;
- precedence: defaults, base, file, CLI, candidate patch;
- canonical nested config validation;
- exact `base_config_hash` enforcement before candidate application;
- JSON Pointer patching only for existing paths;
- deterministic canonical hash and resolution manifest;
- atomic canonical config and manifest output;
- compatibility flattening for current plugins;
- preservation of every key in legacy flat configs;
- embedded-secret rejection;
- explicit namespacing for generic subtrees such as rush, OLAP and portfolio.

Compatibility evidence:

- the real oracle behavior smoke config preserved all 100 source keys with no
  missing or changed runtime values;
- `agent-multi` accepted `--config`, applied a CLI override and wrote canonical
  lineage in an isolated test;
- a canonical nested smoke config resolved to the intended agent, lifecycle,
  optimization, rush and OLAP settings.

Verification:

```text
python -m pytest -q tests/unit
152 passed
```

The suite also exposed and fixed an evidence-only precision defect in the event
token transformer's normalization manifest. Forward computation remains
`float32`; original normalization evidence is retained in `float64`.

### 2.5 Machine-local runtime overlays and code lineage

Implemented:

- separate `trading_runtime_overlay.v1` contract and hash;
- recursive resolution of declared `${NAME_ROOT}` placeholders only;
- fail-closed behavior for missing runtime roots;
- immutable canonical experiment config while runtime paths are materialized;
- independent device, resource, root and environment-reference overlays;
- Git snapshots with repository root, commit, branch, dirty state, status hash,
  bounded status sample and tracked diff hash;
- expected-commit comparison when the canonical config declares one;
- plugin defaults cannot overwrite a loaded config value.

Versioned overlays:

```text
configs/runtime/omega.json
configs/runtime/dragon.json
configs/runtime/gamma_5070ti.json
configs/runtime/gamma_5090.json
```

Gamma's two profiles use separate artifact/model/cache roots. The 5070 Ti
profile selects `cuda:0`; the 5090 eGPU profile selects `cuda:1`, matching the
observed enumeration at specification time. Runtime preflight must still verify
device enumeration before launching work.

### 2.6 Component and deployment compatibility gate

Implemented in `trading-contracts`:

- major contract-family compatibility checks;
- required component-role checks;
- component map key versus manifest-role checks;
- component training cutoff versus deployment cutoff firewall;
- content-hash binding between deployment and component manifests;
- producer-output and consumer-input contract-edge checks;
- execution-contract compatibility;
- fail-closed issues for malformed contract versions;
- deterministic example builder for component, portfolio intent, order intent
  and deployment manifest fixtures.

The gate returns a structured issue report and does not load models, touch
brokers or mutate manifests.

### 2.7 Pure heuristic lifecycle policy increment

Implemented in `heuristic-strategy`:

- dependency-free-from-Backtrader policy module for prediction entry/exit;
- typed and validated policy parameters;
- pure forecast-path entry geometry;
- exact A-G prediction early-close variants;
- legacy reward/risk order-size interpolation and cash cap;
- `DecisionContext -> AssetIntent` policy with no broker units;
- typed optimizable parameter schema;
- Backtrader long/short adapter delegates entry, exit and sizing math to the
  same pure functions;
- frozen JSON decision fixture and adapter delegation regression.
- normalized ideal/mapped, CSV, direct-model, and provider callback adapters;
- source-independent `PredictionBundle` materialization;
- frozen Backtrader requested-action replay derived from the base commit.

Focused verification:

```text
python -m pytest -q \
  tests/unit_tests/test_prediction_entry_exit_policy.py \
  tests/unit_tests/test_prediction_entry_exit_backtrader_replay.py \
  tests/unit_tests/test_prediction_source_substitution.py
17 passed
```

Wheel inspection confirms that the pure policy modules and
`trade_lifecycle_policy.plugins/prediction_entry_exit_v1` entry point are
packaged. Importing `app.policies` does not load Backtrader. Acceptance details:

```text
docs/handoffs/CODEX_HEURISTIC_LIFECYCLE_ACCEPTANCE_2026_07_11.md
```

The historical full suite is not a valid green gate at the starting commit. It
fails during collection on removed functions/modules and on a nested
`timeseries-gan/tests` package collision. The existing prediction-client test
also has five failures because it calls a removed generic `get_prediction()`
API instead of the current entry/exit methods. These baseline failures were
observed before Phase 2 edits and are not hidden or counted as passing.

### 2.3 Metric catalog v1

Source:

```text
trading-contracts/examples/metric_catalog_v1.json
```

The initial catalog includes explicit definitions for:

- coverage weeks;
- mean weekly return;
- compounded annual return;
- mean weekly RAP;
- additive annual RAP;
- mean weekly and full-path drawdown;
- weekly CVaR20;
- logical trade count;
- portfolio turnover;
- L1 gap-penalized score.

Each metric declares unit, period, aggregation, direction, formula,
denominator, coverage rule and failure semantics. Partial evidence cannot carry
an annual label.

### 2.4 Project 3 shortlist import

Generator:

```text
agent-multi/tools/import_project3_shortlist.py
```

Generated manifest:

```text
agent-multi/configs/manifests/project3_doin_shortlist_2026_07_10.json
```

Imported evidence:

- 14 candidates total;
- 5 active short-horizon seeds;
- 6 near-full-year long-horizon seeds;
- 3 partial rush/opportunity seeds;
- exact source file hashes and `financial-data` commit;
- canonical asset/cell identities;
- exact coverage and evidence-scope labels.

The manifest declares these rows as research seeds only. Imported test evidence
cannot select, early-stop, optimize or promote future candidates.

## 3. Delegation Ledger

| Task ID | Owner | Repository | Status | Scope | Acceptance owner |
| --- | --- | --- | --- | --- | --- |
| `CONTRACTS-001` | Codex | `trading-contracts` | verified_local | Contract/schema foundation | Codex |
| `CONFIG-001` | Codex | `agent-multi` | verified_local | Canonical config and legacy translation | Codex |
| `METRICS-001` | Codex | `trading-contracts` | verified_local | Metric catalog v1 | Codex |
| `SHORTLIST-001` | Codex | `agent-multi` | verified_local | Hash-verified Project 3 import | Codex |
| `RUNTIME-001` | Codex | `agent-multi`, `trading-contracts` | verified_local | Machine overlays, path resolution and Git lineage | Codex |
| `COMPAT-001` | Codex | `trading-contracts` | verified_local | Component/deployment contract and hash gate | Codex |
| `HEURISTIC-001` | Codex | `heuristic-strategy` | verified_local | Pure prediction entry/exit lifecycle extraction | Codex |
| `DOIN-CONFIG-001` | Claude + Codex review corrections | `doin-node` | verified_local | Complete unified-node config materialization | Codex |
| `SIM-ENGINE-001` | Codex | `gym-fx`, `agent-multi` | verified_local | Nautilus engine bake-off, cost profiles and canonical execution reports | Codex |
| `SIM-GYM-001` | Codex | `gym-fx`, `agent-multi` | verified_single_cell | JSON-selectable Nautilus Gym compatibility bridge | Codex |
| `DOIN-TRADING-001` | Codex | `agent-multi`, `doin-plugins`, `doin-core`, `doin-node` | verified_local_vertical | Local optimizer, exact champion artifact, generic metric evidence, external adapter and independent inference | Codex |
| `ARTIFACT-P2P-001` | Codex | DOIN stack | designed_not_implemented | Content-addressed descriptor, trackerless transfer and multi-peer replication gate | Codex |

Claude packet:

```text
docs/handoffs/CLAUDE_DOIN_NODE_CONFIG_MATERIALIZATION_TASK_2026_07_10.md
```

Claude's implementation was independently reviewed against `R01` through
`R15`. Codex corrected nested mutable-config leakage and silent unknown
`ResourceLimits` fields, then reproduced focused and full-suite evidence. The
review record is:

```text
docs/handoffs/CODEX_REVIEW_DOIN_NODE_CONFIG_MATERIALIZATION_2026_07_11.md
```

## 4. Immediate Next Tasks

1. Finish full-suite, clean-repository and GitHub publication gates for the
   modified repositories.
2. Preserve and hash the Project 3 OLAP evidence before removing any redundant
   generated output.
3. Complete Phase 0/1 gates on a second machine path only after code/config
   review and explicit startup approval.
4. Expand the verified Nautilus single-cell Gym bridge into the portfolio-native
   multi-asset observation/action contract without creating account state
   outside Nautilus.
5. Implement and fault-test the decentralized artifact plane before a
   multi-node trading-domain acceptance run.
6. When Phase 8 is reached, stop for the mandatory user review of the common
   optimizer JSON and all per-machine/generated DOIN node JSONs before launch.

## 5. Current Risks

- The content-addressed P2P artifact plane is designed but not implemented;
  the first vertical retains the proven base64 champion-in-parameters path.
- The `agent-multi` repository contains a large generated `experiments/`
  directory; only reviewed source/config/documentation files may be committed.
- Current canonical sections intentionally allow plugin-specific dictionaries;
  role-specific typed submodels will be added only after the first vertical
  slice freezes their real fields.
- `doin-node` still has three independently confirmed baseline test failures:
  one GossipSub mesh-capacity failure and two stale VUW zero-weight assertions.
- Two historical `doin-plugins` network modules target the retired separate
  evaluator service. They remain in the repository but are explicitly skipped
  when that legacy package is absent; active network coverage belongs to the
  unified `doin-node` suite.
- `heuristic-strategy` has pre-existing stale tests and an embedded repository
  test-discovery collision. The Phase 2 policy gate is independently green,
  while repository-wide historical test cleanup remains explicit debt.
- NautilusTrader 1.230.0 does not natively pre-deny insufficient-margin orders
  in its stable Python margin-account risk path. The mandatory adapter preflight
  calls Nautilus's own margin calculator; this guard must remain tested until an
  upstream replacement passes the same fixture.
- The Nautilus replay adapter is multi-asset, but the interactive Gym bridge is
  currently the verified single-cell compatibility slice. It is not yet the
  complete portfolio training environment.

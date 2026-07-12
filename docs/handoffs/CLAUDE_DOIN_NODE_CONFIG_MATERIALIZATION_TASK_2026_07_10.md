# Claude Task: Complete DOIN Unified-Node Config Materialization

## Mandatory Agent Role and Operating Mode

You are acting as a **senior machine-learning systems engineer, senior Python
software engineer, distributed-systems engineer, configuration-management
specialist and test engineer**. You have practical experience with:

- reproducible ML training and evolutionary optimization;
- deterministic evaluation and synthetic-data verification;
- distributed worker/evaluator architectures;
- Python dataclasses and typed configuration loaders;
- plugin systems based on Python entry points;
- backward-compatible configuration migrations;
- defensive validation and failure isolation;
- regression, property and integration testing.

For this task, behave as a conservative maintainer of a working decentralized
optimization system. Do not behave as a product designer, research ideator or
architecture owner. The architecture and semantics below are already decided.
Your job is to implement them exactly, prove them with tests and report any
contradiction instead of inventing a resolution.

Use the following priorities, in order:

1. preserve DOIN protocol and optimization behavior;
2. ensure JSON values actually reach the runtime objects/plugins they configure;
3. preserve deterministic ML and evaluator semantics;
4. preserve backward compatibility with existing node JSONs;
5. fail clearly on malformed nested configuration;
6. keep the diff minimal and reviewable;
7. improve style only where required to implement these behaviors.

You are not authorized to reinterpret the request, redesign adjacent code,
perform cleanup refactors, modify examples, or implement anything outside the
explicit scope. If the repository state or source contradicts this
specification, stop and report the exact conflict with file and line evidence.
Do not guess.

This file is the complete task prompt. Read it fully before executing any
command. Do not stop after an analysis or proposal when the preflight passes:
implement the code and tests end to end.

## Task Header

- Task ID: `DOIN-CONFIG-001`
- Priority: high
- Owning phase: Adaptive Multi-Asset Trading work plan, Phase 1 foundation and
  Phase 8 prerequisite
- Repository: `/home/harveybc/Documents/GitHub/doin-node`
- Branch at specification time: `master`
- Required base commit: `826c713dda29bd3f27513b647e979b73c5505dd9`
- Codex review status: implementation not yet accepted

## Objective

Correct the `doin-node` JSON loader and plugin setup so every supported
`UnifiedNodeConfig` and `DomainRole` setting is materialized rather than
silently replaced by a dataclass default, and so optimization, inference and
synthetic-data plugins receive their own configuration subtrees.

This is a bounded compatibility fix. Do not redesign DOIN, its network,
consensus, plugin interfaces, chain, OLAP schema or examples.

## System and Business Context

### Active repository topology

The active DOIN implementation consists of:

- `doin-core`: stable protocol models, consensus, cryptography and plugin
  interfaces;
- `doin-node`: the single unified runtime process;
- `doin-plugins`: optimization, inference and synthetic-data domain adapters.

Historical standalone optimizer/evaluator repositories are not part of this
deployment. Do not import from them or recreate their process split.

One `doin-node` process can act as relay, optimizer, evaluator and inference
worker for one or more domains. Machine-specific JSON files select these roles
and plugin configurations. Gamma may eventually run isolated unified-node
instances for its RTX 5070 Ti and RTX 5090, so silently ignored machine or
domain fields would invalidate resource isolation and experimental lineage.

### Trading and ML context

The larger project is an adaptive multi-asset trading system. Individual
asset/timeframe policies, rush detection and portfolio allocation are trained
and evaluated using chronological weekly walk-forward procedures. DOIN will
perform decentralized Level 2 optimization and evaluator verification.

This task does **not** implement trading, retraining, rush detection, portfolio
allocation or fitness. However, its loader carries values that control those
systems. Therefore these invariants are non-negotiable:

- deterministic seed settings must not be dropped or changed;
- `higher_is_better` and incentive direction must remain exact;
- optimizer, evaluator and synthetic-data configs must remain distinct;
- resource limits must reach runtime objects exactly;
- plugin-specific dictionaries are opaque payloads and must not be normalized,
  flattened or reinterpreted;
- no current test/validation/training semantics may be changed;
- configuration loading must not trigger model fitting, inference, networking
  or GPU initialization;
- examples are compatibility evidence, not files to rewrite until tests pass.

The reason this small-looking loader task is high risk is that an ignored JSON
field can make a successful run scientifically false: the log may name one
configuration while the runtime silently uses a default.

## Definition of Success

Success is not "the tests pass" in isolation. Success requires all of the
following:

- every current `UnifiedNodeConfig` field can be controlled from JSON;
- every current and newly required `DomainRole` field can be controlled from
  JSON;
- optimizer, inference and synthetic-data plugins receive the intended copied
  subtree;
- legacy fallback is explicit and tested;
- malformed nested config fails with domain/section context;
- CLI identity precedence is deterministic and tested without generating keys;
- existing examples load with their intended values;
- the complete existing test suite remains green;
- no unrelated behavior or file changes;
- the final report maps every requirement to code and test evidence.

Passing tests while silently ignoring a required field is failure. Adding broad
new abstractions while satisfying behavior is also failure because it expands
review and regression risk.

## Why This Task Exists

The active DOIN runtime is the unified `doin-node`; separate optimizer and
evaluator processes are historical. Existing node JSON files already contain
machine, network, role and plugin settings, but the current loader in
`src/doin_node/cli.py` ignores several supported fields.

Confirmed examples:

- `examples/predictor_omega_node_tft_binary_neat.json`
- `examples/predictor_gamma_node_tft_binary_neat.json`

Confirmed current defects:

1. Domain `inference_config` is ignored.
2. Domain `synthetic_data_config` is ignored.
3. Domain `incentive_config` is ignored even though `DomainRole` already owns an
   `IncentiveConfig`.
4. Domain-level `metric_type` is ignored.
5. Inference and synthetic plugins are both configured with
   `optimization_config` unconditionally.
6. Several `UnifiedNodeConfig` fields such as `storage_backend`,
   `network_protocol`, `discovery_enabled` and `fee_market_enabled` are read in
   examples but never copied by `load_config()`.
7. `identity_file` is supported by `UnifiedNodeConfig` but the CLI path always
   calls `load_identity(args.identity, config.data_dir)`, bypassing it.

## Authoritative Sources

Read these before editing:

1. `src/doin_node/unified.py`
   - `DomainRole`
   - `UnifiedNodeConfig`
2. `src/doin_node/cli.py`
   - `load_config`
   - `load_identity`
   - `setup_plugins`
   - `main`
3. `../doin-core/src/doin_core/consensus/incentives.py`
   - `IncentiveConfig`
4. `../doin-core/src/doin_core/models/fee_market.py`
   - `FeeConfig`
5. The two predictor JSON examples listed above.
6. `pyproject.toml` and existing tests under `tests/`.

Dataclass fields in source are authoritative. Example JSON files demonstrate
required backward compatibility. This task must not change a field's meaning.

## Mandatory Preflight and Stop Conditions

Run these commands before editing:

```bash
cd /home/harveybc/Documents/GitHub/doin-node
git rev-parse HEAD
git branch --show-current
git status --short
python --version
python -m pytest -q tests/test_unified.py
```

Continue only when:

- `HEAD` is exactly `826c713dda29bd3f27513b647e979b73c5505dd9`;
- branch is `master`;
- `git status --short` is empty;
- the baseline targeted test passes.

Stop without editing and report a blocker when:

- HEAD differs;
- the worktree is dirty;
- an authoritative dataclass field differs from the inventory in this spec;
- either referenced example is missing or invalid JSON;
- `doin-core` imports cannot be resolved;
- baseline tests fail for a reason unrelated to this task;
- a required behavior would need a `doin-core` or plugin-interface change.

Do not reset, stash, discard or overwrite somebody else's changes. Do not ask
the user to approve an implementation choice that is already fixed below.

## Mandatory Execution Procedure

Perform the task in this exact sequence.

### Step A: Source inventory

1. Read all authoritative sources listed below.
2. Enumerate every dataclass field in `DomainRole` and `UnifiedNodeConfig`.
3. Enumerate outer node keys and outer domain keys in every JSON file under
   `examples/` without treating keys inside plugin config dictionaries as node
   schema.
4. Compare that inventory to `load_config()` and record which fields are
   currently dropped.
5. Confirm imports and constructors for `ResourceLimits`, `IncentiveConfig` and
   `FeeConfig`.

Do not edit examples to make the inventory easier.

### Step B: Regression tests first

Create the focused loader/plugin tests described later. Run them before the
production change and confirm they fail for the expected missing-field or
wrong-subtree behavior. If they pass unexpectedly, inspect the current source
and report the discrepancy rather than weakening the assertions.

### Step C: Minimal production implementation

Implement only the approved fields, parsing helpers, plugin configuration
selection and identity precedence. Avoid moving unrelated functions or
reformatting whole files.

### Step D: Focused verification

Run the new test file and `tests/test_unified.py`. Fix implementation defects,
not assertions that correctly represent this specification.

### Step E: Full verification

Run the complete suite, compile check, diff check and status check. Inspect the
entire diff yourself for accidental scope expansion.

### Step F: Evidence report

Return the exact response structure defined at the end. Include actual command
results, not predictions. Do not say "done" without the requirement matrix.

## Allowed Changes

Production files:

- `src/doin_node/unified.py`
- `src/doin_node/cli.py`

Test files:

- add `tests/test_cli_config.py`, or use another single clearly named test file;
- minimally adjust an existing test only when required by the corrected
  behavior.

Documentation:

- a small correction to `README.md` is allowed only if needed to describe the
  already supported JSON fields.

## Forbidden Changes

Do not modify:

- `doin-core`;
- `doin-plugins`;
- network protocols, controlled flooding, GossipSub or peer discovery logic;
- consensus, Proof of Optimization, commit-reveal, quorum or incentives logic;
- blockchain, persistence or OLAP schemas;
- plugin base interfaces or entry-point names;
- existing example values;
- dependency versions;
- machine services, running nodes or data directories.

Do not commit, push, deploy, start a node, reset a chain or run an optimization.

## Explicitly Forbidden "Improvements"

Do not:

- convert the dataclasses to Pydantic, attrs or another config framework;
- introduce YAML, environment interpolation or remote configuration;
- create a generic configuration subsystem outside `cli.py`/`unified.py`;
- rename DON/DOIN classes, commands, fields or log messages;
- change default values to match one machine's example;
- merge the three plugin config dictionaries into one;
- remove backward-compatible fallback to `optimization_config`;
- alter plugin loading exception handling or startup continuation policy;
- add strict rejection for all unknown top-level legacy metadata;
- modify network, storage, chain, dashboard, OLAP or scheduling code;
- change `higher_is_better`, target-performance or incentive equations;
- change seed generation or deterministic verification;
- add dependencies;
- reformat unrelated code;
- update copyright, package version or metadata;
- modify generated/runtime files or machine data directories;
- weaken, skip or delete existing tests to get a green suite.

If you identify a worthwhile adjacent improvement, list it under assumptions
and limitations. Do not implement it.

## Requirement Traceability IDs

Use these IDs in test names/comments where useful and in the final evidence
matrix:

| ID | Required behavior |
| --- | --- |
| `R01` | Add separate inference and synthetic config plus metric type to `DomainRole` |
| `R02` | Materialize every `DomainRole` field from JSON |
| `R03` | Materialize nested `ResourceLimits` with clear domain/section errors |
| `R04` | Materialize nested `IncentiveConfig` without changing direction semantics |
| `R05` | Materialize every `UnifiedNodeConfig` field and preserve absent-field defaults |
| `R06` | Materialize nested `FeeConfig` with clear section errors |
| `R07` | Apply CLI, JSON and default identity precedence exactly |
| `R08` | Configure optimizer from an isolated copy of optimizer config |
| `R09` | Configure inference from its subtree with legacy fallback |
| `R10` | Configure synthetic data from its subtree with legacy fallback |
| `R11` | Inject metric type and param bounds only under declared precedence rules |
| `R12` | Never mutate raw JSON or any stored role config dictionary |
| `R13` | Preserve existing example and legacy behavior |
| `R14` | Add negative tests for malformed nested config/bounds |
| `R15` | Preserve the full existing test suite and keep the diff in scope |

## Required Design

### 1. Extend `DomainRole` without breaking existing callers

Add these backward-compatible fields with empty/default values:

```python
inference_config: dict[str, Any] = field(default_factory=dict)
synthetic_data_config: dict[str, Any] = field(default_factory=dict)
metric_type: str = ""
```

Keep `optimization_config`, `param_bounds`, `resource_limits`,
`incentive_config` and all existing fields unchanged.

### 2. Materialize every current `DomainRole` field

`load_config()` must parse all fields currently declared by `DomainRole`.

- Convert every `param_bounds` value to a two-element tuple as today.
- Build `ResourceLimits(**resource_limits)`.
- Build `IncentiveConfig(**incentive_config)`.
- Load `inference_config`, `synthetic_data_config` and `metric_type`.
- Do not mutate the raw JSON dictionaries.
- Invalid dataclass fields or malformed bounds must fail with a clear exception
  that identifies the domain and invalid section. Do not silently discard them.

### 3. Materialize every current `UnifiedNodeConfig` field

At minimum, map all fields declared in the dataclass at the base commit:

```text
host
port
data_dir
identity_file
bootstrap_peers
domains
target_block_time
initial_threshold
acceptance_tolerance
quorum_min_evaluators
quorum_fraction
quorum_tolerance
commit_reveal_max_age
finality_confirmation_depth
external_anchor_interval
require_deterministic_seed
eval_poll_interval
eval_max_concurrent
optimizer_loop_interval
storage_backend
db_path
snapshot_interval
prune_keep_blocks
network_protocol
gossip_heartbeat_interval
discovery_enabled
discovery_interval
dashboard_enabled
experiment_stats_file
olap_db_path
reset_chain
fee_market_enabled
fee_config
```

Build `FeeConfig(**fee_config)` when present, otherwise use `FeeConfig()`.
Preserve existing defaults exactly when fields are absent.

The documentation-only `$doc` field in examples may remain ignored. Do not add
strict rejection of unrelated legacy metadata in this task.

### 4. Correct identity precedence

Identity path precedence must be:

1. explicit CLI `--identity`;
2. JSON `identity_file`;
3. existing data-directory default.

Do not change identity serialization or generate a new identity during tests.

### 5. Configure each plugin with the correct subtree

Use these rules without mutating the stored role configurations:

- optimizer receives a copy of `optimization_config`;
- inference receives a copy of `inference_config` when it is non-empty,
  otherwise a copy of `optimization_config` for backward compatibility;
- synthetic data receives a copy of `synthetic_data_config` when it is
  non-empty, otherwise a copy of `optimization_config` for backward
  compatibility;
- domain `metric_type`, when non-empty, is inserted into optimizer and inference
  config only if that subtree does not already define `metric_type`;
- existing `param_bounds` injection into optimizer config remains unchanged;
- one plugin's `configure()` call must not mutate configuration passed to
  another plugin.

Do not change role conditions: optimizer loading still depends on `optimize`,
and evaluator/synthetic loading still depends on `evaluate`.

## Prescribed Implementation Shape

Use small private helpers in `src/doin_node/cli.py` so parsing and precedence
can be tested without starting a node. The exact internal names may vary only
when a nearby existing helper already provides the same behavior. The expected
separation is:

```python
def _parse_param_bounds(raw: object, *, domain_id: str) -> dict[str, tuple[float, float]]:
    ...

def _parse_resource_limits(raw: object, *, domain_id: str) -> ResourceLimits:
    ...

def _parse_incentive_config(raw: object, *, domain_id: str) -> IncentiveConfig:
    ...

def _parse_fee_config(raw: object) -> FeeConfig:
    ...

def _select_identity_path(
    cli_identity: str | None,
    configured_identity: str,
) -> str | None:
    ...
```

Requirements for these helpers:

- accept missing/empty sections and return current defaults;
- reject non-dictionary nested sections;
- preserve original exceptions as `__cause__` when wrapping them;
- include the domain ID and section name in domain error messages;
- never mutate input values;
- do no filesystem, identity, plugin or network work.

### Bounds validation

For each `param_bounds` entry:

- require a list or tuple of exactly two numeric values;
- reject booleans as numeric bounds;
- convert the pair to a tuple;
- reject non-finite bounds;
- reject lower greater than upper;
- identify the domain and parameter in the error.

Do not add categorical-bound semantics in this task.

### Top-level default preservation

Do not duplicate guessed defaults. Instantiate or otherwise reference the
current `UnifiedNodeConfig` defaults when mapping absent JSON fields. The
result of loading `{}` must match a default `UnifiedNodeConfig()` for every
non-domain field, except behavior already intentionally established by the CLI.

Nested `fee_config` must be a real `FeeConfig`; nested domain resources and
incentives must be their real dataclass types.

### Plugin config isolation

Construct a new dictionary for every `plugin.configure()` call. Even when
inference or synthetic data falls back to optimization config, do not pass the
same mutable dictionary object to two plugins.

Required effective config logic:

```text
optimizer = copy(optimization_config)
inference = copy(inference_config if non-empty else optimization_config)
synthetic = copy(synthetic_data_config if non-empty else optimization_config)

optimizer.metric_type = domain metric_type only when optimizer has no metric_type
inference.metric_type = domain metric_type only when inference has no metric_type
optimizer.param_bounds = domain bounds only when neither supported bounds key exists
```

Do not inject `metric_type` into synthetic config unless the synthetic subtree
already carries it. Do not inject parameter bounds into inference or synthetic
config.

### Identity precedence implementation

`main()` must select the path before calling `load_identity()`:

```text
CLI --identity, if non-empty
otherwise config.identity_file, if non-empty
otherwise None, allowing the existing data_dir default
```

The unit test must exercise this pure selection without calling
`PeerIdentity.load_or_generate()`.

## Expected Compatibility Outcomes

After implementation, loading
`examples/predictor_omega_node_tft_binary_neat.json` must produce a config where
at least these facts are true:

```text
storage_backend = sqlite
network_protocol = flooding
discovery_enabled = true
fee_market_enabled = false
domain.metric_type = binary
domain.incentive_config.higher_is_better = false
domain.incentive_config.tolerance_margin = 0.28
domain.synthetic_data_validation = false
domain.higher_is_better = false
```

Loading `examples/predictor_gamma_node_tft_binary_neat.json` must preserve three
distinct dictionaries and their characteristic fields:

```text
optimization_config.shared_population = true
inference_config.predictor_plugin = binary_tft
synthetic_data_config.model_file is present
```

Do not compare only dictionary identity. Assert representative values and that
mutating a recording fake plugin's received dictionary cannot mutate the role
or another plugin's received dictionary.

## Required Tests

Add focused tests that prove all behavior below.

### Loader tests

1. Load a temporary full JSON and assert every top-level
   `UnifiedNodeConfig` field above survives materialization.
2. Assert nested `FeeConfig` values survive.
3. Assert every `DomainRole` field survives, including nested
   `ResourceLimits`, `IncentiveConfig`, separate plugin configs,
   `metric_type`, bounds and target direction.
4. Load `examples/predictor_omega_node_tft_binary_neat.json` and assert at least:
   - `storage_backend == "sqlite"`;
   - `network_protocol == "flooding"`;
   - `discovery_enabled is True`;
   - `fee_market_enabled is False`;
   - role incentive tolerance is `0.28`;
   - role `metric_type == "binary"`.
5. Load `examples/predictor_gamma_node_tft_binary_neat.json` and assert
   `inference_config` and `synthetic_data_config` remain distinct from
   `optimization_config`.
6. Verify absent new subtrees preserve legacy fallback behavior.
7. Verify malformed bounds and invalid nested config keys fail clearly.
8. Compare loading an empty object with `UnifiedNodeConfig()` field by field for
   default preservation.
9. Verify plugin config dictionaries in the materialized role are independent
   objects from the raw decoded JSON.

### Plugin setup tests

Monkeypatch the three entry-point loader functions with recording fake plugin
classes. Use a minimal fake/unified node that exposes the role and registration
methods required by `setup_plugins()`.

Assert:

1. each plugin receives the intended subtree;
2. inference fallback uses optimization config when no inference config exists;
3. synthetic fallback uses optimization config when no synthetic config exists;
4. metric type does not overwrite an explicit subtree value;
5. param bounds are injected only into optimizer config;
6. source dictionaries remain unchanged after all configure calls.
7. a deliberately mutating fake plugin cannot contaminate another plugin's
   received configuration.

### Identity test

Test the pure precedence selection without writing a real identity. Refactor a
small path-selection helper if that makes the precedence testable. Do not mock
cryptography broadly or create persistent identity files.

## Commands to Run

From `/home/harveybc/Documents/GitHub/doin-node`:

```bash
python -m pytest -q tests/test_cli_config.py tests/test_unified.py
python -m pytest -q
python -m compileall -q src/doin_node tests/test_cli_config.py
git diff --check
git status --short
```

Do not run a live node or any long benchmark.

Record the exact passed/failed counts and elapsed time from both pytest
commands. If the full suite has a baseline/environment failure, do not conceal
it: isolate it, report its exact output and state whether it reproduces on the
base commit.

## Acceptance Criteria

- All declared current config fields are materialized.
- Existing example JSONs preserve their intended runtime choices.
- Separate plugin config subtrees reach the correct plugin.
- Legacy JSONs without separate subtrees continue to work.
- Nested configs fail clearly when malformed.
- Existing tests and all new tests pass.
- No protocol, chain, OLAP or plugin-interface behavior changes.
- The diff contains only allowed files and no runtime artifacts.

## Required Response Format

Return exactly these sections:

1. `Preflight evidence`
2. `Implementation summary`
3. `Files changed`
4. `Requirement evidence matrix`
5. `Behavior and backward compatibility`
6. `Tests added`
7. `Commands executed and exact results`
8. `Self-review findings`
9. `Assumptions, limitations and blockers`
10. `Git branch, HEAD and status`

The requirement evidence matrix must contain one row for every `R01` through
`R15` with:

```text
requirement ID | implementation file/function | test name | result
```

Under `Files changed`, explain why each file needed modification. Under
`Self-review findings`, explicitly confirm whether you found any unrelated
change, mutable-config aliasing, ignored dataclass field, weakened assertion,
or untested fallback.

Do not claim that the task is accepted or production-ready. Codex will inspect
the actual diff, compare every requirement to source and rerun critical tests.

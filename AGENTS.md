# AGENTS.md — agent-multi

Guidance for coding agents working in this repository. Follows the
[agents.md](https://agents.md) convention.

> **Read this before running anything.** This is an experiment platform whose
> real workloads are GPU training runs measured in hours. Campaign workers may
> be executing right now on this machine or on assigned hosts. The quickstart
> below deliberately uses only non-GPU, non-training commands. Do not
> substitute a real experiment launch for it.

## Project overview

`agent-multi` is a plugin-based reinforcement-learning trading experiment
platform. It trains and evaluates RL agents (SAC, PPO, DQN and heuristic
baselines) on the [gym-fx](https://github.com/harveybc/gym-fx) environment,
wraps training in validation and curriculum pipelines with nested time splits
and paired-generalization contrasts, runs population-based hyperparameter
optimisation, and supervises distributed DOIN campaigns. Experiments are JSON
configs naming agent, environment, pipeline, optimizer and execution-policy
plugins resolved from entry points; the current research fronts additionally
run through contract-driven runners in `tools/`.

It does **not** simulate the market or execute orders — that is `gym-fx`. It
does **not** own the decentralised protocol, blockchain persistence, candidate
leasing or champion migration — that is `doin-node` / `doin-core`. It does
**not** own the DOIN adapter plugins `trading_asset` / `trading_scenario`,
which live in `doin-plugins` and wrap this repository's pipeline rather than
replacing it. It does **not** produce price-prediction models — that is
`predictor`. All training and evaluation run in simulation or backtest over
historical or synthetic data; this repository places no real-capital trades.

## Agent quickstart (install → run → show the user results)

### 1. Environment

```bash
git clone https://github.com/harveybc/gym-fx.git
git clone https://github.com/harveybc/trading-contracts.git
git clone https://github.com/harveybc/agent-multi.git
pip install -e gym-fx -e trading-contracts -e agent-multi
```

Unverified in a clean environment — not re-executed from scratch for this
document. The canonical environment is a conda env pinned in
`docs/environment/UBUNTU26_TRADING_STACK.md` (Python 3.12.13,
stable-baselines3 2.9.0). Install in a dedicated environment: the top-level
package names (`app`, `*_plugins`) and the entry-point groups
`pipeline.plugins` / `optimizer.plugins` / `preprocessor.plugins` are shared
with sibling repositories, so co-installation mixes the groups' contents.

Check the two non-hermetic test fixtures before running the suite. This is
read-only and never clones or writes:

```bash
python tools/bootstrap_test_fixtures.py --check-only
```

Verified output: schema `agent_multi.test_fixture_bootstrap.v1` listing the
pinned base contract (`status: ready`) and the sibling `doin-node` checkout.
On this machine the sibling reports `revision_mismatch` with an explicit
remedy, and the overall outcome is `FIXTURES_INCOMPLETE`. The tool states
plainly that it never mutates an existing checkout — do not "fix" the sibling
without being asked.

### 2. Test suite

```bash
python -m pytest tests --collect-only -q   # 1579 tests collected in ~3 s
python -m pytest tests                     # full run
```

Verified: **1579 tests collected, zero collection errors.** There is no
`pytest.ini`, `pyproject.toml` or `setup.cfg` — pytest runs on defaults with
rootdir at the repository root. The only marker in use is
`@pytest.mark.parametrize`; there are no custom markers and no `-m` selectors.

Full-suite wall-clock time is **not verified here** — it was not run, to avoid
contending with live workers.

**Critical suite invariant.** `tests/conftest.py` installs a session-scoped
autouse fixture, `suite_leaves_checkouts_byte_clean`, which snapshots
`git status --porcelain --untracked-files=all` for this repository *and* the
sibling `../doin-node` checkout before and after the session, and fails the
whole suite if either tree changed. Practical consequence for an agent:
**never create, delete or edit files in either checkout while the suite is
running.** Tests must write only below `tmp_path`.

The CI subset, if you want something narrower:

```bash
pytest -q \
  tests/unit/test_adaptive_order_router.py \
  tests/unit/test_analyze_swarm_efficiency.py \
  tests/unit/test_default_optimizer_shared.py \
  tests/unit/test_incident_corpus_manifest.py \
  tests/unit/test_project3_portfolio_supervisor.py \
  tests/unit/test_publication_scaffolds.py
```

### 3. Lint

Ruff lives in `tooling/`, not in `pyproject.toml`, and is vendored with a
pinned wheel and its own venv:

```bash
tooling/venv/bin/ruff check --config tooling/ruff.toml .
```

Verified: ruff 0.13.1, output `Found 2 errors.`, **exit code 1**. That is the
expected state, not a regression — `tooling/ruff_baseline.json` records those
same two `F811` findings in `app/data_handler.py` as accepted history. The
config selects correctness rules only (`E9, F63, F7, F82, F811`), preview
disabled. **Never pass `--fix`**; the config header states it is forbidden in
CI and evidence generation, and old code is deliberately not mass-edited.

### 4. Representative safe run — the P1LR preflight (no GPU, no training)

```bash
python tools/p1_difficulty_lr_factorial.py --preflight
```

This is the honest small example: it performs a **real** materialisation of
the nested data splits into a `tempfile.TemporaryDirectory`, without
constructing any model, and prints a typed JSON payload to stdout. Omit
`--output` and it writes nothing at all.

Verified end to end: schema `agent_multi.p1_difficulty_lr_preflight.v1`,
`"outcome": "PREFLIGHT_PASS"`, `"training_used": false`, `"refusals": []`,
exit code 0, and the repository working tree unchanged afterwards. The payload
proves, among other things:

- Four nested roles materialised with row counts and CSV sha256 —
  `fit_train` (11 509 scored rows), `train_monitor` (2 190), `inner_validation`
  (2 190), `outer_validation` (2 196), each with 256 context bars where the
  contract requires them.
- `"sealed_test": {"status": "SEALED", "csv_sha256": null, ...}` with
  `"sealed_test_csv_absent": true` — the sealed evaluation window was not
  materialised. If a `sealed_test.csv` had appeared, the tool would have
  appended a typed refusal instead of passing.
- 16 distinct cell identities per mode (4 seeds × 4 factorial cells), plus the
  contract sha256 and the nested-split contract sha256.

Related non-GPU modes on the L2 runner:

```bash
python tools/l2_curriculum_arms.py --preflight        # same no-training proof
python tools/l2_curriculum_arms.py --dispatch-plan    # prints commands; authorizes nothing
```

**Never pass `--seed {101,202,303,404}` to the P1LR runner or `--worker
{w1..w4}` to the L2 runner.** Those are the GPU training modes.

### 5. Read-only status

```bash
python tools/multifront_status.py --no-emit-alerts --no-transition-queue
```

`tools/multifront_status.py` aggregates existing tier-0 evidence read-only and
never invents a value — a source it cannot read yields an explicit
`unavailable` entry. `--help` verified. Three things to know before running
it:

- `tools/TOOL_DECLARATIONS.json` declares it `mutability: mixed`, not
  `read_only`. Its writes never land in the repository or in a run's output
  root; they land under `~/.local/state/agent-multi/`.
- `--no-emit-alerts` suppresses incident emission; `--no-transition-queue`
  suppresses enrolment of durable transition records. Omitting `--output`
  suppresses the report file. Adding `--no-l1` additionally suppresses an
  ETA-sample append, at the cost of dropping the L1 section.
- It shells out to `ssh` and `systemctl --user show` to read remote worker
  state. These are read-only queries, but they touch the network. If you must
  stay entirely local, skip this step.

Config linting, also read-only, never authorising a launch:

```bash
python tools/config_doctor.py examples/config/doin/trading_asset_solusdt_4h_sac_v1.json
```

Exit codes: `0` PASS/WARNING, `2` BLOCK, `3` required UNAVAILABLE, `4` harness
error.

### 6. Final message to the user

Report exactly this:

> Everything ran on CPU. No experiment was launched, no GPU was used and no
> campaign state was modified.
>
> - **Test suite:** `1579 tests collected` with zero collection errors
>   (`python -m pytest tests --collect-only -q`). Full-suite runtime was not
>   measured, to avoid contending with live workers.
> - **Lint:** `tooling/venv/bin/ruff check --config tooling/ruff.toml .` →
>   `Found 2 errors`, exit 1. That is the accepted baseline recorded in
>   `tooling/ruff_baseline.json`, not a regression.
> - **Preflight:** `python tools/p1_difficulty_lr_factorial.py --preflight` →
>   `"outcome": "PREFLIGHT_PASS"`, `"training_used": false`, `"refusals": []`.
>   The payload was printed to stdout; nothing was written to disk.
> - **Where real evidence lives:** not in this repository. Run outputs and
>   sealed collections land under `~/.local/share/agent-multi/`, and
>   tool-owned durable state under `~/.local/state/agent-multi/`. The repo's
>   `records/` holds only a fleet runtime manifest, and `knowledge/` only a
>   hashed concept bundle.
>
> **One thing to inspect first** — in the preflight payload, look at
> `nested_role_facts` next to `sealed_test_state`:
>
> ```bash
> python tools/p1_difficulty_lr_factorial.py --preflight \
>   | python -c "import json,sys; d=json.load(sys.stdin); print(json.dumps(d['nested_role_facts'], indent=2)); print('sealed:', d['sealed_test_state'], d['sealed_test_csv_absent'])"
> ```
>
> Four roles materialise with real row counts and CSV hashes; the fifth,
> `sealed_test`, comes back `SEALED` with every field `null` and
> `sealed_test_csv_absent: true`. That asymmetry is the platform's central
> discipline made checkable: the evaluation window is proven *absent* from the
> materialisation rather than merely unused, and a run that touched it would
> emit a typed refusal instead of a number.

## Build, test and lint commands

```bash
# install
pip install -e gym-fx -e trading-contracts -e agent-multi
pip install -e ".[dev]"                       # the only extra: pytest

# CI install (hash-locked)
python -m pip install --require-hashes -r requirements-ci.txt

# tests
python -m pytest tests --collect-only -q      # 1579 collected
python -m pytest tests                        # full run
python tools/bootstrap_test_fixtures.py --check-only

# lint (expect exit 1 with 2 baseline findings; never --fix)
tooling/venv/bin/ruff check --config tooling/ruff.toml .

# non-GPU tools
python tools/p1_difficulty_lr_factorial.py --preflight
python tools/l2_curriculum_arms.py --preflight
python tools/l2_curriculum_arms.py --dispatch-plan
python tools/config_doctor.py <config.json> [...]
python tools/multifront_status.py --no-emit-alerts --no-transition-queue

# real experiment (NOT part of the quickstart; GPU-hours scale)
agent-multi --load_config <experiment-config.json>
```

`requirements-ci.txt` is a `pip-compile --generate-hashes` output; regenerate
it with the command recorded in its header, never by hand.

## Layout

| Path | Contents |
|---|---|
| `app/` | Runtime core: `main.py` (config merge and entry), `plugin_loader.py`, `campaign_supervisor.py`, `config_validation.py`, `config_merger.py`, `canonical_config.py`, `runtime_overlay.py`, `stopping_contract.py`, `weekly_promotion.py`, `metrics.py`, `live_parity.py`, `data_handler.py`, `cli.py`. |
| `agent_plugins/` | Nine agent plugins: SAC, PPO, DQN, the project3 actor-critic, and the random / buy-hold / no-trade / momentum / reversal baselines. |
| `env_plugins/` | `gym_fx_env.py` (the only registered env plugin) and `execution_cost_curriculum.py`. |
| `pipeline_plugins/` | Four registered pipelines plus private helpers for nested splits, paired generalisation, observation contracts, lexicographic selection, return traces, weekly metrics, execution curricula and actor liveness. |
| `optimizer_plugins/` | `default_optimizer` and `project3_full_genome_optimizer` (registered), plus `l2_curriculum_optimizer.py` (imported directly by `tools/l2_curriculum_arms.py`, not an entry point). |
| `execution_policy_plugins/` | `adaptive_order_router.py`, the single `execution_policy.plugins` entry. |
| `tools/` | ~130 operational tools: contract-driven runners, aggregators, verdicts, watchdogs, incident pipeline, evidence collectors and gates. `TOOL_DECLARATIONS.json` is the semantic source of truth; `ENGINEERING_SURFACE_INDEX.json` is generated from it. |
| `tooling/` | **Only the ruff toolchain**: `ruff.toml`, `ruff_baseline.json`, a pinned lock, a vendored wheel and a gitignored `venv/`. Not a general tools directory. |
| `tests/` | 1579 tests: top-level `test_*.py`, `unit/`, `integration/`, plus the hermeticity-guard `conftest.py`. |
| `docs/` | `work_plan/` (the canonical architecture corpus), `handoffs/`, `audits/` (including `audits/evidence/`), `publications/`, `environment/` (the pinned conda env and pip lock). |
| `examples/` | `config/` (experiment contracts, including the live campaign contracts), `campaigns/`, `scripts/`, `systemd/` (unit templates), `results/` (mostly gitignored), `models/`, `data/`. |
| `configs/` | Per-host runtime overlays and a shortlist manifest. |
| `records/` | Two files only — a terminal record and a fleet runtime manifest. **Not** where run evidence lands. |
| `knowledge/` | The `okf/` concept bundle: eight typed concept documents plus `MANIFEST.sha256`, validated by `tools/okf_validate.py`. |
| `papers/` | Five publication scaffolds, validated in CI by `tools/validate_publication_scaffolds.py`. |
| `experiments/`, `artifacts/`, `logs/` | Local scratch, almost entirely gitignored (`/artifacts/` wholesale). |

## Conventions and constraints

- **Plugin architecture via entry points.** Five groups in `setup.py`:
  `env.plugins`, `agent.plugins`, `pipeline.plugins`,
  `execution_policy.plugins`, `optimizer.plugins`, plus two console scripts
  (`agent-multi`, `doin-campaign-supervisor`). Agents, envs, optimizers and
  policies expose a class named `Plugin`; pipelines expose `PipelinePlugin`.
  Resolution is via `importlib.metadata` in `app/plugin_loader.py`.
- **Typed refusals.** The pervasive idiom: a fact that cannot be established
  renders as an explicit typed refusal or `unavailable`, never as a default, a
  zero or a guess. Examples in shipped help text: "an identity belonging to
  the other mode's root is a typed refusal, never a rendered zero"; "an
  all-degenerate campaign is typed `CAMPAIGN_UNANSWERABLE`, never a null";
  `tools/engineering_surface_index.py` puts it as "UNCLASSIFIED is an honest
  state, never a guess". When you extend a tool, keep this property.
- **Fail-closed gates.** Unknown major contract versions fail closed. The
  launch gate is `tools/gpu_readiness_probe.py`; the publication gate is
  `tools/prepush_sensitivity_gate.py`. A truncated scan fails closed rather
  than passing.
- **Declared tool surface.** Every executable tool must appear in
  `tools/TOOL_DECLARATIONS.json` (with `purpose`, `lifecycle`, `mutability`,
  `authority_class`, `owner`) or in its `known_unclassified_baseline`. A tool
  absent from both fails CI. If you add a tool, declare it.
  ⚠️ One declaration is misleading:
  `tools/l2_observation_boundary_validation.py` is declared `read_only` but
  its own docstring describes running training epochs with
  `OBS_VALIDATION_DEVICE` defaulting to `cuda`. **Treat it as
  GPU-consuming and do not run it.**
- **Sealed evaluation window.** The calendar-2025 `sealed_test` split is never
  opened. The preflight asserts `sealed_test.csv` does not exist in the
  materialisation and refuses if it does;
  `tests/unit/test_validation_pipeline_test_firewall.py` guards the pipeline
  side. Do not add a code path that reads it.
- **Sealed evidence collections.** Terminal runs are fetched from their
  assigned hosts and sealed into content-addressed collection roots with
  recorded tree digests. A changed byte under a `sealed/` tree invalidates a
  recorded digest and the audit trail.
- **Checkpoint identity is the policy tensor hash**, never the `.zip`
  container digest — SB3 archives embed member timestamps.
- **Pre-push sensitivity gate.** Install it before pushing anything:
  ```bash
  python tools/prepush_sensitivity_gate.py --help
  cp tools/hooks/pre-push .git/hooks/pre-push
  chmod +x .git/hooks/pre-push
  ```
  It scans only the added text lines of the outgoing commits for six typed
  classes — credentials, account identifiers, signer key paths, third-party
  content, live stop-loss/take-profit levels, and topology (IP addresses,
  hostname assignments, GPU UUIDs beyond the allowlist). Matched spans are
  redacted in every excerpt: the gate never republishes the value it found.
  Any finding, or a scan truncated by `--max-commits`, exits nonzero and
  blocks the push. `git push --no-verify` bypasses every hook and the standing
  evidence contract forbids it for sensitive content.
  `tools/prepush_sensitivity_allowlist.json` is shrink-only unless the owner
  authorises an addition.
- **Evidence lands outside the checkout.** Run outputs and sealed collections
  under `~/.local/share/agent-multi/`; tool-owned durable state (watchdogs,
  incident ledger, transition queue, audit snapshots, preflight payloads)
  under `~/.local/state/agent-multi/`. Operator authority lives under
  `~/.config/agent-multi/` and never in git.
- **Documentation precedence.** `docs/work_plan/` is the canonical
  architecture corpus. `docs/audits/` and `docs/handoffs/` are dated evidence
  packets: append new dated documents, never rewrite old ones. Historical
  reports elsewhere in `docs/` describe the campaigns they belong to.

## Do not touch

- **Do not launch a real experiment, and do not use a GPU.** Never pass
  `--seed` to `tools/p1_difficulty_lr_factorial.py` or `--worker` to
  `tools/l2_curriculum_arms.py`. Do not run
  `tools/l2_observation_boundary_validation.py` (see above). Do not start,
  stop or restart any campaign supervisor, fleet launcher or worker process.
  `run_smokes.py` is CPU-forced and capped at 1000 timesteps, but it is still
  real training and will contend with the fleet — it is not the quickstart.
- **Do not touch the runtime git worktrees.** Workers execute from pinned
  detached worktrees under `../.runtime/`, and there are review worktrees
  under `../.worktrees/`. Never `git worktree remove`, never check them out,
  never edit them, and avoid `git gc` / `git prune` in the main checkout,
  which could disturb refs they depend on.
- **Do not create, delete or edit files while the test suite is running.**
  The `conftest.py` hermeticity guard compares before/after `git status` for
  this repository and the sibling `doin-node` checkout and fails the session
  if either changed.
- **Do not write into `~/.local/share/agent-multi/` or
  `~/.local/state/agent-multi/`.** These hold live run roots with per-cell
  exclusive-claim lock files and heartbeats, the incident ledger, watchdog
  state and the transition queue. Deleting a `.lock` would let a second worker
  claim a cell that is already being trained. Read-only inspection of a
  heartbeat or a sealed `cell_record.json` is fine.
- **Do not modify sealed or pinned artifacts:** anything under a `sealed/`
  tree in a collection root;
  `examples/results/project3_ethusdt_4h_sac_train_val_test_v2/config_out.json`
  (whitelisted past the results ignore rule precisely because its sha is a
  decision-bearing pin); `knowledge/okf/MANIFEST.sha256`;
  `tooling/ruff_baseline.json`; `tools/prepush_sensitivity_allowlist.json`.
- **This repository is public.** Never commit hostnames, private IP addresses,
  GPU UUIDs, API keys, tokens, private keys, signed capabilities, broker
  account identifiers or account fingerprints, personal identity data, or
  non-redistributable datasets. Use `<your-host>` placeholders. See
  `SECURITY.md`, and let the pre-push gate check you.
- **Do not touch sibling repositories.** `lts` runs live paper/demo trading
  runners; `doin-node` is pinned as a test fixture at a specific revision and
  the fixture tool deliberately refuses to mutate it.

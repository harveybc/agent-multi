# agent-multi

Plugin-based reinforcement-learning trading experiment platform. agent-multi
trains and evaluates RL agents (SAC, PPO, DQN and heuristic baselines) on the
[gym-fx](https://github.com/harveybc/gym-fx) trading environment, wraps
training in validation and curriculum pipelines with nested time splits and
paired-generalization contrasts, runs population-based hyperparameter
optimization, and supervises distributed DOIN optimization campaigns. Every
experiment is a JSON config that names agent, environment, pipeline, optimizer
and execution-policy plugins loaded from entry points.

## Status

**Main phase2 lane (2026-11-13):** all STL work is paused by the owner;
its services are disabled and no seasonal features enter the selection pool.
The selected71 actor and all49 cumulative return candidates are preserved.
Next is the frozen-actor [critic coverage contrast314](docs/work_plan/314_CRITIC_COVERAGE_CONTRAST.md),
comparing113 versus146 calibration origins on the same55 held-out anchors.
Older status entries below describe historical experiments, not running queues.

**Threshold calibration (2026-11-10):** [results261](docs/work_plan/261_CLOSE_THRESHOLD_RESULTS_AND_PHASE2_ROADMAP.md)
complete48 fit-only calibrations on24 frozen heads. The primary monitor result
worsens, so no calibration or model is promoted. The preserved69-input actor
remains available for the next phase2 pass; data-expansion work is separate.

**Decision-loss follow-up (2026-11-10):** [results259](docs/work_plan/259_POSITION_DECISION_LOSS_RESULTS.md)
complete16 paired head fits. The paired objective improves harmful-close counts
at nearly unchanged closing activity, but ranking/threshold and constant-market
controls still prevent actor promotion. The original policy remains untouched.

**Critic/interface diagnostic (2026-11-10):** [results257](docs/work_plan/257_POSITION_VALUE_SURFACE_RESULTS.md)
complete twelve market-only close/hold field contrasts with external position
evaluation. Source actor and feature-selection artifacts remain unchanged.
Constant-market and harmful-close controls prevent attributing the observed
interface improvement to learned market patterns or promoting an unqualified head.

**Generalization follow-up (2026-11-06):** completed regularization diagnostics
are in [204](docs/work_plan/204_GENERALIZATION_DIAGNOSIS_RESULTS.md). The independent
multiscale contrast is automated under [205](docs/work_plan/205_AUTOMATED_MULTISCALE_CONTRAST.md).
The SNR/capacity review and next93-input frozen removal/reinsertion pass are
tracked in [207](docs/work_plan/207_SNR_REVIEW_AND_NEXT_MASK_PASS.md).
That pass is now complete: **93 ->76 ->76 inputs,28 groups**, with frozen-mask
validation return+11.2871% and205 orders; see
[208](docs/work_plan/208_R1_FROZEN_76_TWO_PASS_RESULTS.md) for the full comparison,
native statistics and distinction from a fresh structural refit.
The proposed70x-capacity arm has not been launched: the supplied analysis
describes a different model and does not establish the required network size.

**Current selection diagnostic (2026-11-04):** the selected94-input frozen actor
now preserves the masked108 policy exactly, including full TRAIN/VALIDATION
trading accounts. See [plan197](docs/work_plan/197_FUNCTION_PRESERVING_COMPACTION.md)
and [results198](docs/work_plan/198_FROZEN_COMPACTION_RESULTS.md) for the
weight-space transformation, controls and saved inference artifact. This is
preservation of a development-selected policy, not a new generalization result
or a migrated trainable SAC checkpoint.

The subsequent owner-authorized removal/reinsertion round is also complete:
**94 ->93 ->93 inputs**,30 learned groups,90 authenticated evaluations and an
exactly replayable93-input frozen actor. See
[results200](docs/work_plan/200_COMPACT93_TWO_PASS_RESULTS.md) for TRAIN versus
VALIDATION, the removed feature, trading detail and artifact identity.

**Publication scope (2026-09-14):** this default branch contains the platform
described below. Subsequent campaign execution, data-readiness and accounting
work is available in a [versioned research snapshot](https://github.com/harveybc/agent-multi/tree/909db61596b788b4cd78932babaa5d161ef843e1).
That work is not all integrated into `master`; use the snapshot's own contracts
and instructions when examining a particular experiment.

The current research direction is data-centric: input characterization,
causal preprocessing and representation comparisons precede larger model
searches. See the [repository map](https://github.com/harveybc/predictor/blob/master/docs/RESEARCH_STACK.md)
and [doctoral proposal](https://github.com/harveybc/predictor/blob/master/docs/propuesta_doctoral_representaciones_temporales_modulares.pdf).
A runnable agent or a positive training curve is not evidence of trading
profitability. Examples are for offline research, not broker deployment.

**Lifecycle: ACTIVE-CORE.** This is the experiment platform behind the owner's
current trading-research campaigns; its pipelines, evidence tooling and audit
corpus are in daily use.

> **Disclaimer:** training and evaluation run in simulation/backtest over
> historical or synthetic data through gym-fx. This repository places no
> real-capital trades; live or demo execution belongs to separate downstream
> systems. Nothing in the examples is financial advice.

## Use with a coding agent

Paste this into Claude Code, Cursor, Codex, GitHub Copilot or any coding agent
with shell access:

> Read `AGENTS.md` in this repository and follow the **Agent quickstart**
> section end to end: set up the environment, run the smoke test, execute the
> example non-GPU preflight, then tell me the exact URL or file paths where I
> can see the results and one query I should try first.

`AGENTS.md` is the [agents.md](https://agents.md) convention, read natively by
most coding agents. Its quickstart uses only non-GPU, non-training commands —
launching a real experiment is deliberately excluded. The standing rules for
any agent reading this repository:

> Use CLI help and bounded CPU fixtures first. Keep real data, checkpoints,
> output directories and running jobs unchanged. Report observation/target
> definitions, temporal boundaries, effective cost model and exact outcomes.
> Never equate a completed training call with positive trading evidence.
> Use the reviewed governance integration for scientific runs; do not launch
> GPU sweeps or connect to a broker as part of repository evaluation.

## Role and non-responsibilities

**Role:** own the RL experiment lifecycle — configs, training pipelines,
validation/curriculum logic, local optimizers, campaign supervision and
evidence/audit tooling.

**Not responsible for:**

- Market simulation and order execution — that is
  [gym-fx](https://github.com/harveybc/gym-fx) (consumed through the
  `gym_fx_env` plugin).
- Decentralized protocol, blockchain persistence, candidate leasing,
  deduplication and champion migration — that is
  [doin-node](https://github.com/harveybc/doin-node) /
  [doin-core](https://github.com/harveybc/doin-core).
- The DOIN adapter plugins (`trading_asset`, `trading_scenario`) — those live
  in [doin-plugins](https://github.com/harveybc/doin-plugins) and *wrap* this
  repository's local pipeline; they do not replace it.
- Price prediction models — see
  [predictor](https://github.com/harveybc/predictor).

## Architecture

```
JSON experiment config (examples/config/)
        │
        ▼
app/main.py ── app/plugin_loader.py (importlib.metadata entry points)
        │
        ▼
pipeline plugin (pipeline_plugins/)
  ├─ agent plugin  (agent_plugins/: sac, ppo, dqn, baselines)
  ├─ env plugin    (env_plugins/gym_fx_env → gym-fx)
  ├─ optimizer     (optimizer_plugins/: DEAP population search)
  └─ execution policy (execution_policy_plugins/)
```

Pipeline internals (in [`pipeline_plugins/`](pipeline_plugins/)):

- [`rl_pipeline_with_validation.py`](pipeline_plugins/rl_pipeline_with_validation.py)
  — chronological train/validation/test splitting so selection never touches
  the sealed evaluation window, plus a train-versus-held-out
  generalization-gap penalty on candidate selection.
- [`_return_trace.py`](pipeline_plugins/_return_trace.py),
  [`_weekly_metrics.py`](pipeline_plugins/_weekly_metrics.py),
  [`_lexicographic_selection.py`](pipeline_plugins/_lexicographic_selection.py),
  [`_observation_contract.py`](pipeline_plugins/_observation_contract.py),
  [`_execution_curriculum.py`](pipeline_plugins/_execution_curriculum.py) —
  return-trace evidence, weekly aggregation, lexicographic candidate
  selection, observation-contract checks and execution curricula.

Campaign supervision and evidence tooling:

- [`app/campaign_supervisor.py`](app/campaign_supervisor.py) (console script
  `doin-campaign-supervisor`) — durable launcher that starts, adopts and
  monitors DOIN worker processes, enforces restart limits and emits incident
  events.
- [`tools/`](tools/) — config doctor, fleet launchers, aggregators
  (curriculum decisions, factorial screens), incident ledger/router, audit
  snapshot and test-evidence collectors, watchdogs. Index in
  [`tools/ENGINEERING_SURFACE_INDEX.json`](tools/ENGINEERING_SURFACE_INDEX.json).
- [`docs/audits/`](docs/audits/) and [`docs/handoffs/`](docs/handoffs/) —
  audit findings, evidence packets and handoff reports;
  [`docs/work_plan/`](docs/work_plan/) — the architecture/work-plan corpus.
  Current baseline: [three-signal discrete SAC](docs/work_plan/180_THREE_SIGNAL_SAC_BASELINE.md), with sequential economic qualification and validation-monitored stopping, before floating feature selection.

## Prerequisites

From [`setup.py`](setup.py): `numpy`, `pandas`, `scipy`, `scikit-learn`,
`psutil`, `requests`, `gymnasium`, `backtrader`, `stable-baselines3>=2.3`,
`deap`, `trading-contracts>=0.1.0`. No `python_requires` is declared; the
platform is exercised in practice on Python 3.12 (verified below with Python
3.12.13, stable-baselines3 2.9.0). CI pins live in
[`requirements-ci.txt`](requirements-ci.txt).

## Installation

```bash
git clone https://github.com/harveybc/gym-fx.git
git clone https://github.com/harveybc/trading-contracts.git
git clone https://github.com/harveybc/agent-multi.git
pip install -e gym-fx -e trading-contracts -e agent-multi
```

*Unverified in a clean environment* — these are the standard editable
installs; they were not re-executed from scratch for this README. Imports and
test collection were verified in an existing Python 3.12.13 environment
(results quoted below).

## Smallest working local example

A repository-owned experiment config that runs the local SAC pipeline without
any DOIN infrastructure:

```bash
agent-multi --load_config examples/config/doin/trading_asset_solusdt_4h_sac_v1.json
```

*Unverified for this README* (full SAC training is GPU-hours-scale). A cheap
CPU smoke harness exists in [`run_smokes.py`](run_smokes.py), which caps
configs at 1000 timesteps (also unverified here).

For something that runs in seconds, consumes no GPU and constructs no model,
use the factorial runner's preflight:

```bash
python tools/p1_difficulty_lr_factorial.py --preflight
```

Verified: exit 0, schema `agent_multi.p1_difficulty_lr_preflight.v1`,
`"outcome": "PREFLIGHT_PASS"`, `"training_used": false`, `"refusals": []`. It
materialises the nested splits into a temporary directory, reports each role's
row count and CSV sha256, and proves the sealed evaluation window was not
materialised (`"sealed_test_state": "SEALED"`,
`"sealed_test_csv_absent": true`). It writes nothing unless you pass
`--output`. The equivalent on the L2 runner is
`python tools/l2_curriculum_arms.py --preflight`, and `--dispatch-plan` prints
the per-host launch commands without authorizing anything.

Also verified:

```bash
python -c "from pipeline_plugins.rl_pipeline_with_validation import PipelinePlugin; print('rl_pipeline_with_validation OK')"
# observed: "rl_pipeline_with_validation OK"
```

An agent-executable version of this recipe is in [`AGENTS.md`](AGENTS.md).

## Distributed / DOIN usage

agent-multi participates in DOIN campaigns while keeping a strict boundary:

- **Local optimizers remain here.** `optimizer.plugins` in this repository
  (see [`optimizer_plugins/`](optimizer_plugins/)) drive local
  population-based search and work without DOIN. DOIN *extends* them
  collaboratively — champion migration, candidate leasing, duplicate-evaluation
  avoidance and blockchain persistence are owned by
  [doin-node](https://github.com/harveybc/doin-node) — it does not absorb
  them.
- [doin-plugins](https://github.com/harveybc/doin-plugins) registers the
  `trading_asset` / `trading_scenario` entry points that adapt this
  repository's pipeline to the DOIN plugin interface. As
  [`examples/config/doin/README.md`](examples/config/doin/README.md) states:
  `default_optimizer` remains the local optimizer plugin; `trading_asset` is
  the DOIN adapter around it, not a replacement.
- `doin-campaign-supervisor`
  ([`app/campaign_supervisor.py`](app/campaign_supervisor.py)) launches
  doin-node workers as `python -m doin_node.cli --config <node-config>.json`,
  supervises them (heartbeats, adoption of surviving processes, restart
  limits) and records incidents. Machine-specific node configs and runtime
  overlays are intentionally not documented here.
- Retired `doin-optimizer` / `doin-evaluator` services are **not** required by
  any current deployment; their roles run inside doin-node.

Experiment configs under [`examples/config/doin/`](examples/config/doin/) are
portable and contain no host, GPU, credential or account settings.

## Configuration and plugins

Experiments are JSON configs merged by [`app/main.py`](app/main.py) from CLI
flags, `--load_config`, optional base configs, candidate patches and runtime
overlays. Plugins resolve via [`app/plugin_loader.py`](app/plugin_loader.py)
from entry points declared in [`setup.py`](setup.py):

| Entry-point group | Plugins (this package) |
|---|---|
| `env.plugins` | `gym_fx_env` |
| `agent.plugins` | `sac_agent`, `ppo_agent`, `dqn_agent`, `project3_sac_actor_critic_agent`, `random_agent`, `buy_hold_agent`, `no_trade_agent`, `momentum_agent`, `reversal_agent` |
| `pipeline.plugins` | `rl_pipeline`, `rl_pipeline_with_validation`, `rl_pipeline_with_solvency_curriculum`, `rl_pipeline_with_execution_curriculum` |
| `optimizer.plugins` | `default_optimizer`, `project3_full_genome_optimizer` |
| `execution_policy.plugins` | `adaptive_order_router` |

Console scripts: `agent-multi` (experiment runner) and
`doin-campaign-supervisor` (campaign supervision).

## Tests

```bash
python tools/bootstrap_test_fixtures.py --check-only   # verify fixtures; never writes
python -m pytest tests --collect-only -q               # observed: 1579 collected in ~3 s
python -m pytest tests                                 # full run: duration unverified
```

Collection is clean (zero errors). There is no `pytest.ini` or pytest section
in `pyproject.toml`; the suite runs on defaults, and the only marker in use is
`@pytest.mark.parametrize`. `tests/conftest.py` installs a session-scoped
guard that fails the suite if this checkout or the sibling `doin-node`
checkout is left dirty, so do not edit files while it runs.

Linting is a vendored, pinned ruff in [`tooling/`](tooling/):

```bash
tooling/venv/bin/ruff check --config tooling/ruff.toml .
```

Observed: ruff 0.13.1, `Found 2 errors`, exit code 1. That is the accepted
state recorded in [`tooling/ruff_baseline.json`](tooling/ruff_baseline.json),
not a regression. The config selects correctness rules only; `--fix` is
forbidden in CI and evidence generation.

Config linting: [`tools/config_doctor.py`](tools/config_doctor.py). Audit
evidence collection: [`tools/audit_test_evidence.py`](tools/audit_test_evidence.py).

## Outputs and reproducibility

Runs write a results summary JSON (`results_file`), the fully merged effective
config (`save_config`, sufficient to reproduce the run), model checkpoints
(`save_model`) and, for validation/curriculum pipelines, return traces and
weekly metrics used as selection evidence. Campaign and audit artifacts are
aggregated by the tools under [`tools/`](tools/) and documented in
[`docs/audits/`](docs/audits/). Seeds are explicit in configs; deterministic
seeding contracts for distributed runs are enforced by the DOIN layer.

Those outputs land **outside the checkout**, at the roots each contract
declares: run outputs and sealed collections under
`~/.local/share/agent-multi/`, tool-owned durable state (watchdogs, incident
ledger, transition queue, audit snapshots, preflight payloads) under
`~/.local/state/agent-multi/`. Inside the repository, [`records/`](records/)
holds only a fleet runtime manifest and [`knowledge/`](knowledge/) only a
hashed concept bundle — neither is an evidence root.

## Safety and credentials

Training data are local CSV files; no credentials are required for local
experiments. Published fleet examples use documentation-reserved addresses;
deployed routes, runtime overlays and credentials remain outside version
control. Never commit hostnames, private IP addresses, API keys, broker account
identifiers or account fingerprints into experiment configs. See
[`SECURITY.md`](SECURITY.md) for the public-repository boundary and pre-push
check.
All performance figures produced here are simulation results and do not
guarantee live performance.

## Limitations and migration notes

- No `python_requires` or upper dependency bounds are declared in packaging
  metadata.
- Top-level package names (`app`, `*_plugins`) are shared conventions across
  sibling repositories; install agent-multi in its own environment (or run
  from the repository root) so its packages are not shadowed.
- The `pipeline.plugins`, `optimizer.plugins` and `preprocessor.plugins`
  group names are also used by sibling packages (predictor, gym-fx,
  preprocessor); entry-point names are unique per group within this package,
  but co-installation mixes the groups' contents.
- Historical experiment reports under [`docs/`](docs/) describe the campaigns
  they belong to; always prefer the work-plan corpus in
  [`docs/work_plan/`](docs/work_plan/) for current architecture.

## Related repositories

- [gym-fx](https://github.com/harveybc/gym-fx) — trading environment consumed by `gym_fx_env`
- [doin-node](https://github.com/harveybc/doin-node) — unified DOIN participant runtime supervised by this repo's campaign supervisor
- [doin-core](https://github.com/harveybc/doin-core) — shared decentralized protocol primitives and plugin ABCs
- [doin-plugins](https://github.com/harveybc/doin-plugins) — DOIN adapter plugins (`trading_asset`, `trading_scenario`) around this repo's pipeline
- [trading-contracts](https://github.com/harveybc/trading-contracts) — shared trading data contracts (install dependency)
- [predictor](https://github.com/harveybc/predictor) — phased deep-learning prediction platform
- Deeper docs: [`docs/work_plan/05_DOIN_TRADING_DOMAIN_INTEGRATION.md`](docs/work_plan/05_DOIN_TRADING_DOMAIN_INTEGRATION.md)

## License

This repository does not currently include a LICENSE file; no license terms
are published. Contact the owner before reusing the code.

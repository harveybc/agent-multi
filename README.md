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

**Lifecycle: ACTIVE-CORE.** This is the experiment platform behind the owner's
current trading-research campaigns; its pipelines, evidence tooling and audit
corpus are in daily use.

> **Disclaimer:** training and evaluation run in simulation/backtest over
> historical or synthetic data through gym-fx. This repository places no
> real-capital trades; live or demo execution belongs to separate downstream
> systems. Nothing in the examples is financial advice.

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
configs at 1000 timesteps (also unverified here). What was verified:

```bash
python -c "from pipeline_plugins.rl_pipeline_with_validation import PipelinePlugin; print('rl_pipeline_with_validation OK')"
# observed: "rl_pipeline_with_validation OK"
```

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
python -m pytest tests --collect-only -q   # observed: "898 tests collected in 3.19s"
python -m pytest tests                     # full run: unverified for this README
```

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

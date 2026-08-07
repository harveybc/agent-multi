# Musashi to General Satoshi III: SAC Inner-Curriculum and Normal-Fine-Tuning Order

Date: 2026-08-07 America/Bogota
Author: General Musashi, independent auditor and experimental lead
Recipient: General Satoshi III, technical lead
Owner authority: the owner explicitly approved testing easy pretraining inside
SAC followed by carefully controlled normal fine-tuning
Priority: P0 Front 1, immediately after the completed D1 packet
Runtime authority: Paper/Demo only; no Live-capital authority

## 1. Role and Required Standard

Act simultaneously as:

- a senior reinforcement-learning scientist with expertise in SAC, off-policy
  replay, distribution shift, catastrophic forgetting and curriculum learning;
- a quantitative-trading researcher preserving chronological causality,
  transaction costs, activity, drawdown and mandatory SL/TP;
- a senior Python/plugin architect preserving repository ownership and the
  local-first optimizer contract;
- a distributed-experiment engineer capable of coordinating four heterogeneous
  GPUs without duplicated work or ambiguous artifacts; and
- a forensic evidence engineer who distinguishes a trained terminal policy
  from an unchanged warm-start checkpoint.

Do not return another conceptual roadmap. Implement the bounded work below,
execute the authorized mechanism screen, preserve the evidence and return one
audit packet. Do not ask the owner to approve individual files, tests or the M0
launch; this order carries that approval.

Use codebase-memory MCP for code discovery when available. Index and query
`agent-multi` and `gym-fx` first; use `rg` for config values, logs and Markdown.
Do not repeatedly read entire repositories.

## 2. Mission and Correct Mental Model

Determine whether easy training is useful as an **inner SAC weight-learning
curriculum**, and build the smallest normal fine-tuning mechanism that retains
an active learned policy under the unchanged normal validation contract.

The question is not whether DOIN has many or few genes. DOIN currently chooses
candidate hyperparameters, but every candidate contains a SAC policy with
millions of learned weights. Difficulty acts on those inner weights,
transitions, replay and gradients. Unlike NEAT, SAC is off-policy and
gradient-trained; an abrupt dynamics change, fresh replay and new optimizer can
erase useful behavior even when the easy policy survived.

The immediate deliverable is a four-seed, four-arm **mechanism screen**, not a
new broad DOIN campaign. If the mechanism survives, measured parameters become
typed genes in the later R3 SAC dynamics domain. If it fails, the evidence
directs R3 and cannot be hidden by selecting the original anchor.

## 3. Established Facts: Do Not Rediscover or Misstate Them

### 3.1 Exact data and observation contract

- ETHUSDT, 4-hour bars.
- Dataset SHA-256:
  `1b447c66e68495e826c53e2ab2b08ecd3922c8fdc735747628f8d0435ebe440f`.
- 18,085 rows.
- Train: 13,699 bars, 2017-09-28 04:00 through 2023-12-31.
- Validation: 2,196 bars, calendar 2024.
- Disclosed test: 2,190 bars, calendar 2025; **disabled and forbidden for
  selection, tuning or this mechanism screen**.
- Observation: 83 features over 32 bars, 32-price window, 32-return window and
  four agent-state values; 2,724 flattened elements.
- Rolling scaling window: 256 bars.

The 32/256 values are currently used, not scientifically frozen. Do not change
them here because that would confound the SAC transition diagnosis.

### 3.2 Exact completed D1 evidence

Authoritative root:

```text
/home/harveybc/.local/share/agent-multi/
  eth_curriculum_decision_20260807_v2/
```

Key files:

```text
decision_summary.json
fleet_manifest.json
fleet_preflight.json
seed101/seed_packet.json
seed202/seed_packet.json
seed303/seed_packet.json
seed404/seed_packet.json
```

Evidence hashes:

```text
decision_summary.json
  3f3eeb940b04317c3bcc976a7e6bb230b38ce8ab6d23cdd6212701f9f9f85239
fleet_manifest.json
  0f39d7e8e9e7c8d6a9fb007e8ca166950f4d335e2f79bf4284fcf13f7993c6e2
```

D1 completed all 12 arms and all 20 replica checks. Raw 2024 validation:

| Seeds | Mean weekly | Annualized | Total | Max DD | Trades |
| --- | ---: | ---: | ---: | ---: | ---: |
| 101, 404 | +0.0513556% | +2.70870% | +2.71308% | 2.66335% | 136 |
| 202, 303 | -0.0393064% | -2.11349% | -2.11683% | 3.77031% | 130 |

Those values were identical for selected `N14`, `EN4_10` and diagnostic `E4`
inside each seed. That was not equal successful learning:

- selected `N14` and `EN4_10` fell back to the warm-start anchor;
- all 14 post-anchor normal epochs in `N14` failed activity;
- all 10 normal epochs after easy in `EN4_10` failed activity;
- all eight normal-trained terminal artifacts produced zero validation trades
  and zero return;
- `E4` retained 130-136 normal-validation trades after easy but did not improve
  the anchor's raw metrics; and
- margin/recapitalization telemetry was unavailable, so D1 did not prove that
  would-margin-call continuation caused the retained activity.

Do not call this an easy failure, easy superiority, or a successful tie. The
measured defect is **activity collapse during normal SAC updates**.

### 3.3 What the code already does

Easy is already inside SAC training. Do not build a second outer easy layer.

`pipeline_plugins/rl_pipeline_with_solvency_curriculum.py` currently:

1. loads the shared SAC anchor;
2. trains weights under `easy_chronological_continuation`;
3. saves a `post_easy` artifact that must remain active under normal probes;
4. reloads it through `agent_plugins.sac_agent.Plugin.load_for_training`;
5. constructs a new SAC from the normal candidate config;
6. transfers policy and compatible entropy state, but not optimizer moments;
7. starts normal training with a fresh replay buffer; and
8. selects only under normal validation.

D1 used learning rate `1e-4`, batch 256, replay capacity 200,000, learning
starts 1,000, gamma 0.99, tau 0.005, train frequency 1, gradient steps 1,
fixed entropy coefficient 0.2, action threshold 0.1 and 20,000 timesteps per
epoch.

The `execution_cost_curriculum_epochs` field does not prove that the cost
wrapper was active. D1 configs contain no `execution_cost_curriculum` contract.
Treat D1 as a binary easy/normal contract until direct facts prove otherwise.

## 4. Required Reading and Ownership Map

Read in this order and list the functions actually inspected in the delivery:

1. `docs/work_plan/33_ETH_DECISION_RESEARCH_AND_MULTI_ASSET_ROADMAP.md`,
   sections 3, 4 D1-D5, 5 R0/R3 and 12.
2. `docs/work_plan/19_EXECUTION_CURRICULUM_AND_ORDER_ROUTING.md`, sections 3
   and 15-16.
3. `tools/eth_curriculum_decision_experiment.py`: `_execution_id`,
   `_base_config`, `_make_anchor`, `run_arm`, artifact publication and packet
   construction.
4. `tools/eth_curriculum_fleet.py` and
   `tools/aggregate_curriculum_decision.py`.
5. `pipeline_plugins/rl_pipeline_with_solvency_curriculum.py`:
   `_easy_training_config`, `_easy_probe`, `_train_easy_phase`, `run_pipeline`.
6. `pipeline_plugins/rl_pipeline_with_validation.py`:
   `_set_env_training_progress`, `_training_progress_for_epoch`,
   `_eval_on_split`, epoch loop, checkpoint history and artifact publication.
7. `agent_plugins/sac_agent.py`: `build`, `load_for_training`, replay and
   entropy behavior.
8. `agent_plugins/project3_sac_actor_critic_agent.py`.
9. `gym-fx/app/bt_bridge.py`: normal termination,
   `_continue_after_would_margin_call`, economic debt and diagnostics.
10. `env_plugins/execution_cost_curriculum.py` and
    `pipeline_plugins/rl_pipeline_with_execution_curriculum.py`; prove whether
    they were in D1's call path.
11. `tests/unit/test_solvency_curriculum_pipeline.py`,
    `tests/test_decision_experiment_contract.py`,
    `tests/unit/test_sac_warm_start_expansion.py`, and relevant `gym-fx` tests.

Ownership:

- SAC construction/transfer/training: `agent-multi`.
- Easy/normal pipeline and evidence: `agent-multi`.
- Margin dynamics and direct environment telemetry: `gym-fx`.
- Local optimizer and typed genes: external `agent-multi` plugin.
- Migration/pooling: existing DOIN repositories; **do not modify them for M0**.
- DOIN node JSON changes happen only after the local optimizer passes.

## 5. Mandatory Knowledge Check Before Editing

Create before the first code commit:

```text
docs/audits/evidence/eth_sac_inner_curriculum/
  SATOSHI_III_CODE_PATH_FACTS_2026_08_07.json
```

With source references, state:

- where easy weights are trained and saved;
- where normal SAC is reconstructed;
- whether actor, critic, target critics, entropy state, optimizer moments and
  replay survive the boundary;
- exact D1 normal learning rate and entropy mode;
- whether the execution-cost wrapper was active;
- why the selected checkpoint can equal the anchor while terminal is inactive;
- which facts distinguish raw-action collapse from threshold/order rejection;
  and
- whether direct zero margin events differ from unavailable telemetry.

Trace any unknown before editing. Do not ask the owner what code can answer.

## 6. WP0: Reproduce and Diagnose D1 Without Training

Implement a read-only collector, suggested path:

```text
tools/eth_sac_training_diagnostics.py
```

Produce:

```text
docs/audits/evidence/eth_sac_inner_curriculum/
  D1_TRAINING_COLLAPSE_DIAGNOSTICS_2026_08_07.json
  D1_TRAINING_COLLAPSE_EPOCHS_2026_08_07.csv
```

For every seed/arm/epoch, report available direct facts:

- phase, epoch, cumulative timesteps and replay before/after;
- configured/observed learning rate;
- entropy mode, coefficient, target and loss;
- actor/critic parameter sums and epoch deltas;
- actor loss, critic loss, entropy loss and update count;
- raw-action mean/std/min/max, hold/non-hold and dominant-action rate on fixed
  normal validation observations;
- thresholded actions, entries, protected submissions/rejections, closes and
  order families;
- train-tail/validation trades;
- weekly, annualized and total return and drawdown with units;
- actual commission, spread, slippage and financing;
- termination, would-margin-call and recapitalization facts;
- anchor, post-easy, best and terminal hashes; and
- exact checkpoint eligibility reason.

Classify missing fields as `not_instrumented`, `not_applicable`,
`source_unavailable`, `direct_zero` or `invalid`. Never turn missing into zero.
Never display the lexicographic transport scalar as performance.

Answer explicitly:

1. Did raw actions collapse, or did threshold/order logic suppress them?
2. Did collapse begin in the first normal epoch in every seed?
3. Did post-easy differ from anchor in tensors and action trace?
4. Was fixed entropy 0.2 actually used?
5. Did any easy episode approach/cross normal margin termination?

WP0 is CPU/read-only and starts immediately. Do not mutate D1 evidence.

## 7. WP1: Instrument the Boundary Before Rerunning

Primary files:

```text
pipeline_plugins/rl_pipeline_with_solvency_curriculum.py
pipeline_plugins/rl_pipeline_with_validation.py
agent_plugins/sac_agent.py
```

Touch `gym-fx/app/bt_bridge.py` only if telemetry cannot be propagated without
changing dynamics; any `gym-fx` change needs its own tests and commit.

Required boundary evidence:

- source/target artifact SHA-256;
- policy tensor hash before and immediately after transfer;
- actor/critic/target-critic distances;
- source/target entropy modes/values;
- `optimizer_state_transferred=false`;
- `replay_transitions_transferred=0` for M0;
- target actor/critic optimizer learning rates;
- replay size before normal collection, after learning starts and at end;
- gradient update count and actor/critic/entropy losses;
- raw/thresholded action diagnostics at boundary and after every epoch;
- model hash/load proof for every phase artifact; and
- direct normal evaluation metrics after each phase.

Do not carry easy replay in M0. Do not add actor freezing, custom replay, KL
penalties or a new SAC algorithm. Add typed validation for every new field;
reject booleans-as-numbers, NaN/infinity, nonpositive rates, invalid
multipliers, unknown phases, negative epochs, unequal compute and defaults.

## 8. WP2: Implement the M0 Local Mechanism Screen

Use separate files; do not rewrite D1 records:

```text
tools/eth_sac_inner_curriculum_screen.py
tools/eth_sac_inner_curriculum_fleet.py
tools/aggregate_eth_sac_inner_curriculum.py
examples/config/phase_3_eth_sac_dynamics/m0_contract.json
tests/test_eth_sac_inner_curriculum_contract.py
tests/unit/test_sac_normal_finetune.py
```

Reuse validated helpers. Copying the full D1 runner and allowing drift is
forbidden. Preserve D1 compatibility if extracting common code.

### 8.1 Frozen M0 factors

Each worker owns one D1 seed/anchor and runs all arms sequentially:

| Arm | Schedule | Easy LR | Normal LR | Total updates |
| --- | --- | ---: | ---: | ---: |
| `N2_LR1` | 2 normal epochs | N/A | 1e-4 | 40,000 |
| `E1_N1_LR1` | 1 easy + 1 normal | 1e-4 | 1e-4 | 40,000 |
| `E1_N1_LR03` | 1 easy + 1 normal | 1e-4 | 3e-5 | 40,000 |
| `E1_N1_LR01` | 1 easy + 1 normal | 1e-4 | 1e-5 | 40,000 |

Everything else remains D1-identical. Normal uses fresh replay. Each epoch is
20,000 timesteps. `N2` uses two normal epochs for equal compute.

This tests only:

1. whether easy changes survival relative to equal-compute normal-only; and
2. whether lower normal fine-tuning LR prevents first-epoch collapse.

Do not add replay retention, actor freezing, synthetic data, feature changes,
autoencoders, alternate objectives or 2025 evaluation.

### 8.2 Anchors, workers and output

Use exact D1 anchors and verify hashes:

```text
seed101 -> omega RTX 4070 Laptop, exact GPU UUID
seed202 -> dragon RTX 4090 Laptop, exact GPU UUID
seed303 -> gamma RTX 5070 Ti Laptop, exact GPU UUID
seed404 -> gamma RTX 5090, exact GPU UUID
```

Output:

```text
~/.local/share/agent-multi/eth_sac_inner_curriculum_m0_20260807_v1/
```

Every identity binds full code revisions, data/anchor hash, seed, GPU UUID,
resolved config, schedule, update budget, artifacts, raw traces and replica
observations. Replica topology:

```text
omega -> dragon
dragon -> gamma-replica
gamma-5070ti -> dragon-replica
gamma-5090 -> dragon-replica
```

Namespace replicas by experiment/seed/arm; no collisions.

### 8.3 M0 decision facts and rules

Report per phase/terminal:

- raw metrics;
- `activity_survived_normal`;
- `weights_changed_from_anchor` and tensor/action distances;
- `post_easy_activity` and `terminal_activity`;
- `normal_updates_applied`;
- `anchor_selected_as_best`; and
- `terminal_usable`, never inferred from best-checkpoint success.

Mechanism survival requires, for the same arm in at least 3/4 seeds:

- terminal validation trades > 0;
- non-hold raw actions and nonzero action dispersion;
- a protected entry and no unprotected entry;
- no evaluation error and a loadable terminal artifact;
- terminal weights differ from anchor;
- normal gradient updates occurred; and
- decision facts come from normal validation.

No positive-profit gate exists at M0.

Interpretation:

- `N2` fails and reduced-LR E/N survives: supports inner easy plus gentle normal
  fine-tuning.
- `N2` survives and E/N fails: easy handoff is harmful.
- all fail: proceed to R3 diagnosis, not larger curriculum confirmation.
- all survive without meaningful difference: easy adds no demonstrated value;
  retain `easy_epochs=0` control.
- no margin events: do not attribute the result to solvency relaxation.

## 9. WP3: Fleet Orchestration and No-Idle Successor

Before launch: commit/push, run tests, sync exact clean revisions, prove every
SHA/data/anchor, replica SSH and GPU UUID, and write a preflight packet.

Launch all four workers together using user systemd services and guardians that
distinguish completed seed packets from crashes. Status must expose host, seed,
GPU/UUID, temperature, utilization, VRAM, PID binding, arm/phase/epoch,
completed/active/remaining, elapsed/ETA, replicas, recoveries and queue totals.
The queue contains 16 arm executions.

M0 is local paired research and writes no blockchain. Never resume rejected
`full-v2` as its successor.

Materialize both deterministic successor branches before M0 finishes:

1. `mechanism_pass`: M1 four-seed equal-compute confirmation with winning
   normal-LR region and `N14` control;
2. `mechanism_fail`: R0/R3 collapse localization driven by raw action, entropy,
   reward and critic evidence, not another repeated N14/EN4_10 run.

Do not launch an unmeasured full DOIN campaign just to heat GPUs. Select only
the branch justified by complete M0 evidence. Maximum transition gap: 15 min.

## 10. WP4: Conditional M1 Confirmation

Run only if one E/N arm passes M0. Use four seeds, 14 total epochs per arm,
20,000 timesteps/epoch and the same 2024 normal validation. Compare at minimum:

- `N14` control;
- original `E4_N10` with normal multiplier 1.0; and
- `E4_N10` with the M0-supported multiplier.

Any fourth arm requires M0 evidence. Do not invent graded-cost/replay factors.
Terminal models remain first-class evidence. Anchor fallback is not success.

## 11. WP5: Later R3 Local Optimizer and DOIN Adapter

Do not edit `doin-node`. After M0/M1 establishes bounds:

1. extend the local `agent-multi` SAC dynamics optimizer;
2. keep it runnable without DOIN;
3. add typed conditional genes supported by evidence;
4. expose the existing thin optimizer plugin through DOIN config; and
5. materialize synchronized node JSON files for one chain/pool.

Candidate genes, after evidence:

```text
easy_epochs_gene                    integer including 0
normal_finetune_lr_multiplier_gene  log-float with M0-derived bounds
normal_epochs                       derived = fixed_total - easy_epochs
```

No replay gene without its own experiment; no actor-freeze/KL/custom replay
genes without local fixtures. Inactive genes cannot leak defaults.

Relevant paths:

```text
optimizer_plugins/project3_full_genome_optimizer.py
tools/project3_full_genome_config.py
examples/config/phase_3_eth_sac_dynamics/optimization/
examples/campaigns/phase_3_eth_sac_dynamics_fleet_v1/
```

## 12. Mandatory Tests

At minimum prove:

1. easy is train-only and all evaluations force normal;
2. 2025 cannot enter M0/M1 selection/traces;
3. D1 records remain loadable/unchanged;
4. normal LR reaches actor and critic optimizers;
5. policy tensors match immediately after transfer;
6. optimizer moments are not transferred;
7. replay is empty at M0 boundary and fills from normal only;
8. equal total timesteps across M0 arms;
9. invalid rates/multipliers/schedules fail before model construction;
10. raw versus thresholded collapse is distinguishable;
11. direct-zero margin events differ from unavailable;
12. every entry has mandatory SL/TP;
13. inactive terminal cannot be usable because anchor traded;
14. hashes/load proof/replica evidence are mandatory;
15. duplicate IDs/packets/replica paths fail;
16. Git abbreviation cannot create false lineage drift;
17. wrong GPU UUID/PID binding fails;
18. restart reuses only complete compatible arms;
19. guardians do not relaunch completed seeds; and
20. the interpretation selects exactly one successor branch.

Run focused tests, full `agent-multi`, relevant `gym-fx` if changed, one local
mechanical smoke, then four-worker M0. Report exact pass counts and failures.

## 13. Stop Conditions

Stop/preserve evidence if test rows leak, an entry lacks SL/TP, an artifact is
missing/unloadable, identity changes, duplicate work/replica collision occurs,
GPU binding is wrong, unavailable becomes zero, anchor is represented as
terminal learning, an arm changes undeclared factors, or defaults are hidden.

Do not stop merely because return is negative. This is a mechanism test.

## 14. Required Delivery

Return:

```text
docs/handoffs/
  SATOSHI_III_SAC_INNER_CURRICULUM_M0_DELIVERY_2026_08_07.md
```

Include:

1. exact clean/pushed commits;
2. knowledge-check result;
3. D1 collapse reproduction;
4. diff map by WP;
5. all test commands/results;
6. fleet preflight and GPU map;
7. 16-arm status/duration;
8. raw per-seed metrics with weekly/annual/total units;
9. boundary/action/loss/entropy/replay/margin telemetry;
10. all model/config/data/code hashes and replicas;
11. M0 interpretation;
12. selected successor and materialized job identity;
13. unresolved facts without optimistic substitution; and
14. explicit request for Musashi audit.

Do not close findings you implemented. Provide files, hashes and commands, not
screenshots or prose-only claims.

## 15. Final Direction

The owner's NEAT lesson can apply at the lower SAC level. Easy already lives at
that level, but normal continuation currently destroys activity. Do not add
another layer. Make the handoff measurable, test gentle normal fine-tuning, and
expose only proven controls to later DOIN optimization. Preserve negative
results and never let an unchanged anchor impersonate trained terminal weights.

Begin WP0/WP1 now. Launch M0 on all four GPUs immediately after tests and
preflight pass.

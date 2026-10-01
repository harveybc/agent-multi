# RL temporal-representation lane (lane G): inventory of what exists

Written 2026-10-01 by lane G (RL owner) under M05, from the dispatch at predictor
master b327b771 §3/§4.1/§6 and the RL subplan. Facts below were read from the
checkouts named; anything not run is marked UNVERIFIED.

## Installed RL stack (environment `trading-stack`, both workers and the coordinator)

| Component | Fact |
| --- | --- |
| stable-baselines3 | 2.9.0 (torch 2.13.0+cu130, gymnasium 1.3.0) |
| Keras in the RL env | 3.15.0 / TF 2.21.0; the campaign env (`tensorflow`) holds Keras 3.13.2 / TF 2.21.0 |
| SAC | `agent-multi/agent_plugins/sac_agent.py`: SB3 SAC `MlpPolicy`; needs gym-fx `action_space_mode: continuous` = `Box(-1, 1, (1,))`; the ENV thresholds the scalar at ±`continuous_action_threshold` (0.33 default; contracts `legacy_directional_v1`/v2) into {long, hold, short}; Dict obs flattened by `FlattenObservation`. Tests: `tests/unit/test_sac_warm_start_expansion.py`, `test_sac_normal_finetune.py` (11 passed), plus the L1/M0 suites. `project3_sac_actor_critic_agent.py` = thin subclass with Project 3 defaults. |
| DQN | `agent_plugins/dqn_agent.py`: SB3 DQN `MlpPolicy` on gym-fx default `Discrete(3)` {0 hold, 1 long, 2 short}; FlattenObservation. Before this lane NO test built or trained it (only strings "dqn" absent in a plugin). Now exercised by RL03/RL04/RL05/RL06 (build, learn, target sync, save/reload in a fresh process). |
| discrete SAC | NOT installed (not in SB3 2.9.0, not in any sibling repo). Consequence: no shared-discrete SAC-vs-DQN comparison; the action-space difference is published in every config (`action_space_note`). |
| PPO | `agent_plugins/ppo_agent.py` (not in the four arms) |
| env plugin | `agent-multi/env_plugins/gym_fx_env.py` -> `gym_fx.build_environment` with entry-point bundles data_feed/broker/strategy/preprocessor/reward/metrics; optional `ExecutionCostCurriculumWrapper` |
| gym-fx (origin/master 1eeecb7) | `app/env.py` GymFxEnv (backtrader thread bridge `app/bt_bridge.py`); termination causes `data_end`, `min_equity`, `external_stop`; `flatten_step()` closes through the same path as the policy (action 3 + `force_flat_request`); solvency modes `normal_realistic` / `easy_chronological_continuation`. Preprocessors: `default_preprocessor` (price window), `feature_window_preprocessor` (`features (window, F)` z-scored from strictly past rows + agent state). Rewards `pnl_reward`, `sharpe_reward`, `dd_penalized_reward`; metrics `default_metrics`, `trading_metrics`. |
| cross-lane fact | gym-fx origin/master lacks `app/observation_builder.py`; the `ObservationContract` (feature_order, block_layout, digest) lives UNMERGED on `origin/satoshi/wp21-observation-spec-20260925` (e287ea2). This lane derives the layout from the Dict space in agent-multi (`rl_temporal/observation_layout.py`) and does not merge that branch. |
| entry-point hazard | `preprocessor.plugins` lists `default_preprocessor` twice (gym-fx and a sibling); in `trading-stack` the editable finder maps `agent_plugins` to a live `.runtime/am-sac-v2-slot0-*` checkout. Every run sets PYTHONPATH to the lane's trees first and the tests assert resolution to their checkout (`tests/rl_temporal/_mechanisms.assert_checkout_resolution`). |

## How observations are built today (the "native baseline" representation)

`feature_window_preprocessor.make_observation` emits `{"features": (W, F) float32, "position",
"equity_norm", "unrealized_pnl_norm", "steps_remaining_norm"}` (+ `prices`/`returns` only if
`include_price_window`). `FlattenObservation` concatenates blocks in the Dict's sorted key order
(`equity_norm, features, position, steps_remaining_norm, unrealized_pnl_norm`). SB3's
`FlattenExtractor` hands that vector to the policy MLP (`net_arch`). Time is handled by position in
the vector only: no convolution, attention or bottleneck. That is the control (RL-S0/RL-D0),
declared in `rl_temporal.modular_torch.NativeFlatExtractor` and `native_baseline_card()`, and it is
never labelled as the branching architecture. `pipeline_plugins/_observation_contract.py` is the
fail-closed declaration layer (`UNDECLARED_OBSERVATION_CONTRACT`, `UNWAIVED_RAW_OBSERVATION`).

## Modular temporal representation (predictor engine pin 3ecdb256)

`predictor_plugins/modular_temporal/` (Keras): `FeatureSelect` gather routing; branch = causal
Conv1D (16 ch, k=3, exact GELU) keeping all 24 steps; fusion = channel concat; core = sinusoidal PE,
Dense d_model 64, 2 causal MHA blocks (4 heads, key_dim 16, LayerNorm eps 1e-3, FFN 128), residual
Conv1D stages 24->12->6->6 channels 32->16->8 (factors [2,2,1], valid padding, matched residual
projection); head = Flatten + Dense. `bundle.py` save/load refuses a Keras major.minor mismatch.
`representation_candidate_card.v1` schema: predictor `satoshi/c-contracts-20261001` f509955f.
Donors: M02's run produces ECL (electricity) donors; NO ETH-task donors exist, so RL R1/R2 arms are
BLOCKED_NO_COMPATIBLE_DONOR and only R0 is runnable (never relabelled).

The torch port used by the policies: `agent-multi/rl_temporal/modular_torch.py`
(`ModularTemporalEncoder`, `ModularTemporalExtractor(BaseFeaturesExtractor)`), same contract and the
Keras defaults that matter for parity. Fidelity proven on worker_a: Keras 3.13.2 export (predictor
`tools/export_modular_encoder_npz.py`, branch `satoshi/g-rl-encoder-export-20261001` 600626c8, run in the
pinned env) -> torch import: 63 arrays, 0 unmatched, fused atol 1e-5, latent atol 1e-4
(`docs/audits/evidence/g_rl_20261001/fixture_encoder_export.npz.receipt.json`).

## Champion / DOIN / campaign tooling (reusable, not modified)

- `app/campaign_supervisor.py` + `doin-node` node configs -> `optimizer_plugins/project3_full_genome_optimizer.py` (DEAP GA over `trading_experiment.v1` configs).
- `pipeline_plugins/rl_pipeline_with_validation.py` (2,200 lines): epoch loop `epoch_timesteps` x `max_epochs`, L1 patience on validation composite, activity patience, best checkpoint saved via the agent plugin and RESTORED for final evaluation (line ~1985), baseline evals, actor-liveness probe, compute contract (`_compute_contract.py`, `_n_updates` meaning measured per algorithm on SB3 2.9.0).
- `pipeline_plugins/_nested_splits.py`: `ContextPrefixWrapper` (forced hold over the scaler/window context; refuses account mutation) — reused by `rl_temporal.monitoring.chronological_validation_factory`.
- `pipeline_plugins/_system_config.py`: system manifests with source-tree identity; `tools/l1_fleet_launcher.py`, `collect_l1_factorial.py`, `aggregate_*`, `eth_curriculum_*` (fleet dispatch, sealed collection, replica verification).
- `agent_plugins/_progress_callback.py`: step-interval JSON progress (not time-based); the time-based heartbeat (<=60 s) is `rl_temporal.monitoring.HeartbeatCallback`.

## What this lane added (agent-multi `satoshi/g-rl-temporal-20261001`)

`rl_temporal/`: `observation_layout`, `lake_binding` (manifest binding, resource sha, DRAFT refusal,
causal probe), `modular_torch`, `keras_import`, `consumers` (policy_kwargs from a JSON
`representation` block; `build_model` through the existing plugins), `donor_contract` (R0/R1/R2,
identity, ownership, update counting), `checkpoint` (policy bundle v1), `monitoring` (heartbeat,
hard limits, validation early stopping with restore, chronological validation factory),
`reconciliation`, `warehouse_binding`, `arms` (four arms, pairing check, accounting).
Agent plugins: one hook each in `sac_agent.py`/`dqn_agent.py` (`if config.get("representation")`),
behaviour unchanged otherwise (65 installed tests still pass). Tests: `tests/rl_temporal/` 42.
Configs: `examples/config/rl_temporal/eth_4h_task.json` + 16 materialized cells bound to lane B's
DRAFT manifest sha 22ae7730 (`real_data_fit_allowed: false`).

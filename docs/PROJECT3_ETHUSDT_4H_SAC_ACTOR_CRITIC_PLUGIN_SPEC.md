# Project 3 ETHUSDT 4h SAC Actor-Critic Trading Agent Spec

## Purpose

Implement a Project 3 inspired actor-critic trading agent in `agent-multi` for ETHUSDT 4h trading simulation. The agent should use the Project 3 exported ETHUSDT 4h data and should be compatible with the existing `agent-multi` / `gym-fx` plugin architecture.

This spec is meant to be handed to a Copilot or Hermes coding agent. It is intentionally detailed so the agent can implement, test, and report without guessing.

## Source Context

Canonical Project 3 export:

- Model-ready CSV:
  `/home/harveybc/Documents/GitHub/predictor/examples/data/project3/ethusdt_4h_tech_stat_full_model_ready.csv`
- Warmup-with-NaNs CSV:
  `/home/harveybc/Documents/GitHub/predictor/examples/data/project3/ethusdt_4h_tech_stat_full_with_warmup_nans.csv`
- Exact Project 3 config bundle:
  `/home/harveybc/Documents/GitHub/predictor/examples/data/project3/ethusdt_4h_sac_tech_stat_full_config.json`
- Source run:
  `ethusdt_4h_sac_tech_stat_direct_atr_sltp_s0_20260502T051413Z_project3_stage31_firstwave`

Observed preliminary Stage A result:

- Algorithm: SAC / Soft Actor-Critic
- Asset: ETHUSDT spot
- Timeframe: 4h
- Feature preset label: `tech_stat`
- Strategy: `direct_atr_sltp`
- Seed: `0`
- Total return: `0.1512164417774693`
- Sharpe ratio: `0.011535593140210989`
- Max drawdown: `11.113226854446845%`
- Trades: `426`
- Status: promising screening signal, not production validated

## Critical Finding

The current Project 3 Stage A run used `features_preset=tech_stat`, and the input CSV contains the full `tech_stat` feature matrix. However, the current `gym-fx` `default_preprocessor` does not consume all feature columns. It builds observations from:

- sliding `CLOSE` price window
- close price deltas from that window
- current position
- normalized equity
- normalized unrealized PnL
- normalized remaining steps

Therefore there are two valid implementation paths:

1. **Exact reproduction mode:** reuse the current `default_preprocessor` behavior. This is closest to what the Stage A SAC run actually observed.
2. **Feature-aware mode:** implement a new preprocessor that consumes the exported `tech_stat` feature columns as intended. This is recommended for the new plugin because the user specifically wants the supplied feature matrix used as model input.

Do not claim that the current default Project 3 run fully used every `tech_stat` column unless the new feature-aware preprocessor is implemented and tested.

## Existing Architecture

Repo:

`/home/harveybc/Documents/GitHub/agent-multi`

Important files:

- `agent_plugins/sac_agent.py`
- `agent_plugins/ppo_agent.py`
- `agent_plugins/dqn_agent.py`
- `env_plugins/gym_fx_env.py`
- `pipeline_plugins/rl_pipeline.py`
- `setup.py`

External environment repo:

`/home/harveybc/Documents/GitHub/gym-fx`

Important files:

- `app/env.py`
- `preprocessor_plugins/default_preprocessor.py`
- `strategy_plugins/direct_atr_sltp.py`
- `reward_plugins/pnl_reward.py`
- `data_feed_plugins/default_data_feed.py`

Current plugin contract for `agent-multi` agent plugins:

```python
class Plugin:
    plugin_params: dict
    def __init__(self, config=None): ...
    def set_params(self, **kwargs): ...
    def build(self, env, config): return model
    def train(self, model, config): return model
    def predict(self, model, obs, deterministic=True): return action
    def save(self, model, path): ...
    def load(self, path, env): return model
    def fitness(self, summary, config): return float
    def hparam_schema(self): ...
    # optional:
    def wrap_env(self, env, config=None): ...
```

Current `sac_agent.py` already implements a Stable-Baselines3 SAC wrapper. A new Project 3 specific plugin should reuse that logic where possible instead of duplicating it.

## Target Implementation

### Recommended file additions

In `agent-multi`:

- `agent_plugins/project3_sac_actor_critic_agent.py`
- `examples/config/project3_ethusdt_4h_sac_actor_critic.json`
- `tools/project3_smoke_ethusdt_4h_sac.py`
- `docs/PROJECT3_ETHUSDT_4H_SAC_ACTOR_CRITIC_PLUGIN_SPEC.md` already exists as this spec

Optional, if implementing feature-aware observations in `gym-fx`:

- `/home/harveybc/Documents/GitHub/gym-fx/preprocessor_plugins/feature_window_preprocessor.py`
- update `/home/harveybc/Documents/GitHub/gym-fx/setup.py` entry points:
  `feature_window_preprocessor=preprocessor_plugins.feature_window_preprocessor:Plugin`

### Plugin name

Use:

`project3_sac_actor_critic_agent`

Add to `agent-multi/setup.py`:

```python
"project3_sac_actor_critic_agent=agent_plugins.project3_sac_actor_critic_agent:Plugin",
```

The plugin may be a thin subclass/composition wrapper around the existing `sac_agent.Plugin`, but it must expose Project 3 defaults and validation.

## Algorithm Specification

Use Stable-Baselines3 SAC:

- Policy: `MlpPolicy`
- Action space: continuous `Box(-1.0, +1.0, shape=(1,))`
- Environment maps continuous actions to trade decisions:
  - action >= `continuous_action_threshold` -> long
  - action <= `-continuous_action_threshold` -> short
  - otherwise -> hold
- Default threshold for this spec: `0.1`

Default hyperparameters from Project 3 best preliminary run:

```json
{
  "agent_plugin": "project3_sac_actor_critic_agent",
  "base_algorithm": "stable_baselines3.SAC",
  "policy": "MlpPolicy",
  "total_timesteps": 25000,
  "learning_rate": 0.0001,
  "buffer_size": 200000,
  "learning_starts": 5000,
  "batch_size": 256,
  "tau": 0.005,
  "gamma": 0.99,
  "train_freq": 1,
  "gradient_steps": 1,
  "ent_coef": "auto",
  "target_update_interval": 1,
  "target_entropy": "auto",
  "use_sde": true,
  "net_arch": [256, 256],
  "device": "cuda",
  "train_seed": 0,
  "eval_seed": 0
}
```

The agent must call `model.set_random_seed(seed)` after construction, matching the existing fixed behavior in `sac_agent.py`.

If `use_sde=true`, attempt `model.policy.reset_noise()` after seeding.

## Trading Environment Specification

Use:

```json
{
  "env_plugin": "gym_fx_env",
  "pipeline_plugin": "rl_pipeline",
  "optimizer_plugin": "default_optimizer",
  "data_feed_plugin": "default_data_feed",
  "broker_plugin": "default_broker",
  "strategy_plugin": "direct_atr_sltp",
  "reward_plugin": "pnl_reward",
  "metrics_plugin": "default_metrics",
  "env_mode": "training",
  "mode": "train",
  "action_space_mode": "continuous",
  "continuous_action_threshold": 0.1,
  "initial_cash": 10000.0,
  "commission": 0.0002,
  "slippage": 0.0,
  "leverage": 1.0,
  "rel_volume": 0.05,
  "size_mode": "notional",
  "min_order_volume": 0.0,
  "max_order_volume": 100.0,
  "position_size": 0.01,
  "atr_period": 14,
  "k_sl": 2.0,
  "k_tp": 3.0
}
```

Strategy behavior:

- `direct_atr_sltp` opens bracket orders.
- Stop loss distance: `k_sl * ATR(atr_period)`.
- Take profit distance: `k_tp * ATR(atr_period)`.
- With `rel_volume=0.05`, `size_mode=notional`, and `leverage=1.0`, size is approximately:
  `cash * rel_volume / close`, clamped to `[min_order_volume, max_order_volume]`.
- The strategy skips orders until ATR is warmed and valid.

Reward behavior:

- `pnl_reward` computes:
  `(new_equity - prev_equity) / initial_cash * reward_scale`
- Default `reward_scale=1.0`.

## Data Specification

Input CSV:

`/home/harveybc/Documents/GitHub/predictor/examples/data/project3/ethusdt_4h_tech_stat_full_model_ready.csv`

Properties:

- Rows: `18,085`
- Columns: `90`
- Date range: `2017-09-28 04:00:00` to `2025-12-31 20:00:00`
- Periodicity: `4h`
- Missing cells: `0`
- Duplicate timestamps: `0`
- Price column: `CLOSE`
- Date column: `DATE_TIME`
- Target/reference column for external supervised experiments: `typical_price`

Required OHLCV/time columns:

```text
DATE_TIME, typical_price, OPEN, HIGH, LOW, CLOSE, VOLUME
```

Full feature columns available in the CSV:

```text
return_1, log_return_1, return_5, log_return_5, return_10, log_return_10,
return_20, log_return_20, return_60, log_return_60,
sma_10, ema_10, close_sma_ratio_10,
sma_20, ema_20, close_sma_ratio_20,
sma_50, ema_50, close_sma_ratio_50,
sma_100, ema_100, close_sma_ratio_100,
sma_200, ema_200, close_sma_ratio_200,
macd, macd_signal, macd_hist,
rsi_7, rsi_14, rsi_21,
stoch_k, stoch_d, williams_r_14, cci_14,
roc_10, roc_20, roc_60, mom_10, mom_20,
bb_upper, bb_middle, bb_lower, bb_pct_b, bb_width,
atr_14, natr_14,
hist_vol_10, hist_vol_20, hist_vol_60,
ema_cross_10_50, ema_cross_20_100,
trend_slope_50, trend_strength_50,
obv, obv_delta_20,
volume_sma_10, volume_sma_20, volume_ratio_20, vwap_60, mfi_14,
statistical__log_return_1,
roll_mean_ret_20, roll_std_ret_20, roll_skew_ret_20, roll_kurt_ret_20,
roll_mean_ret_60, roll_std_ret_60, roll_skew_ret_60, roll_kurt_ret_60,
roll_mean_ret_252, roll_std_ret_252, roll_skew_ret_252, roll_kurt_ret_252,
realized_var_12, realized_var_48,
autocorr_lag1_100, autocorr_lag5_100, sqret_autocorr_lag1_100,
vol_regime_high, vol_regime_low, hurst_proxy_200, zscore_close_100
```

## Preprocessing Specification

### Mode A: exact current reproduction

Use:

```json
{
  "preprocessor_plugin": "default_preprocessor",
  "window_size": 32,
  "price_column": "CLOSE"
}
```

Observation produced:

- `prices`: last 32 closes
- `returns`: first-difference of last 32 closes, not percentage returns
- `position`: current discrete position in `[-1, 0, 1]`
- `equity_norm`: `(equity - initial_cash) / initial_cash`
- `unrealized_pnl_norm`: rough position PnL proxy normalized by initial cash
- `steps_remaining_norm`: remaining bars fraction

The SAC plugin flattens this dict observation through `FlattenObservation`.

This mode is valid for reproducing current Project 3 Stage A behavior, but it does not use the exported `tech_stat` columns.

### Mode B: recommended feature-aware implementation

Implement `feature_window_preprocessor` in `gym-fx` and use:

```json
{
  "preprocessor_plugin": "feature_window_preprocessor",
  "window_size": 32,
  "price_column": "CLOSE",
  "feature_columns": [
    "return_1",
    "log_return_1",
    "... all feature columns listed above ...",
    "zscore_close_100"
  ],
  "feature_scaling": "rolling_or_train_only_standard",
  "include_price_window": true,
  "include_agent_state": true
}
```

Feature-aware observation should be a dict:

```python
{
    "features": np.ndarray(shape=(window_size, n_features), dtype=np.float32),
    "prices": np.ndarray(shape=(window_size,), dtype=np.float32),  # optional but useful
    "returns": np.ndarray(shape=(window_size,), dtype=np.float32), # optional reproduction context
    "position": np.ndarray(shape=(1,), dtype=np.float32),
    "equity_norm": np.ndarray(shape=(1,), dtype=np.float32),
    "unrealized_pnl_norm": np.ndarray(shape=(1,), dtype=np.float32),
    "steps_remaining_norm": np.ndarray(shape=(1,), dtype=np.float32)
}
```

The SAC plugin can continue using `FlattenObservation`, so no custom neural architecture is required for the first version.

Scaling rules:

- Do not fit scalers on validation/test/held-out rows.
- For this external trading-simulation plugin, if no split is provided, use deterministic expanding/rolling scaling or a documented train-only fit.
- Never use future rows to normalize past rows.
- Preserve binary columns (`ema_cross_10_50`, `ema_cross_20_100`, `vol_regime_high`, `vol_regime_low`) as 0/1 or signed flags unless a train-only scaler is explicitly desired.
- Validate no NaNs or infs in the observation.

Minimum feature-aware acceptance check:

- `features.shape == (32, 83)` if using the 83 feature columns after excluding date, typical price, and OHLCV.
- Flattened observation should include `32 * 83 + optional price/return/state values`.

## Config File To Add

Create:

`examples/config/project3_ethusdt_4h_sac_actor_critic.json`

Suggested exact-reproduction config:

```json
{
  "env_plugin": "gym_fx_env",
  "agent_plugin": "project3_sac_actor_critic_agent",
  "pipeline_plugin": "rl_pipeline",
  "optimizer_plugin": "default_optimizer",
  "data_feed_plugin": "default_data_feed",
  "broker_plugin": "default_broker",
  "strategy_plugin": "direct_atr_sltp",
  "preprocessor_plugin": "default_preprocessor",
  "reward_plugin": "pnl_reward",
  "metrics_plugin": "default_metrics",
  "mode": "train",
  "env_mode": "training",
  "input_data_file": "/home/harveybc/Documents/GitHub/predictor/examples/data/project3/ethusdt_4h_tech_stat_full_model_ready.csv",
  "date_column": "DATE_TIME",
  "price_column": "CLOSE",
  "headers": true,
  "asset": "ethusdt_4h",
  "features_preset": "tech_stat",
  "action_space_mode": "continuous",
  "continuous_action_threshold": 0.1,
  "window_size": 32,
  "initial_cash": 10000.0,
  "commission": 0.0002,
  "slippage": 0.0,
  "position_size": 0.01,
  "rel_volume": 0.05,
  "leverage": 1.0,
  "min_order_volume": 0.0,
  "max_order_volume": 100.0,
  "size_mode": "notional",
  "atr_period": 14,
  "k_sl": 2.0,
  "k_tp": 3.0,
  "total_timesteps": 25000,
  "learning_rate": 0.0001,
  "buffer_size": 200000,
  "learning_starts": 5000,
  "batch_size": 256,
  "tau": 0.005,
  "gamma": 0.99,
  "train_freq": 1,
  "gradient_steps": 1,
  "ent_coef": "auto",
  "target_update_interval": 1,
  "target_entropy": "auto",
  "use_sde": true,
  "net_arch": [256, 256],
  "device": "cuda",
  "agent_verbose": 0,
  "train_seed": 0,
  "eval_seed": 0,
  "quiet_mode": true,
  "save_model": "./examples/results/project3_ethusdt_4h_sac_actor_critic/policy.zip",
  "results_file": "./examples/results/project3_ethusdt_4h_sac_actor_critic/summary.json",
  "save_config": "./examples/results/project3_ethusdt_4h_sac_actor_critic/config_out.json"
}
```

If implementing feature-aware mode, change:

```json
{
  "preprocessor_plugin": "feature_window_preprocessor",
  "feature_columns": ["return_1", "log_return_1", "...", "zscore_close_100"],
  "feature_scaling": "rolling_or_train_only_standard",
  "include_price_window": true,
  "include_agent_state": true
}
```

## Implementation Steps

1. Inspect current `agent_plugins/sac_agent.py`.
2. Create `agent_plugins/project3_sac_actor_critic_agent.py`.
3. Reuse or subclass the existing SAC plugin.
4. Set Project 3 defaults in `plugin_params`.
5. Add explicit validation:
   - `action_space_mode == "continuous"`
   - `window_size == 32` unless intentionally overridden
   - `price_column == "CLOSE"` unless intentionally overridden
   - `strategy_plugin == "direct_atr_sltp"` for reproduction runs
   - `reward_plugin == "pnl_reward"` for reproduction runs
6. Keep `wrap_env()` behavior: flatten dict observations.
7. Register the entry point in `setup.py`.
8. Add example config.
9. Add a smoke tool that runs a tiny training job:
   - use the same config
   - override `total_timesteps=100`
   - write results under `examples/results/project3_ethusdt_4h_sac_actor_critic_smoke/`
10. Run the smoke test.
11. Report:
   - files changed
   - smoke command
   - summary path
   - whether observation includes all feature columns or only reproduction mode

## DEAP / Optimizer Search Space

If tuning with DEAP, begin with a small, safe search:

SAC hyperparameters:

```text
learning_rate: log-uniform [1e-5, 5e-4]
batch_size: integer choice [128, 256, 384, 512]
gamma: float [0.95, 0.999]
tau: float [0.001, 0.02]
train_freq: integer [1, 8]
gradient_steps: integer [1, 8]
ent_coef: choice ["auto", 0.01, 0.03, 0.1]
use_sde: choice [true, false]
net_arch: choice [[128, 128], [256, 256], [256, 256, 128]]
```

Trading/strategy hyperparameters:

```text
continuous_action_threshold: float [0.05, 0.35]
rel_volume: float [0.01, 0.10]
k_sl: float [1.0, 4.0]
k_tp: float [1.5, 6.0]
atr_period: integer [7, 30]
commission: fixed 0.0002 for baseline; also test 0.0004 pessimistic
slippage: fixed 0.0 for reproduction; also test 0.0002 pessimistic
```

Fitness should not be raw return only. Use:

```text
fitness = total_return
          - 1.0 * max_drawdown_pct / 100
          - 0.0001 * trades_total
          - penalty_if_no_trades
```

Promotion-style checks:

- positive net return
- does not collapse under pessimistic cost
- trades are not zero
- max drawdown acceptable
- result is not one-seed only

## Smoke Commands

After implementation:

```bash
cd /home/harveybc/Documents/GitHub/agent-multi
pip install -e .
pip install -e /home/harveybc/Documents/GitHub/gym-fx
agent-multi --load_config examples/config/project3_ethusdt_4h_sac_actor_critic.json --total_timesteps 100 --quiet_mode
```

Full reproduction run:

```bash
cd /home/harveybc/Documents/GitHub/agent-multi
agent-multi --load_config examples/config/project3_ethusdt_4h_sac_actor_critic.json --quiet_mode
```

Seed sweep example:

```bash
cd /home/harveybc/Documents/GitHub/agent-multi
python tools/seed_sweep.py \
  --config examples/config/project3_ethusdt_4h_sac_actor_critic.json \
  --seeds 0 1 2 \
  --log_root examples/results/project3_ethusdt_4h_sac_actor_critic_seed_sweep \
  --run_tag project3_external_actor_critic
```

## Acceptance Criteria

The implementation is accepted only if:

1. `agent-multi` can load `project3_sac_actor_critic_agent` through entry points.
2. The example config runs a 100 timestep smoke without crashing.
3. The output `summary.json` contains at least:
   - `total_return`
   - `final_equity`
   - `max_drawdown_pct`
   - `sharpe_ratio`
   - `trades_total`
   - `episode_reward`
   - `episode_length`
4. `config_out.json` records the exact runtime config.
5. The implementation states whether it ran:
   - exact reproduction mode, or
   - feature-aware mode.
6. If feature-aware mode is implemented, a test verifies the observation consumes all configured feature columns.
7. No Stage C / held-out Project 3 claims are made from this external experiment.

## Risks And Guardrails

- Current Stage A result is a screening signal, not proof of a production edge.
- Do not tune on the full 2017-2025 file and then claim held-out performance.
- Do not call the exported `tech_stat` features "used by the policy" unless the feature-aware preprocessor is active.
- Do not fit scalers on future rows.
- Do not evaluate only zero-cost results.
- Log every config and seed.

## Prompt For A Coding Agent

Paste this prompt to the Copilot or Hermes coding agent:

```text
You are working in /home/harveybc/Documents/GitHub/agent-multi.

Implement a Project 3 ETHUSDT 4h SAC actor-critic trading agent plugin according to docs/PROJECT3_ETHUSDT_4H_SAC_ACTOR_CRITIC_PLUGIN_SPEC.md.

Read these files before editing:
- agent_plugins/sac_agent.py
- agent_plugins/__init__.py
- env_plugins/gym_fx_env.py
- pipeline_plugins/rl_pipeline.py
- setup.py
- /home/harveybc/Documents/GitHub/gym-fx/app/env.py
- /home/harveybc/Documents/GitHub/gym-fx/preprocessor_plugins/default_preprocessor.py
- /home/harveybc/Documents/GitHub/gym-fx/strategy_plugins/direct_atr_sltp.py
- /home/harveybc/Documents/GitHub/gym-fx/reward_plugins/pnl_reward.py

Goal:
- Add agent_plugins/project3_sac_actor_critic_agent.py as a Project 3 default wrapper around Stable-Baselines3 SAC.
- Register it in setup.py as agent plugin project3_sac_actor_critic_agent.
- Add examples/config/project3_ethusdt_4h_sac_actor_critic.json using the CSV at /home/harveybc/Documents/GitHub/predictor/examples/data/project3/ethusdt_4h_tech_stat_full_model_ready.csv.
- Add a smoke command or small tool that runs total_timesteps=100.

Important:
- The existing default_preprocessor only uses CLOSE window, price deltas, and agent state. It does not consume all tech_stat columns.
- First implement exact reproduction mode with default_preprocessor unless asked to implement feature-aware mode.
- If you implement feature-aware mode, add a gym-fx preprocessor plugin and tests proving the feature matrix is part of the observation.
- Do not change unrelated files.
- Do not remove user changes.
- Do not make production performance claims.

After implementation:
- Run pip install -e . in agent-multi if needed.
- Run a 100-step smoke test.
- Report changed files, command run, summary path, and whether exact reproduction or feature-aware mode was implemented.
```


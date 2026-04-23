# Part III – P4 / P5.5 Results Report

_Status snapshot as of 2026-04-23, compiled from multi-host training sweeps on Dragon (RTX 4090),
Gamma (RTX 5070 Ti), and Omega (RTX 4070)._

This report consolidates findings from the Part III RL training program targeting the
`direct_atr_sltp` strategy across BTCUSDT 1h, ETHUSDT 1h, and EURUSD 1h, using PPO, SAC, and
DQN plugins from `agent-multi`. All runs use the `twelve_atr` feature set (EURUSD adds
`macro` features) and the mandatory SL+TP bracket with ATR-scaled distances.

## 1. Harness

- Launchers: `tools/seed_sweep.py` (in-process, 3 seeds) and per-host nohup groups under
  `~/p4_launch/J*_*.out`.
- Run artifacts: `logs/partIII/<asset>_<algo>_<features>_<strategy>_s<seed>_<utc>_<tag>/`
  containing `summary.json`, `policy.zip`, `config.json`, `train.log`, `git_sha.txt`.
- Aggregators:
  - Train-set scoreboard: [logs/partIII/p4_aggregate.md](../logs/partIII/p4_aggregate.md)
  - Hold-out (d6) leaderboard: [logs/partIII/p5_eval_holdout.md](../logs/partIII/p5_eval_holdout.md)
- Sizing: `fx_units` for EURUSD (`cash*rel_volume*leverage`); `notional` for BTC/ETH.
  All trades enter with both SL and TP brackets.

## 2. Iteration timeline

Tags are incremental and always appear in the run-id folder name.

| Tag | Scope | Primary change |
|---|---|---|
| `p4` | All assets, seeds 0-2 | Baseline PPO configs; EURUSD passive (1 trade total). |
| `p4fix` | All assets, seeds 0-2 | Unblocked trading: reward/action wiring; EURUSD `rel_volume=0.10` caused blow-ups. |
| `p4iter2` | All assets | EURUSD `rel_volume 0.10 → 0.01` (commit 914e9c7); BTC/ETH commission 0.0001→0.0002. |
| `p4iter3` | EURUSD | `commission 0.0005`, reduced `ent_coef` (0.02→0.01). |
| `p4iter4` | BTC/ETH seeds 3-5; EURUSD | Extra seeds to confirm variance, EURUSD tuning. |
| `p4iter5` | BTC/ETH seeds 6-8; EURUSD | Seeds 6-8 on Dragon; best train-set numbers observed. |
| `p4iter6` | BTC/ETH seeds 9-11; EURUSD, SAC ETH seeds 3-5 | EURUSD `ent_coef=0.005` produced bi-modal behaviour (0 / 28 / 4061 trades). |
| `p4iter7` | BTC/ETH seeds 0-2; EURUSD | EURUSD `ent_coef=0.01` still bi-modal; BTC/ETH rerun. |
| `p4iter8` | EURUSD | `k_sl 2.0→2.5`, `k_tp 3.0→4.0` widened brackets → s2 +5.0%/112tr; others neutral. |
| `p4iter9` | EURUSD seeds 3-5 | Seed variance high: s3 −5.8%/303tr, s4 −6.6%/594tr, s5 +0.9%/18tr. |
| `p4iter10` | EURUSD seeds 0-2 (running) | `learning_rate 3e-4 → 1e-4` to damp variance. |
| `p4iter7` (BTC/ETH) | Dragon (running) | Widened SL/TP (`k_sl=2.5`, `k_tp=4.0`) applied to BTC/ETH, matching EURUSD iter8 recipe. |
| `p4iter6` (SAC ETH) | Gamma (running) | SAC ETH seeds 6-8; launched before gSDE seed fix. |

## 3. Best observed results

### 3.1 Train set (d0-d5 in-sample)

| Run | Asset | Algo | Tag | Seed | Return | Trades | Sharpe |
|---|---|---|---|---|---|---|---|
| eurusd iter8 s2 | EURUSD 1h | PPO | p4iter8 | 2 | **+5.0%** | 112 | positive |
| btc iter5 s7 | BTCUSDT 1h | PPO | p4iter5 | 7 | **+2.7%** | 199 | positive |
| eth iter5 s7 | ETHUSDT 1h | PPO | p4iter5 | 7 | **+2.4%** | 321 | positive |
| sac_eth iter4 s2 | ETHUSDT 1h | SAC | p4iter4 | 2 | **+4.56%** | – | +0.064 |

### 3.2 Hold-out (d6, out-of-sample via `tools/p5_eval_holdout.py`)

Latest group aggregates (18-run leaderboard):

| Group | Best seed | Return | DD% | Sharpe | Trades | Final eq |
|---|---|---|---|---|---|---|
| SAC ETH iter5 | s4 | **+5.3%** | 4.48 | +0.005 | 693 | 10,530 |
| SAC BTC iter4 | s4 / s5 | **+4.11%** | 2.54 | – | – | – |
| PPO EURUSD iter2 | s1 | +0.74% | 15.05 | +0.010 | 712 | 10,742 |
| DQN BTC (all) | – | 0.00% | 0.00 | n/a | 0 | 10,000 |
| DQN ETH (all) | – | **−32.8%** mean | high | negative | – | – |

Key observations:

- EURUSD early tags (`p4`, `p4fix`) are unusable: either degenerate (0 trades) or blown out
  (−99% DD). Recovery begins at `p4iter2` after `rel_volume` reduction.
- EURUSD `p4iter5` is the best-behaved bulk group on hold-out (ret ≈ −0.09, DD ≈ 9–10%,
  trades 730–1049); iter8 widened brackets gave the single best train-set number but
  did not dominate the hold-out.
- SAC generalises meaningfully better than PPO/DQN on BTC/ETH hold-out, despite the
  gSDE seeding bug (see §4) undercounting effective seed diversity.
- DQN configurations as tuned are not viable: BTC takes no trades, ETH averages −33% on
  hold-out. Needs re-tuning (reward scaling, exploration schedule) or benching.

## 4. Bugs identified and fixes

### 4.1 EURUSD position sizing blow-up (fixed)

- Symptom: `p4fix` EURUSD runs returned −99%+ DD on both train and hold-out with only
  ~33 trades – each trade ≈ account-sized.
- Cause: `rel_volume=0.10` combined with `leverage=10` under `fx_units` sizing.
- Fix: `rel_volume 0.10 → 0.01` in
  [`examples/config/p4_ppo_eurusd_1h.json`](../examples/config/p4_ppo_eurusd_1h.json)
  (commit 914e9c7).
- Evidence: `p4iter2` onward DDs compress from ~99% to 10–30% while trade counts rise
  into the hundreds/thousands.

### 4.2 SAC gSDE seed bug (fixed, commit b1691e4)

- Symptom: SAC hold-out and train metrics duplicated across seeds — e.g. `s1 == s2`,
  `iter4 s3 == iter2 s1`, `s4 == s5 == iter2 s0`.
- Cause: `agent_plugins/sac_agent.py` passed `seed=int(p["train_seed"])` to the
  `SAC(...)` constructor, but SB3's constructor path does not fully re-seed the gSDE
  exploration noise sampler or the env/action_space RNGs for SAC.
- Fix: after construction, call
  ```python
  model.set_random_seed(seed)
  if p["use_sde"]:
      model.policy.reset_noise()
  ```
  which re-seeds PyTorch/numpy/action_space/env consistently and resamples the gSDE
  exploration matrix per seed.
- Validation plan: next SAC sweep launched post-b1691e4 should show seed-distinct
  metrics across the sweep.

### 4.3 EURUSD bi-modal policies with low entropy (open)

- `ent_coef=0.005` (`p4iter6`) produced (0, 28, 4061) trade counts across 3 seeds.
- `ent_coef=0.01` (`p4iter7`) still bi-modal.
- `p4iter8` (wider SL/TP) broke the pattern on s2 but s3-s5 in `p4iter9` showed
  −5.8% / −6.6% / +0.9% (range >7 pp).
- `p4iter10` reduces `learning_rate 3e-4 → 1e-4` to damp variance while keeping
  widened brackets (in-progress).

## 5. Causal chain validated so far

1. `rel_volume` × `leverage` × `size_mode` → per-trade risk envelope. Too large → DD runaway.
2. `commission` → minimum edge required per trade. 0.0001 over-trades; 0.0002-0.0005 closer
   to production-like.
3. `ent_coef` → exploration breadth. Too low → bi-modal collapse across seeds.
4. `k_sl` / `k_tp` ATR multipliers → trade frequency and win-rate. Widening from (2.0, 3.0)
   to (2.5, 4.0) reduced over-trading and produced the best single EURUSD seed.
5. `learning_rate` → seed variance in outcome space; reducing to 1e-4 is the current lever.

## 6. Hosts and current running jobs

| Host | GPU | Current job | Tag |
|---|---|---|---|
| Dragon | RTX 4090 | PPO BTC seeds 0-2, PPO ETH seeds 0-2 | `p4iter7` (wider SL/TP) |
| Gamma  | RTX 5070 Ti | SAC ETH seeds 6-8 (launched pre-fix) | `p4iter6` |
| Omega  | RTX 4070 (local) | PPO EURUSD seeds 0-2 | `p4iter10` |

## 7. Next actions

- Wait for Dragon `p4iter7` (BTC/ETH wider SL/TP) to finish; compare against `p4iter5` to
  confirm the EURUSD bracket-widening recipe transfers.
- When Omega `p4iter10` finishes: if variance damped, lock config; else revisit
  `n_steps`/`batch_size` or `clip_range`.
- Re-run SAC BTC + ETH (all seeds) on post-b1691e4 code to confirm metrics now differ
  across seeds, then re-run `tools/p5_eval_holdout.py` for a clean SAC leaderboard.
- DQN: either restructure reward scaling + exploration schedule, or shelve until PPO/SAC
  baselines are locked.
- Refresh `p4_aggregate.md` and `p5_eval_holdout.md` once iter7/iter10/SAC-post-fix
  complete.

## 8. Artefact pointers

- Configs: [`examples/config/p4_ppo_btc_1h.json`](../examples/config/p4_ppo_btc_1h.json),
  [`examples/config/p4_ppo_eth_1h.json`](../examples/config/p4_ppo_eth_1h.json),
  [`examples/config/p4_ppo_eurusd_1h.json`](../examples/config/p4_ppo_eurusd_1h.json),
  [`examples/config/sac_btc_1h_twelve_atr.json`](../examples/config/sac_btc_1h_twelve_atr.json),
  [`examples/config/sac_eth_1h_twelve_atr.json`](../examples/config/sac_eth_1h_twelve_atr.json).
- Strategy: [`gym-fx/strategy_plugins/direct_atr_sltp.py`](../../gym-fx/strategy_plugins/direct_atr_sltp.py).
- SAC plugin (post-fix): [`agent_plugins/sac_agent.py`](../agent_plugins/sac_agent.py).
- Hold-out evaluator: [`tools/p5_eval_holdout.py`](../tools/p5_eval_holdout.py).

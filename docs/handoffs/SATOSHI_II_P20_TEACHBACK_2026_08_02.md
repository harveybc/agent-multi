# P20 Teach-Back: Termination Provenance, Split Isolation, Recapitalization

Date: 2026-08-02 22:20 America/Bogota
From: Lieutenant Satoshi II, temporary technical lead
To: General Musashi, temporary independent auditor and research lead
Relay: Gran Loto Blanco, project owner
Status: engineering note for your acceptance gate; no code changed, no
runtime touched. Every claim below was read from code this session; the
one unknown is labeled with its cheapest inspection.

Your K1-K6 corrections are accepted in full. K2 in particular I verified
and it is worse than an inert tail — detailed in answer 3.

## Q1. Where each termination cause is created, and how it stays distinguishable

Current creation sites (all collapse into one boolean today — your K1):

| Cause | Site |
| --- | --- |
| insolvency (strategy thread) | `BTBridgeStrategy.next()` — `_is_broke()` at `gym-fx/app/bt_bridge.py:181-183` |
| insolvency (env thread) | `GymFxEnv.step()` — `new_equity <= self.min_equity` at `gym-fx/app/env.py:343` (duplicate check, different thread, same boolean) |
| data_end | `BTBridgeStrategy.stop()` at `bt_bridge.py:194-196` |
| engine exception | `GymFxEnv._run_cerebro()` at `env.py:529` |
| obs-wait timeout (30 s watchdog) | `GymFxEnv._wait_obs()` at `env.py:543` |
| external stop | `stop_requested` path, `bt_bridge.py:190-192` / `env.py:534` |

Design: `BTBridge` gains `termination_reason: str | None` set ONCE at each
site above with `data_end | min_equity | external_stop | safety_limit |
unknown` (engine exception and obs-timeout map to `safety_limit` with a
`detail` string; first writer wins, later writers never overwrite, so the
race between the two insolvency checks is decided by ledger order, not
luck). Propagation: `GymFxEnv._make_info()` (`env.py:756`) carries it per
step; `GymFxEnv.summary()` (`env.py:788`) carries reason plus your demanded
minimums (reason counts, episode coverage, start/end bars, minimum equity,
cash/equity at termination, split, mode, seed, epoch); the pipeline copies
the summary verbatim into candidate evidence (it already persists
summaries; additive keys survive), OLAP stores them as atomic facts. A
wrapper-transparency regression asserts the reason survives
`agent_plugin.wrap_env()` (`rl_pipeline_with_validation.py:584-586`).

## Q2. How train-only solvency config is fenced from every other environment

Grounded fact (your K3 confirmed): `_make_split_env()` at
`rl_pipeline_with_validation.py:579-583` deep-copies ONE `base_config` and
changes only `input_data_file` per split; the env plugin
(`agent-multi/env_plugins/gym_fx_env.py:_build_env_config`, line 68) merges
that config flat. A flat override therefore reaches every split today.

Design: the P20 dynamics live under a dedicated subtree
`environment.training_dynamics.*` that `GymFxEnv.__init__` applies ONLY
when the env is constructed with an explicit `mode="train"` argument —
a new required parameter, not an inferred one. `_make_split_env` passes
`mode` explicitly per split; train-tail/validation/test construction
hard-strips the subtree (`cfg.pop("training_dynamics")`) BEFORE `make_env`,
so even a mislabeled mode cannot see it. Enforcement is triple: (a) the
strip, (b) the mode gate, (c) a regression that resolves a config carrying
`training_dynamics` and asserts the train-tail/validation/test envs report
realistic `min_equity` and `termination_reason=min_equity` on an insolvency
fixture. Live/demo: the subtree exists only in the gym-fx/agent-multi
config family; LTS consumes no gym-fx config key (separate repo, separate
schema — `DemoExecutionConfig.from_dict` rejects unknown-shaped configs by
requiring its own keys), and the LTS caps are independently regression-
locked. A grep-level CI guard asserts `training_dynamics` never appears in
any `lts/` config.

## Q3. How synthetic recapitalization preserves the loss and forces liquidation

First, the K2 verification you demanded, from code: `_compute_size` at
`gym-fx/strategy_plugins/direct_atr_sltp.py:479-490` computes
`clamp(cash * rel_volume * leverage, min_order_volume, max_order_volume)`
from `broker.getcash()`. With negative cash the product is negative and
the clamp floor is `min_order_volume` — so with `min_order_volume = 0`
orders stop (your inert tail), but with a positive genome floor the agent
would keep emitting MINIMUM-SIZE orders sized from negative capital: not
inert, economically undefined. Either branch invalidates naive
`min_equity` lowering; Arm B must recapitalize, never trade on negative
cash.

Arm B design: on the insolvency event (detected at the existing sites),
inside one env-owned sequence: (1) force-close all positions at market
through the existing close path so the terminal loss is REALIZED into the
episode P&L, the reward at that step (reward derives from equity delta,
`env.py:335-341`) and the equity trace; (2) record an
`insolvency_liquidation` event fact (bar, equity trough, realized loss);
(3) inject `synthetic_recapitalization` restoring cash to the CONSTANT
initial capital — never compounding, so future `_compute_size` sees the
same base as episode start, which answers "without contaminating future
position sizing": size can never inflate beyond the declared base, and
every injection is a counted, hashed fact; (4) continue, `done=False`.
Loss preservation across your K4 layers: the training reward already took
the full negative delta before injection, and the injection step is
excluded from reward by construction (regression: reward sum over the
liquidation+recap pair equals the realized loss exactly); the summary
reports BOTH raw equity series and the economic series (equity minus
cumulative injections) with drawdown computed on the economic series;
L1 selection (`_selection_and_composite`, `rl_pipeline_with_validation.py:
244-270`) and L2 candidate fitness consume train-tail/validation summaries
produced by REALISTIC-solvency rollouts only (Q2 fencing), so no selected
number is recapitalization-contaminated.

## Q4. Chronological continuation without look-ahead, timestamp reuse or replay ambiguity

Continuation reuses the existing forward-only Backtrader cursor: the
liquidation and injection settle at bar t's close inside the same
`BTBridgeStrategy.next()` slot; the next decision occurs at bar t+1 via the
normal `action_ready` handshake (`bt_bridge.py:186-188`). No bar is
revisited, no timestamp reused, and the future-row mutation invariant
(doc 09 §2, existing CI gate) extends over the recap path as a regression.
Replay buffer: the liquidation transition stores
`(obs_t, a_t, r_t_with_full_loss, obs_t+1, done=False)`; the account
features already in the observation (`margin_closeout_percent`,
`margin_available_norm`, `env.py:584-588`, plus balance/equity features)
make the post-recap state fully observable, so SAC's bootstrap sees a
Markov state, not a hidden jump; additionally `steps_since_recapitalization`
enters the info dict (and optionally the observation, ablatable) so the
discontinuity is measurable. No episode boundary is faked: `done` stays
False because the MDP genuinely continues — the alternative (truncated=True
with reset-free continuation) breaks SB3's bootstrap semantics for
continuing states and is rejected.

## Q5. Arm C deterministic starts without validation/test influence

Start offsets are a pure function of train-split facts only:
`offset_i = sha256(candidate_seed || episode_index) mod
(train_end_bar - window_size - min_episode_bars)`, computed at env reset
from the TRAIN dataframe length (`env.py:134-136` loads the split file;
each split env sees only its own CSV per `_make_split_env`), before any
evaluation runs. Validation/test lengths, dates and outcomes never enter
the function; offsets and their hash inputs are persisted in the summary
so you can re-derive them. Train-tail/validation/test resets remain
chronological-from-start, untouched.

## Q6. Falsification metrics and the paired-seed decision rule

Primary metric: realistic-validation selection score — the existing
gap-penalized `train_validation_selection_score`
(`rl_pipeline_with_validation.py:244-270`) computed from
realistic-solvency rollouts, identical formula across arms. Paired design:
N=10 seed pairs per arm (same seeds across A/B/C), equal training
timesteps. Decision rule: exact Wilcoxon signed-rank on paired differences,
alpha 0.05, two comparisons with Holm correction:

- P20 SUPPORTED iff median(B-A) > 0 significant AND median(B-C) > 0
  significant (the second kills your K5 reset explanation);
- P20 FALSIFIED iff B-A not significant, or B-C not significant while
  C-A is (reset-location explains the effect);
- secondary gates reported, never selecting: episode coverage (bars
  experienced / bars available), training insolvency rate, validation
  drawdown (economic series), completed trades, action-collapse guard
  incidence. Positive profit is not an entry criterion in any arm.

## Q7. Tests proving legacy configs and the active campaign unchanged

1. Resolved-config hash regression: the deployed job-0 config resolves to a
   byte-identical canonical hash with the new code (no `training_dynamics`
   key present, no default injected).
2. Golden-episode replay: a fixed-seed short run without the subtree
   produces an identical step trace and summary (modulo the additive
   `termination_reason` field) before/after the change; summary additions
   are additive-only, existing keys byte-stable.
3. Absent-subtree behavior test: `min_equity` default and termination
   semantics identical to today (fixture reproducing env.py:343 behavior).
4. Full gym-fx and agent-multi suites green; the campaign itself is not
   redeployed — workers keep `agent-multi@6a7bf5a`/`gym-fx@40a5c84`
   lineage until a declared boundary, so the running chain cannot observe
   the change at all (runtime-lineage rule, successor prompt §8).

## Unknown (labeled, with cheapest inspection)

- Whether the deployed job-0 genome's `min_order_volume` is zero or
  positive (decides which K2 branch the current campaign sits in):
  `unknown` — cheapest inspection is reading the resolved config of one
  worker via its existing `/api/candidate` payload or the archived
  resolved-config artifact; no GPU, no restart. I will attach it to the
  instrumentation packet for `AUD-F1-20260802-059`.

## Sequencing Acknowledged

S0-S2 L0 corrections remain first: 053-058 are returned and await your
verification. Instrumentation for 059 (termination provenance, CPU-only,
no swarm/domain/broker action) prepares only after that queue clears your
blade. K6 accepted: nothing here delays or mutates job 0/job 1.

# RL temporal lane: cost-pilot plan (bounded, train-only, per resource class)

Status 2026-10-01: NOT STARTED and NOT STARTABLE. Start condition (owner, subplan): a FROZEN
selected-feature manifest for the task and its point-in-time lake resources registered and readable.
Lane B's ETH 4h manifest is `DRAFT_NOT_FROZEN` (sha 22ae7730; blockers B1 availability contract with
the owner as question 19, B2 lake parents outside the census / producer not temporally verified, B3
C127 split contract 0.6/0.2/0.2 by rows vs the task split, B4 variant B inner-validation freeze).
`rl_temporal.lake_binding.SelectedFeatureBinding.refuse_real_data_pilot()` and every cell's
`pilot_gate` enforce this; the matrix will be re-materialized against the FROZEN sha when lane B
publishes it. Until then: no real-data fit, no GPU.

## Resource classes (materially different)

| Class | Arms | What is measured |
| --- | --- | --- |
| C1 native, 83 features | RL-S0, RL-D0 | obs dim 1996 -> MLP [64,64]; SAC actor 132,098 + critics 2x132,097 params; DQN q_net ~128k |
| C2 modular R0, 83 branches | RL-S1, RL-D1 | extractor 173,632 params (83 branches x 64 + core), latent 48 + 4 extras -> MLP [64,64]; SAC heads 7,682 / 2x7,681; DQN q_net 7,747 |
| C3 modular R1/R2 | RL-S1/D1 secondary | BLOCKED: no ETH-task donor; priced from a measured donor receipt when M02 produces one |

Parameter/compute difference is part of the report, not hidden: the control carries its parameters in
the first MLP layer, the modular arm in the encoder.

## Pilot protocol (per class, per algorithm; one host at a time; CPU first)

1. Place on worker_a or worker_b only, under `$HOME/.local/bin/crispdm-run -m <cap>` with a fresh
   admission per child; first cap from the fixture measurement (suite peak 1.17 GB RSS at 3G) plus the
   full-data materialization estimate; never shrink a cap to get admitted. Scratch under
   `~/.local/state/scratch/g-rl/`.
2. Train-only: `epoch_timesteps 2000 x 2 epochs`, `learning_starts 2000`, validation evaluation
   enabled only to exercise the mechanism (1 chronological episode), heartbeat 30 s, hard limits
   `max_wall_s 1800`, `max_timesteps 4000`. Label `PILOT_NOT_A_RESULT`.
3. Measure: wall per 1k transitions, gradient updates (compute contract), peak RSS (cgroup), env
   step time, evaluation episode wall, checkpoint bytes. Record the receipt with
   `rl_temporal.warehouse_binding` fields and `resources.host_alias` = worker_a/worker_b.
4. Derive the campaign cap per class = 1.25 x measured peak; the GPU request (external 5090 preferred)
   only after the measured CPU pilot, through the existing admission envelopes.
5. ETA = measured runtime per epoch x `max_epochs` upper bound (200) separately from queue delay;
   early stopping (`l1_patience 20`) makes the bound an upper bound, not a forecast.

## Then: the finite paired matrix

4 arms x 4 paired seeds (101, 202, 303, 404) = 16 cells (R0). Same env/costs/balance/episodes/reward
within each contrast (`check_pairing` proves it); equal tuning budget per algorithm family. Metrics on
identical validation episodes: net return, max drawdown, Sharpe (per-bar, ddof=1, not annualized;
undefined cases named), turnover units, trades closed, exposure fraction; baselines: no-trade on the
same episodes; heuristic strategy only with its forecasts' naive-gate evidence, otherwise
UNAVAILABLE. Paired contrasts S1-S0 and D1-D0 reported per seed, not best-seed. Finalists go to M05's
LTS shadow/paper lane as `rl_temporal.policy_bundle.v1` + result record with
`execution_authorized: false`. No financial claim from fixtures or from a pilot.

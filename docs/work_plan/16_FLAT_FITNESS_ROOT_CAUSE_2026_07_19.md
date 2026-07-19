# Flat Fitness Root-Cause Report

Status: root cause confirmed; defective campaign stopped; corrected fleet prepared
Date: 2026-07-19
Affected job: `btcusdt-1h-kitchen-sink-guarded-sac-shared-v2`

## 1. Symptom

The BTC fitness plot did not record a single improvement. This was not credible
convergence. More than 170 evaluations with different DEAP genomes reduced to
only two fitness values, while the accepted champion remained exactly
`0.003845942437862551`.

The four workers shared the same seed, generation, population fingerprint,
chain and component revisions. Candidate genes and the resulting SB3 training
attributes were different. The shared pool and parameter plumbing were not the
cause.

## 2. Confirmed Cause

The canonical experiment declared `kitchen_sink_guarded`, and its dataset has
185 model columns, but both the canonical JSON and generated DOIN node JSON
selected `default_preprocessor`. That plugin consumes only:

- an unscaled `CLOSE` window;
- raw price differences;
- position, equity, unrealized PnL and remaining-step state.

It ignored all engineered input columns. Raw BTC levels then saturated the SAC
gSDE actor at its clipped latent mean `-2`. The deterministic action after the
policy squashing transform was consequently `tanh(-2) = -0.96402758` for every
train, train-tail and validation step. Action thresholds could not change an
always-short policy, and many training hyperparameters converged to the same
execution trace and fitness.

Direct artifact evidence from the accepted on-chain model:

| Split | Steps | Action min | Action max | Action std | Effective policy |
| --- | ---: | ---: | ---: | ---: | --- |
| Train | 26,246 | -0.96402758 | -0.96402758 | 0 | always short |
| Train tail | 169 | -0.96402758 | -0.96402758 | 0 | always short |
| Validation | 8,761 | -0.96402758 | -0.96402758 | 0 | always short |

The artifact SHA-256 is
`86f61616742f723f80c4e6a651f17ae86ff4e13ea1cf3c4c62166e5611607862`.
It is retained as diagnostic evidence and is not portfolio-eligible.

## 3. Corrective Contract

Every enriched Phase 1 asset policy now requires:

- `feature_window_preprocessor`;
- an explicit materialized list of every dataset feature column;
- causal `rolling_zscore` scaling with a 256-row history;
- a bounded normalized tensor with `feature_clip: 10.0`;
- `include_price_window: false` so raw levels cannot bypass scaling;
- explicit binary-feature passthrough;
- the normalized agent-state block;
- a fail-closed observation-contract validator.

`gym-fx` now derives its `observation_space` from the actual enabled blocks.
This fixed a second integration defect where `FlattenObservation` still
required stale `prices` and `returns` keys after raw prices were disabled.
Warm-up observations with insufficient scaling history use neutral continuous
values rather than clipped raw levels.

The corrected BTC environment was exercised directly:

| Diagnostic | Corrected value |
| --- | ---: |
| Feature columns | 185 |
| Flattened observation size | 5,924 (`32 x 185 + 4`) |
| Median feature mean over 513 steps | 0.0502 |
| Median feature standard deviation | 1.0188 |
| Median clipping rate | 0% |
| Untrained deterministic actions sampled | 512 |
| Distinct actions at 6 decimals | 510 |
| Action standard deviation | 0.0799 |
| Old saturated action occurrences | 0 |

## 4. Candidate Acceptance Guard

Every train-tail and validation rollout now records raw action mean, standard
deviation, range, dominant side and dominant-side rate. A candidate receives
fitness `-1e9` and `candidate_rejected_reason =
deterministic_policy_action_collapse` when all configured diagnostic splits:

- have at least 64 actions;
- have action standard deviation at or below `1e-5`; and
- place at least 99.9% of actions on one discrete side.

The rejection is a valid evaluated result, not a transport failure. It remains
visible in DOIN metrics and cannot become a champion artifact.

## 5. Historical Artifact Disposition

The prior SOL champions also used `default_preprocessor`, so they are not
evidence for their declared enriched profiles. Direct deterministic replay
shows that they are not constant policies:

| Artifact | Actions | Distinct at 6 decimals | Action std | Classification |
| --- | ---: | ---: | ---: | --- |
| SOLUSDT 4h | 2,190 | 2,183 | 0.3569 | usable `price_state_only` baseline |
| SOLUSDT 1h | 8,760 | 6,193 | 0.4316 | usable `price_state_only` baseline |

Their exact weights, hyperparameters and metrics remain useful and are not
discarded or repeated in the immediate queue. They must be loaded with their
historical `default_preprocessor` contract and cannot be labeled as feature-pack
champions. A future feature-aware SOL comparison is optional, not a prerequisite
for starting portfolio work.

## 6. Campaign Disposition

- `fleet_v4` BTC shared-v2 was stopped on all four workers and preserved.
- Its supervisors remained inactive; workers that ignored `SIGTERM` for 30
  seconds were stopped by process group after persistence was verified.
- No v2 state directory, blockchain, log or artifact was deleted.
- `phase_1_asset_policy_fleet_v5` starts fresh `shared-v3` domains for BTCUSDT
  1h, ADAUSDT 1h, EURUSD 4h and DOGEUSDT 4h.
- Generated node JSONs inherit the canonical preprocessor and pipeline plugins;
  stale node-level plugin overrides can no longer silently revert the input
  contract.

The corrected fleet cannot start unless every participant reports the same
plan hash, domain semantic hash, dataset hash, component revisions, genesis
lineage and generation-zero population fingerprint.

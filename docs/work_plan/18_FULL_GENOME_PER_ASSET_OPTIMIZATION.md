# 18. Full-Genome Per-Asset Optimization

Status: v1 implementation and local/distributed contract verified; first
USDCAD campaign materialized
Decision date: 2026-07-27

## 1. Purpose

Produce one reproducible, load-tested champion library for the selected
asset/timeframe cells before portfolio optimization. E0-E3 are retained as
data priors. E4 is retained as executable baseline evidence and warm-start
weights. Neither is repeated merely to rename results.

## 2. What E4 Proved and Did Not Prove

E4 proved:

- causal E3 contract materialization;
- train/validation/test split execution;
- SAC model save/load;
- exact artifact bytes, size and SHA-256;
- canonical return, RAP and drawdown metrics;
- distributed result collection in the evidence OLAP.

E4 fixed SAC to `[256, 256]`, fixed its training and execution settings, used
`max_epochs: 40`, `l1_patience: 8`, and represented the precomputed context as
one policy row. It therefore did not optimize the complete policy stack.

## 3. Executable Genome v1

The external `agent-multi` optimizer owns a typed, JSON-decodable genome. DOIN
uses its existing shared-population interface and does not own trading logic.

| Layer | Evolvable choices |
| --- | --- |
| Features | explicit feature-family masks over the asset's causally materialized wide frame |
| Preprocessing | none, rolling z-score or expanding z-score; history and clipping |
| Observation | 24/72/168/336/720 context hours converted to native bars |
| Model | executable SAC MLP architecture |
| Training | learning rate, batch, buffer, learning starts, gamma, tau, train frequency, gradient steps, entropy settings |
| Execution | action threshold, relative volume and ATR-based SL/TP |

Categorical and conditional genes are decoded through a versioned manifest.
Repair rules reject or normalize invalid, noncausal, unavailable and
resource-incompatible combinations before training. Seeds, protected dates,
minimum costs, causality rules and the L1 budget are evaluator facts, not
fitness-exploitable genes.

Data-bundle, publication-lag, staleness, cross-asset-set, learned encoder and
early-close genes remain required extensions, but they are materialized only
where E2 evidence and source coverage support real choices. They are not
represented by placeholder genes. In particular, E2 favored market context for
BTC-perp, GBPJPY and USDJPY and macro-market context for NZDUSD; those bundles
must be present before those assets enter the immutable campaign queue.
USDCAD favored `tech_stat_decomp` without external context and is therefore the
first valid v1 campaign.

## 4. Full-Fidelity L1 Contract

```json
{
  "max_epochs": 2000,
  "l1_patience": 60,
  "l1_patience_start_epoch": 40,
  "l1_min_delta_fractional_weekly_rap": 0.00001,
  "restore_best_checkpoint": true,
  "selection_split": "validation",
  "selection_uses_test": false
}
```

`epoch_timesteps` is derived from valid training transitions and recorded in
the resolved config. The patience counter cannot advance before both the
off-policy learning barrier and epoch 40. The best eligible checkpoint is
restored before final metrics and artifact hashing.

Ranking-only reduced-fidelity evaluations, when used, are stored under a
different protocol ID. They cannot publish or replace a full-fidelity
champion. A candidate promoted from a ranking stage must be retrained and
evaluated under this contract.

## 5. Staged Search

Each asset runs one sequential DOIN campaign using all available workers:

1. feature families, preprocessing and observation;
2. model architecture and training dynamics;
3. execution-cost curriculum and risk geometry in a distinct warm-started
   domain;
4. restricted joint refinement around the best prior stages;
5. final three-seed confirmation and frozen protected-test evaluation.

Stages preserve compatible best parameters. The first USDCAD campaign stops
after step 2 because its execution environment has free fills; optimizing
execution/risk genes there would teach the wrong objective. A hash-verified
champion then warm-starts step 3 under strictly positive, visible costs and
robust validation. This is a new domain and chain, never a mutation of the
active one.

## 6. Fitness and Metrics

DOIN receives one higher-is-better fitness during the shared search:

```text
mean(train-tail selection score, validation selection score)
  - gap_beta * abs(train-tail score - validation score)
  - declared action-collapse/no-trade penalties
```

This is `train_validation_l1_score`. Seed stability is calculated when the
frozen champion is confirmed across three seeds; protected-test output remains
outside search. The scalar never replaces the raw metric vector. Every
candidate stores, with
explicit split, unit and week count:

- mean weekly return and annual return;
- mean weekly RAP and annual RAP;
- drawdown and downside tail;
- trades, active weeks, turnover and costs;
- seed/subperiod stability;
- training epochs, steps, stopping reason and runtime.

The protected test is absent from candidate selection, patience, migration and
stage advancement. It is evaluated only for the frozen final candidate.

## 7. Champion Contract

No asset is ready for portfolio optimization until it has:

- load-tested Stable-Baselines3 `.zip` weights;
- encoded genome and fully decoded canonical JSON;
- exact feature/preprocessing/context manifest;
- source, dataset, code and protocol hashes;
- validation and frozen-test metric vectors;
- training history and stopping evidence;
- artifact SHA-256, byte size and format;
- DOIN chain/domain/campaign lineage.

## 8. Execution Order

1. Implement and unit-test the mixed genome and decoder.
2. Verify local optimization independently of DOIN on omega.
3. Materialize the first production per-asset config and four machine
   overlays.
4. Start one DOIN campaign; verify one chain, one stage and unique candidate
   claims across omega, dragon, gamma 5070 Ti and gamma 5090.
5. Run the selected assets sequentially and archive each champion.
6. Build the frozen per-asset library.
7. Begin portfolio optimization.
8. Add rush activation and weekly retraining/fine-tuning as separately
   measurable components.

Items 1-3 are complete for `USDCAD@4h`. The four generated worker configs
produce the same encoded 28-gene schema, initial population and population
fingerprint. Item 4 is the active deployment boundary.

## 9. Dataset Deployment Gate

Before any worker may create or join the campaign, the supervisor resolves the
machine overlay and verifies the dataset against its versioned manifest. Asset
identifiers are compared case-insensitively because source manifests use
lowercase directory identifiers while canonical trading configs use uppercase
instrument symbols. Timeframe remains an exact match, and the CSV must match
the manifest SHA-256 byte for byte. A mismatch blocks the campaign before a
chain or candidate claim can be created.

Gamma's two workers are isolated by physical NVIDIA GPU UUID in the campaign
profile. Each process sees exactly one CUDA device and addresses it as
`cuda:0`; this prevents both workers from silently selecting the faster eGPU
when the generic agent device is `cuda`.

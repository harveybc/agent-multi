# Grouped SAC Feature Extractor and Pretraining

Status: implemented infrastructure only; not yet a complete state-of-the-art
training system and no candidate architecture is promoted.

## Owner correction, 2026-08-26

The flat MLP is retired as a prospective champion architecture. It remains an
architectural baseline when a bounded, capacity/compute-declared comparison is
needed to measure the incremental value of temporal structure. Existing valid
MLP evidence must be reused first; a new MLP run requires the comparison to show
why inference from existing evidence is insufficient. It may not become the
default winner merely because stronger candidates fail a gate.

Data identity precedes topology. No architecture campaign may start until each
input family has both (a) causally timestamped historical coverage for every
declared fold and (b) a named real-time source available to the Demo trading
loop. A feature that cannot be reproduced live is not an eligible trading
input, regardless of its backtest value.

## Decision

The current `gym-fx` observation is a structured dictionary. Its `features`
block is `(32, 83)` and account state is emitted in separate scalar blocks.
Flattening this structure before SAC destroys both temporal position and
semantic feature identity. The grouped extractor keeps the dictionary and
assigns every feature to exactly one semantic family.

The first strong candidate is multibranch and preserves temporal and semantic
structure. The table names starting implementations, not universal winners:

| Family | Current variables | Baseline branch | Alternatives to test |
|---|---:|---|---|
| returns and momentum | 16 | PatchTST + causal TCN | iTransformer, GRU |
| trend and level | 23 | TFT-style variable selection + Transformer | TCN, GRU |
| bounded oscillators | 9 | GRU | causal TCN |
| volatility and distribution | 29 | TimesNet-style + causal TCN | PatchTST, GRU |
| volume and flow | 6 | causal TCN + GRU | Transformer |
| account and position state | 4 | MLP | gated MLP |

The MLP is appropriate only for the four non-temporal account/position scalars.
Fusion starts as typed gated fusion with cross-family attention. The SAC actor
and twin critics consume the fused latent but have separate configurable heads;
the action surface must represent target exposure and explicit close/hold
authority, while native SL/TP remains the hard protection envelope. Auxiliary
heads may predict multi-horizon return quantiles, volatility, barrier-hit
probability and regime, but none may use future information unavailable at the
decision timestamp.

DeepLOB is not a candidate for these inputs: it is designed around the spatial
structure of limit-order-book levels, which this H4 dataset does not contain.
Mamba is deferred until longer windows justify its linear long-sequence
advantage; a 32-token window does not.

## Configuration contract

`feature_extractor_plugin=grouped_features_extractor` switches SAC from
`MlpPolicy` plus `FlattenObservation` to `MultiInputPolicy`. The effective
architecture lives under `feature_extractor_config` and is recorded with the
experiment configuration. Precedence is:

1. component plugin defaults;
2. component `params` in the architecture;
3. the final experiment JSON supplied to the run.

Unknown nested parameters refuse. Every configured feature must belong to one
and only one branch. A new column therefore cannot silently enter a generic
"other" branch.

Materialize the baseline from the existing feature-aware experiment:

```bash
python tools/materialize_grouped_sac_config.py \
  --base examples/config/project3_ethusdt_4h_sac_actor_critic_feature_aware.json \
  --output examples/config/project3_ethusdt_4h_sac_grouped_features_v1.json
```

## Pretraining design

Pretraining is per semantic branch, using train-only observation windows
emitted under the same observation contract as RL. The current next-step Huber
and direction plugins are only a minimum implementation and are not sufficient
to claim the design is complete. The executing runner must support a declared
mixture of masked-patch reconstruction, hierarchical contrastive learning,
multi-horizon quantile returns, volatility and barrier-hit objectives.
Reconstruction alone is not authoritative because it may preserve
high-variance detail that has no value for trading.

Three arms are required before adopting pretraining:

1. strong grouped architecture trained end-to-end from random initialization;
2. independently pretrained branches, then end-to-end SAC fine-tuning;
3. shared multiscale temporal pretraining, then end-to-end SAC fine-tuning.

All arms keep the SAC controller, reward, splits, seed set and compute budget
fixed. The 2025 test remains sealed. Branch choice is first screened on
train/monitor and then decided on inner/outer validation. Pretraining artifacts
must bind feature names and order, window size, observation contract, data
hashes, objective plugin, topology, seed and source commit before loading.

## Search order

1. Materialize and validate the historical/live data-availability matrix.
2. Prove the grouped candidate can overfit a tiny train-only fixture and that
   gradients reach every branch.
3. Wire the pretraining objectives to an executing, artifact-producing runner.
4. Compare strong branch families independently with a cheap factorial; do not search
   fusion and branches simultaneously.
5. Freeze branch winners and compare gated fusion variants.
6. Compare no-pretraining, independent pretraining and shared pretraining.
7. Only then run DOIN topology and parameter optimization, with categorical
   genes for branch plugin names and bounded genes for their topology.
8. Run easy-to-normal only on a strong architecture whose easy dynamics are
   empirically active. The completed flat-MLP curriculum is archival evidence,
   not a control that justifies another long run.

## Research basis

- Bai, Kolter and Koltun, *An Empirical Evaluation of Generic Convolutional and
  Recurrent Networks for Sequence Modeling* (2018): TCN as a strong generic
  sequence baseline.
- Lim et al., *Temporal Fusion Transformers for Interpretable Multi-horizon
  Time Series Forecasting* (2021): typed covariates, gating and local/global
  temporal processing.
- Yue et al., *TS2Vec* (AAAI 2022): hierarchical contrastive representations at
  multiple temporal scales.
- Nie et al., *PatchTST* (2023): channel-independent patching and masked
  self-supervised pretraining.
- Wu et al., *TimesNet* (2023): explicit multi-period temporal variation.
- Zhang, Zohren and Roberts, *DeepLOB* (IEEE TSP 2019): CNN/LSTM evidence for
  order-book tensors, recorded here mainly to prevent applying it to the wrong
  data type.

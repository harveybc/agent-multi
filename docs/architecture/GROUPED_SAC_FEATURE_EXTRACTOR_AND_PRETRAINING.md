# Grouped SAC Feature Extractor and Pretraining

Status: implemented infrastructure; candidate architectures are not promoted.

## Decision

The current `gym-fx` observation is a structured dictionary. Its `features`
block is `(32, 83)` and account state is emitted in separate scalar blocks.
Flattening this structure before SAC destroys both temporal position and
semantic feature identity. The grouped extractor keeps the dictionary and
assigns every feature to exactly one semantic family.

The baseline is deliberately a comparison point, not a claim of superiority:

| Family | Current variables | Baseline branch | Alternatives to test |
|---|---:|---|---|
| returns and momentum | 16 | causal TCN | GRU, patch Transformer |
| trend and level | 23 | Transformer | TCN, GRU |
| bounded oscillators | 9 | GRU | small TCN, MLP |
| volatility and distribution | 29 | causal TCN | GRU, TimesNet-style branch |
| volume and flow | 6 | GRU | TCN, MLP |
| account and position state | 4 | MLP | gated MLP |

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
emitted under the same observation contract as RL. Initial objectives are
next-step Huber regression and direction classification. Reconstruction alone
is not authoritative because it may preserve high-variance input detail that
has no value for trading.

Three arms are required before adopting pretraining:

1. grouped architecture trained end-to-end from random initialization;
2. independently pretrained branches, then end-to-end SAC fine-tuning;
3. one shared temporal encoder, then end-to-end SAC fine-tuning.

All arms keep the SAC controller, reward, splits, seed set and compute budget
fixed. The 2025 test remains sealed. Branch choice is first screened on
train/monitor and then decided on inner/outer validation. Pretraining artifacts
must bind feature names and order, window size, observation contract, data
hashes, objective plugin, topology, seed and source commit before loading.

## Search order

1. Prove the grouped baseline can overfit a tiny train-only fixture and that
   gradients reach every branch.
2. Compare branch families independently with a cheap factorial; do not search
   fusion and branches simultaneously.
3. Freeze branch winners and compare concat versus gated fusion.
4. Compare no-pretraining, independent pretraining and shared pretraining.
5. Only then run DOIN topology and parameter optimization, with categorical
   genes for branch plugin names and bounded genes for their topology.
6. Re-run the accepted easy-to-normal SAC curriculum against the flat MLP
   control. Architecture work does not replace that causal experiment.

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


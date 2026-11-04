# Structural93 baseline versus specialized temporal attention

Owner authorization2026-11-04: fresh93-input fit on the secondary GPU; the
primary GPU adds a time-series Transformer after fusion and before the TCN
core. Keep the selected compact actor from200 unchanged as an inference reference.

## Literature and repository inspection

`predictor/predictor_plugins/predictor_plugin_tft.py` declares itself TFT-inspired:
GRNs, LSTM and multihead attention; optional positional encoding defaults off.
It is not a full reproduction of the variable-selection/static-covariate TFT.
Its attention has no causal mask. Since its forecasting head reads the end of
an already historical input window, this alone is NOT evidence of future leakage,
but it does not meet our stricter intermediate-prefix invariance contract.

- [TFT](https://arxiv.org/abs/1912.09363):2019 preprint,2021 journal publication;
  recurrent local processing, temporal attention and input-selection machinery.
- [PatchTST](https://arxiv.org/abs/2211.14730):ICLR2023, temporal patches and shared
  channel-independent Transformer representations. The
  [official implementation](https://github.com/yuqinie98/PatchTST/blob/main/PatchTST_supervised/layers/PatchTST_backbone.py)
  includes positional encoding and optional RevIN, enabled by default there.
- [TimeXer](https://arxiv.org/abs/2402.19072):NeurIPS2024; explicit endogenous/
  exogenous embeddings with patch self-attention and variate cross-attention.
  Our fused latent channels no longer preserve that variable distinction.
- [Chronos-2](https://github.com/amazon-science/chronos-forecasting):released2025,
  pretrained zero-shot univariate/multivariate/covariate forecasting.
- [TimesFM3](https://github.com/google-research/timesfm):2026 native multivariate
  pretrained forecasting; self-hosted pretrained weights have a separate
  noncommercial license. Not downloaded or inserted into this comparison.

Research decision: use a small causal, channel-independent patch-attention
adapter inspired by PatchTST, NOT claim a full PatchTST replication or the best
2026 model for RL. Foundation forecasting benchmarks do not establish which
module improves a13-tick discrete SAC representation. Other architectures can
be tested later without mislabelling a one-layer adapter as an entire forecaster.

## Fixed architecture and training contract

Both arms:93 real input channels,30 groups,13 observations at stride2; two
residual TCN blocks per extractor(width8), fusion to16 channels, two TCN core
blocks(width16), final dense8 and categorical NOP/BUY/SELL. Same encoder option
in actor and twin critics. Fresh seed7 weights/optimizer/replay; no transplantation
of the selected policy or edits to its immutable artifact.

Variant, at the exact fusion/core boundary:

```text
grouped TCNs -> fusion + SiLU [B,16,13]
 -> shared trailing patches(length3,stride1) per fused channel
 -> shared Linear(3,8) + fixed sinusoidal positions [B*16,13,8]
 -> one causal TransformerEncoderLayer(heads2,FF16,dropout0,GELU,pre-LayerNorm)
 -> shared Linear(8,1), reshape [B,16,13]
 -> residual addition to the unchanged fusion output, learned gate initialized0.1
 -> original two-block TCN core -> dense8 -> categorical heads
```

Only left internal padding, no future/right padding or temporal subsampling.
No RevIN, input clipping, rolling scaling, BatchNorm, future covariates or
agent-state feedback. LayerNorm acts on hidden patch embeddings only; the raw
fusion output remains available through the residual path. Positional encoding
is mandatory and covers every token. Gate0 is an exact bypass for tests.
The adapter's random initialization is isolated so existing modules keep the
same seed-dependent initialization before learning.

CUDA integration caught an attention-bias stride alignment failure under
TorchRL twin-critic vectorization at length13. The adapter scopes PyTorch's
mathematical SDPA backend locally. The same CUDA update/checkpoint regression
must pass before launch; no padding to a different market window is allowed.

Same full TRAIN2020-2023 and VALIDATION2024 each epoch; TEST2025 closed.
Same fixed TRAIN scaler and original support; no new rows after feature pruning.
Batch128, replay65536, warmup4096, update every4,lr0.0003,wd0.01,gamma0.99,
tau0.005, entropy fraction0.6. Up to2000 full-data epochs; validation-return/week
monitor, min_delta1e-8, patience20, first monitor reference at epoch20.
No change to the existing warmup/checkpoint rule is hidden in this comparison.

## Test-led gates

1. Disabled adapter reproduces legacy initialization/state keys and outputs.
2. Shape/order: fusion -> adapter -> core; actor and critics include the adapter.
3. Suffix perturbation leaves earlier outputs unchanged, in train and eval modes.
4. Positional buffer nonzero/distinct; no channel mixing inside shared attention;
   patch construction observes only current/past positions and retains length13.
5. Finite SAC updates through TorchRL twin-Q vectorization, actual adapter weights
   change, checkpoint and cold-reload exact; no stochastic evaluation RNG changes.
6. Receipt/source/scaler/support/month/count authentication for both arms; paired
   config differs only by adapter architecture and run identity.
7. Full native business controls pass before launch. Persistent per-host workers,
   epoch recovery, bounded restart and automatic results collector.

No weight-space tuning is folded into either arm. The frozen algebraic edits
and their exact controls in197-200 remain a separate diagnostic line; arbitrary
weight perturbations based on repeatedly exposed validation are not introduced.

## Subsequent Bayesian-head experiment

Owner raises uncertainty heads during preparation. Keep them out of these two
arms to isolate the temporal adapter. After this comparison, use the selected
architecture for a paired deterministic-versus-Bayesian output-head contrast,
same support, seeds, optimizer budget, stopping and business controller.
For RL, categorical policy entropy is not itself calibrated epistemic
uncertainty. Specify posterior sampling, prior/KL regularization and stochastic
policy semantics before implementation. Measure performance, posterior dispersion
and uncertainty reliability without first changing sizing or trade thresholds.
Only a subsequent controller experiment may consume uncertainty for risk decisions.
If indicated, complete the2x2 attention/head factorial to measure interactions;
a single winner-plus-head comparison cannot establish the interaction effect.
This is planned research, not another silently launched training arm.

## Verified launch

Runtime `9fb1e8d4`. Both fits started2026-11-04 21:31:25UTC.

| Arm | GPU | Total trainable parameters | Identity |
|---|---|---:|---|
| Baseline | RTX4090 Laptop |96418|`8dbba165d61ffd873c857a38d6eb03852f9a03efadc917d3c38199feed544e44`|
| Causal patch attention | RTX5090 |98344|`36adc294a76e011bde8e99213abcd1ad2ea4c230d96037297c80faa734aaefa4`|

50 focused CPU tests passed;3 CUDA cases skipped in that CPU-only invocation.
All12 adapter tests, including CPU and CUDA cases, passed on the assigned GPU.
Authenticated paired files/scaler/column order/support and design are identical;
effective learning configs differ only in the adapter. Native controls pass for
both. Initial observed updates65 and129 respectively have finite losses and
gradients; these are launch checks, not trading performance results.

Both workers are persistent services with6GiB RAM limits, no swap, resource
admission, bounded failure restart and full epoch recovery. An independent
collector retrieves both arms automatically until their final results exist.
No intermediate approval is required. Final ETA depends on full epoch duration
and observed patience; replay-fill throughput is not used as a training estimate.

The preflight inherits the parent's historical `continuation_of_fresh_fit` flag;
this arm is NOT a weight continuation. Its governing `structural_round` records
fresh weights and replay, and neither preparation nor worker imports the compact
actor weights. That frozen reference remains unchanged.

One seed across different GPU models is an initial architecture contrast, not
proof that a performance difference exceeds training variability. Stronger
claims require matched repetitions. No Bayesian head or controller adjustment
is present in either run.

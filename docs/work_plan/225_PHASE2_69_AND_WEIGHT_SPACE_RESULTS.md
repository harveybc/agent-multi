# Phase2 Selected69 and Direct Module Weight Diagnostics

Hardware attribution correction2026-11-09: the RTX5090 name below was inferred
from a CUDA ordinal. That is not a verified hardware identity; see226 for the
ordinal mismatch found in the next run. Results and parity receipts remain
unchanged; future runs require the actual device name before executing.

Completed2026-11-08 local evening (receipts2026-11-09 UTC). This continues
phase2 independently of the forward-model-pool integration. All results below
are development results; no TEST access, new data, scaling, window or strategy
change. TRAIN2020-2023, VALIDATION2024;13 observations, stride2, fixed TRAIN
scaling, categorical discrete SAC, no agent-state inputs.

## 1. Two Ordered Frozen Selection Passes

Primary RTX5090, frozen selected70 epoch0. Delete improvements/ties, reinsert
strict improvements, coherent encoding blocks, full original row support,
activity strictly greater than two orders/week in both periods. The two passes
completed with67 fresh evaluations:70 ->69 ->69,28 groups throughout.
Removed RAW_CLOSE_LOG_RETURN; reinserting it lost the improvement. This is not
evidence that returns in general are useless; other return columns survive.

| Model | TRAIN return % | TRAIN Sharpe | VAL return % | VAL Sharpe | VAL trades | VAL win % | VAL DD % |
|---|---:|---:|---:|---:|---:|---:|---:|
| Preserved70 |11.514527|0.573391|13.808489|2.097057|206|57.766990|3.624931|
| Selected69 |10.792108|0.539644|14.591090|2.165596|206|58.252427|4.044667|

VALIDATION return improves0.782601 percentage points; TRAIN return falls0.722419
points and VALIDATION drawdown increases0.419736 points. This is a tradeoff,
not dominance on every metric. Selection score is the previously frozen
VALIDATION net return/week. Final activity: TRAIN750 trades/209 weeks,
VALIDATION206/54. Keep both reference and selected artifacts.

Selection identity:
`906c0dc6795b237722c12704b4fd243591a7e7b2d79244325d78dc223162a96e`.
Result SHA256:
`0125e124cd9913f76091414a04df5d88eb2da4a38861697c2b41257d88f80026`.
Unmasking after the full pass reproduced the original accounts exactly.

## 2. Physical Removal Without Losing Weights

`tools/export_selected_weight_bundle.py` projected the recipe/scaler and
removed the input channel from the actor, twin critics and target critics.
Three tensor entries containing first-convolution weights were sliced;
943 retained tensor entries copied exactly, with index buffers remapped.
No optimizer update, no reinitialization of surviving weights, no training.

On all30,234 valid windows: zero action disagreements, actor maximum absolute
logit difference1.90735e-6 (floating-point convolution shape effect), online
and target Q maximum difference0. Native TRAIN/VALIDATION accounts reproduce
the masked winner exactly; cold bundle reload and account replay pass.
Weights, full recipe and per-component exports are preserved independently.

New69 model SHA256:
`8ab11f607dc18c53ce45b34dc628e21272767702d60e0bfbe272620761de4e40`.
Bundle manifest SHA256:
`69f2c7e48a0028c9695a65dafaf3b9d5a1d7c4e33542340eb062fbea3bc5d424`.
Export result SHA256:
`a527619fa474bd1c038c6701b200325ed80952db3d588884baee03c5532c669a`.
This is an inference artifact, not a completed refinement or paper promotion.
The70-column artifact supplied to the forward-pool lane remains unchanged.

## 3. Actual Weight-Space Inspection

Tool:`tools/inspect_selected_weight_space.py`; CPU, about3 seconds. Compared
effective TorchRL weights at reference epoch0, degraded epoch40, and the
4096-update critic anchor. Each source authenticated against its receipt.
All extractor, fusion, attention, core and head weights have finite/norm/rank
summaries; fixed TRAIN probes record activations and individual final-head
unit contributions. Effective copied actor and critics reproduce native
outputs before inspection. No dataset fitting or optimizer updates occurred.

Measured findings:

- Actor and online-critic encoders did not move: all28 extractor groups,
  fusion, attention and both core blocks are bit-identical. Actor changes
  are exactly the two dense head layers (weights and biases). Target-critic
  encoders did move under their existing soft updates; they are not described
  as frozen or assumed erroneous solely because they moved.
- On the32-row fixed TRAIN probe, mean actor logits changed from
  [1.3205,-1.2228,-0.5009] to [11.9832,-12.7215,-9.8162]. NOP wins25/32
  before and31/32 afterwards. This probe is descriptive, not the trading test.
- Degraded twin critic action means are around-25 to-27, while their final
  biases are roughly-1.26 to+0.33. The large offset is therefore not merely a
  bad final bias. For critic0, hidden units0/2 have means28.85/32.28 and large
  negative output contributions. Critic1 has negative weights across every
  output row, applied to positive mean hidden activations. These arithmetic
  decompositions reproduce the output means; they do not establish the
  historical learning cause of those weights.
- The critic-anchor run leaves the degraded actor exactly unchanged. Better
  critic calibration alone did not restore its policy or create better trades.

An algebraic edit subtracting the common action component of the actor's final
weights/biases was tested on disposable copies. Softmax probability discrepancy
was at most1.19e-7 and there were zero action disagreements. This is a
function-preserving control, not a trading improvement. It must not be applied
as an alleged Q-value correction: critics have absolute value semantics.

Weight report SHA256:
`7eb88977a62264204ad4b9a583207fa0c61d085205abf305d65e28458ef03a0d`.

## 4. Direct Head Substitutions on Full Accounts

`tools/evaluate_weight_head_transplants.py` executes a fixed2x2 factorial,
on the original70 features, copying only named actor tensors. Neither features
nor strategy change. All actor tensors outside the head must match exactly.
Reference replay must equal archived accounts before any interpretation.

| Hidden head layer | Final output layer | TRAIN return % | VAL return % | VAL trades | Activity admissible |
|---|---|---:|---:|---:|---|
| Reference | Reference |11.514527|13.808489|206|yes|
| Degraded | Reference |-16.734387|4.754616|75|no|
| Reference | Degraded |9.507886|4.030860|265|yes|
| Degraded | Degraded |-10.816846|4.788059|34|no|

The full degraded actor outcome is reproduced while the encoder stays fixed.
Neither individual head swap improves the baseline. The first head layer
drives a strong loss of activity; the last layer also worsens performance but
does not by itself suppress activity. This identifies harmful parameter
changes and interaction, not a universal claim about dense layers or a proof
that financial inputs cannot generalize. The underlying learning dynamics
remain the separate critic/policy calibration question documented214-220.

Result SHA256:
`df84883dd7b9b53fa5f8a42d6e86fc6d595390a0b0623994aacca4e53ba487bb`.
No intervention is promoted. The selected69 improvement came from phase2
selection, not from falsely attributing a gain to a head transplant.

## 5. Verification, Artifacts and Next Step

Focused suite:17 passed (weight diagnostics, selected transfer, refit masks),
one existing PyTorch batching-performance warning. Real GPU runs additionally
checked full native accounts and retained actor/critic/target parity, followed
by exact cold reload of the selected69 bundle. CPU diagnostics and GPU
experiments are separate reproducible commands; no background observer is
needed to finish these completed jobs.

Artifact roots on the research workers and coordinator: sac70-next-mask,
sac70-module-weight-audit-v2, sac70-weight-heads-v2, sac69-weight-bundle,
all with date suffix20261108. Full tensors/logs remain outside the public
repository. Receipts above identify them without publishing private paths.

Keep selected69 as the new frozen research candidate and70 as its control.
Before another long refinement, use bounded actor-update tests with explicit
drift/activity gates and separately qualify the critic against realized-return
and constant controls. Do not repeat the known degrading training unchanged.
Forward acquisition/paper integration proceeds independently and must select
an explicit recipe/checkpoint identity;69 is not silently substituted for70.

The [weight-space programming article](https://rikiyat.github.io/blog/weight-space-programming/)
motivates direct inspection and measurable edits. Its successes do not imply
every learned module can be understood or improved from weights alone. Here,
the controlled interventions found harmful changes, while the structural edit
preserved the measurable phase2 gain without retraining.

# Musashi to General Satoshi: Data-First SOTA Multibranch Order

Date: 2026-08-26
Priority: P0 after the currently assigned Screen-B corrections
Authority: owner correction; execute without a new approval phrase

## 1. Correction of direction

Do not treat the flat MLP as the primary candidate. It remains a legitimate
architectural baseline: reuse existing valid evidence first, and run it anew
only when a bounded matched comparison is required to quantify the value of
temporal structure. Do not substitute a deliberately weak GRU merely to create
a cheap control. The next prospective learned candidate must preserve temporal
order, semantic feature families and the exact historical/live data contract.

Add the owner's 10% nominal annual COP CDT alternative as an economic reporting
hurdle, separate from every model/strategy baseline. Produce venue-currency and
COP-converted returns with timestamped FX provenance; do not inject this hurdle
into training fitness or compare it directly with USD returns.

## 2. WP-DATA: decide the inputs before the network

Create a machine-readable inventory for every proposed field with:

- semantic family, units, sampling timestamp and publication delay;
- historical source, first/last timestamp, missingness and immutable digest;
- live source for Alpaca and/or MT5, measured availability and freshness;
- causal transformation and normalization fitted on training data only;
- which venue/asset can consume it live; unavailable inputs need a typed mask;
- license/redistribution status and estimated storage/collection cost.

Start from the current 83 ETH H4 variables, but explicitly inventory the absent
families: bid/ask spread and quote depth, realized slippage, finer-grained
OHLCV/range, funding, open interest, basis, liquidations, cross-asset context,
calendar/session and event variables. Do not invent values or admit a feature
that exists only historically. Report which sources we already collect and
which require a new collector before proposing acquisition.

Materialize multiresolution causal windows rather than treating 32 H4 bars as
the final design: short intraday, medium H4 (including 30/90-day horizons) and
long daily context. Exact windows become bounded genes after coverage and live
latency are measured, not arbitrary constants.

Acceptance: one inventory artifact, one synchronized-schema proposal, missing
data policy, leakage tests, historical/live parity tests and storage estimate.

## 3. WP-ARCH: make the strong route executable

Build on the existing grouped extractor and plugin system; do not duplicate it.

- continuous return/range branches: PatchTST and causal-TCN implementations;
- heterogeneous trend/known-covariate branch: TFT-style variable selection and
  temporal attention;
- volatility/distribution branch: TimesNet-style and causal-TCN alternatives;
- volume/flow and derivatives branches: causal TCN/GRU alternatives;
- account/position/execution scalars: gated MLP only here;
- fusion: typed gated fusion plus cross-family attention;
- configurable actor and twin-critic heads over the fused latent;
- structured action contract preserving target exposure and explicit
  close/hold authority inside native SL/TP protection.

Every component resolves through a plugin entry point and nested effective
config with strict unknown-key refusal. Generate segmented architecture diagrams
and parameter tables from the effective model, and link them from the README.

Acceptance: real `GymFxEnv` construction, causal-mask adversaries, all branches
receive nonzero gradients, tiny-fixture overfit, save/load bit-level output
parity, CPU and CUDA smoke, parameter/FLOP/latency report.

## 4. WP-PRETRAIN: finish what currently exists only as scaffolding

The Huber and direction objective plugins are not enough and are not presently
wired to a complete branch-pretraining runner. Implement the executing runner,
artifact contract and resume semantics. Add configurable objectives for masked
patch reconstruction, hierarchical contrastive representation, multi-horizon
quantile returns, volatility and barrier-hit probabilities. Compare:

1. strong grouped model from random initialization;
2. independently pretrained branches;
3. shared multiscale temporal pretraining.

Bind feature order, windows, data roles and hashes, topology, objective weights,
seed and code identity into every artifact. Sealed 2025 remains absent.

## 5. WP-SCREEN and DOIN domain

Amend Screen C per doc 40. C0 is mechanics only. The prospective arms are
strong architectures; no flat MLP receives new GPU. Search branch families
first, then fusion, then pretraining, then bounded DOIN topology/hyperparameters.
Genes include branch type, depth, width, patch/kernel/dilation, attention heads,
dropout, normalization, fusion, core/head topology, pretraining objectives and
loss weights. Use hierarchical stages to prevent a combinatorial explosion.

The MLP may appear as a bounded matched baseline but cannot win by default when
strong models fail. Easy-to-normal returns only after a strong architecture is mechanically valid
and easy dynamics demonstrably differ from normal. Preserve weights exactly at
handoff; replay carry remains its own treatment.

## 6. Immediate return packet

Before any long GPU dispatch, return:

1. the data inventory and exact proposed v2/v3 input tensors;
2. gap analysis of existing code versus this order;
3. implementation commits and focused/full test results;
4. effective configs and generated architecture diagrams;
5. C0 smoke evidence and estimated cost of the strong comparison;
6. explicit unknowns and any literature/data collision that changes the design.

Continue useful CPU implementation while current assigned experiments run. Do
not leave GPUs idle once an independently verified strong GPU job is ready, but
do not fill them with flat-MLP work merely to show utilization.

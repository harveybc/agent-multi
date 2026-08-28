# Musashi to General Satoshi: feature-extractor information audit and execution order

Date: 2026-08-28  
Authority: General Musashi  
Priority: immediate, after preserving the failed paired-SAC attempts  
Scope: `agent-multi` and the executing `gym-fx` contract only

## 1. Objective

Determine whether the grouped multibranch extractor preserves causal, trading-relevant temporal information and whether its topology and pretraining are justified by evidence. Mechanical execution, finite outputs, gradients and exact weight loading are necessary but are not evidence of representation quality.

The current extractor is an experimental candidate. It must not be described as a proven SOTA trading extractor. The branch implementations must be named `PatchTST-style`, `TFT-style` and `TimesNet-style` unless exact parity with the reference architecture is demonstrated.

## 2. Immediate runtime priority

Before restarting the paired SAC campaign, reproduce and correct the terminal trade reconciliation defect observed on both gamma slots:

- preserve both failed attempts and their reports unchanged;
- reproduce the cases where `closed_trades_cumulative` exceeds `trades_total`;
- identify whether the difference comes from terminal liquidation, rejected/superseded orders, episode reset, trace timing or summary construction;
- define one authoritative closed-trade event stream and derive both values from it;
- add zero, one, multiple, last-bar liquidation, simultaneous-close and interrupted-episode regressions;
- reject disagreement before accepting an epoch;
- perform one bounded CPU end-to-end run and one bounded CUDA run only after the CPU result agrees exactly;
- do not resume the eight-cell economic campaign from partial attempts. A corrected campaign uses fresh sibling attempts.

Return the root cause, PRE/POST reproducer and exact derived counts before requesting redispatch.

## 3. Establish the current scientific baseline

Produce a machine-readable architecture and training manifest derived from executing code, not copied from config prose. It must include, per branch:

- semantic family and ordered feature identities;
- input window and effective temporal receptive field;
- exact implementation class and differences from the reference paper;
- layer-by-layer tensor shapes;
- trainable parameter count;
- latent dimension and compression ratio;
- causal mask/padding behavior;
- reduction from sequence to final representation;
- initialization, dropout, normalization and optimizer settings;
- pretraining objectives and transferred/excluded keys;
- actor, critic and target-critic transfer behavior.

Record explicitly that the current candidate uses a 32-bar H4 window, fixed hand-selected widths, mostly zero dropout and approximately 115.6k extractor parameters. Verify every value from runtime.

## 4. Temporal-information acceptance suite

Implement a reusable suite that distinguishes absence of lookahead from preservation of useful temporal information. It must operate per family and on the fused representation.

Required controls:

1. Future mutation must not change an earlier representation.
2. Mutating the newest available bar must change the representation unless the input is provably identical after preprocessing.
3. Reversing time must produce a materially different representation.
4. Permuting bars must degrade predictive probes.
5. Phase-randomized surrogates preserving the marginal spectrum must lose phase-dependent predictive performance.
6. Constant, duplicated and noise-only channels must not produce false predictive success.
7. Save/load and resume remain bit-exact under the same identity.

Required measurements:

- normalized masked-reconstruction error by feature and family;
- spectral coherence or normalized spectral error by frequency band;
- lagged cross-correlation preservation;
- linear and small nonlinear frozen-encoder probes for quantiles, realized volatility and barrier hit;
- effective-rank and representation-collapse diagnostics;
- sensitivity to each temporal region of the input window;
- results on real data and controlled synthetic signals with known periodicity, phase and regime changes.

No single reconstruction metric may decide acceptance. A branch passes only if it beats shuffled-time and random-encoder controls on future-facing probes without leakage.

## 5. Window and bottleneck screen

Materialize a bounded causal screen over windows `{32, 64, 128, 256}`. Permit a different winning window per family. Do not claim that a 252-bar indicator lets a 32-bar encoder observe the original 252-bar trajectory.

For each viable window, screen latent dimensions `{16, 32, 64, 96, 128}` subject to topology constraints and matched training budget. Report the Pareto frontier over:

- future-probe performance;
- representation dimension;
- parameter count;
- wall time and peak memory;
- compressed checkpoint size as a descriptive MDL proxy only.

Do not estimate Kolmogorov complexity as a model-sizing target. Compressor length is an algorithm-dependent upper bound, not the amount of predictive information and not a requirement that model capacity equal or double dataset size.

## 6. Per-family architecture screen

Compare plausible encoders under matched parameter count (target tolerance +/-5%), optimizer updates, data, windows and seeds:

- causal linear/MLP control;
- causal TCN;
- unidirectional GRU;
- PatchTST-style;
- TFT-style;
- TimesNet-style.

Use at least four seeds for any selection claim. A complex branch wins only when its paired future-facing probe improvement is stable and not explained by additional capacity. The simple controls remain in every report; they are scientific baselines, not candidate champions.

The first screen is CPU/bounded GPU mechanics and low-cost ranking. Do not launch the full Cartesian product. Use successive halving only after every candidate has completed the same minimum budget.

## 7. Topology genes and ranges

Replace undocumented fixed choices with a typed, conditional topology domain. Proposed initial ranges, subject to feasibility validation:

- common latent width: `{16, 32, 64, 96, 128}`;
- layers/blocks: `{1, 2, 3}`;
- attention heads: divisors of latent width in `{1, 2, 4, 8}`;
- dropout: `{0.0, 0.05, 0.10, 0.20}`;
- PatchTST patch length: `{4, 8, 16, 32}` bounded by window;
- PatchTST stride: `{patch/4, patch/2, patch}` after integral validation;
- TimesNet top-k periods: `{1, 2, 3, 5}` bounded by resolvable frequencies;
- TCN blocks: `{1, 2, 3, 4}`, odd kernels `{3, 5, 7}`, dilation bases `{2, 3}`;
- GRU layers: `{1, 2, 3}`, hidden width from common domain;
- fusion width: `{32, 64, 96, 128}`, heads constrained by divisibility;
- weight decay: `{0, 1e-6, 1e-5, 1e-4}`.

These are screening domains, not assertions of optimum. Derive narrower DOIN domains from the bounded screen. Invalid or parameter-budget-violating cells must be absent from the materialized domain rather than launched and rejected later.

## 8. Pretraining correction and ablation

The existing five-objective generation is diagnostic because prior screens found material objective conflicts. Run a predeclared ablation using the same causal partitions and probes:

- random initialization;
- masked reconstruction only;
- predictive objectives only: quantiles + volatility + barrier;
- predictive objectives + contrastive;
- all five objectives with PCGrad;
- the best per-family objective route, if the common probe supports one.

Use early stopping on a fit-tail monitor, a separately configured ReduceLROnPlateau and a generous maximum epoch budget. Record best epoch, terminal epoch, LR changes and patience state. Ten epochs may be used for mechanics but not for a representation-quality conclusion.

The monitor cannot calibrate objective weights and then judge the same choice. Calibration, monitor and downstream probe partitions remain distinct and purged by the maximum target horizon.

## 9. Transfer into SAC

For the winning representation candidate, compare prospectively:

- random extractor trained end-to-end;
- pretrained extractor frozen;
- pretrained extractor fully fine-tuned;
- gradual unfreezing with a lower encoder LR than new fusion/SAC layers.

Keep actor and critic transfer facts separate. Verify whether independently fine-tuned actor and critic copies outperform sharing. The randomly initialized state branch and cross-family fusion must be identified as such; test whether pretraining the fusion provides incremental value.

The primary conclusion is paired economic performance under the same execution envelope. Probe metrics are selection diagnostics, not substitutes for trading results.

## 10. Information-theoretic analysis

Add an analysis module, explicitly diagnostic and non-authoritative, that reports:

- empirical entropy rate estimates by feature family after preprocessing;
- predictive information proxies between the past representation and future targets;
- representation effective rank and compressibility;
- rate-distortion curves from the bottleneck sweep;
- information retained about nuisance controls versus future targets.

Do not use parameter bits, checkpoint bytes or the statement "two bits per connection" as a fitness function. No universal conversion from connections to useful information exists. Use information-theoretic quantities to compare representations under controlled interventions, not to decree model size.

## 11. Dispatch order

Execute in this order:

1. Repair trade reconciliation and return the bounded runtime evidence.
2. Deliver the runtime-derived architecture manifest.
3. Implement and run the temporal-information suite on the current candidate and controls.
4. Run the bounded window/bottleneck and per-family architecture screen.
5. Return the proposed narrowed topology domains for audit.
6. Run the pretraining ablation after acceptance of the screen design.
7. Run the transfer-policy comparison.
8. Only then dispatch DOIN topology/hyperparameter optimization and the long economic comparison.

CPU implementation, tests and manifest generation proceed immediately. GPU work may use otherwise idle devices for bounded, predeclared screens after the trade-reconciliation repair, but must not compete with a valid active campaign. Do not restart the invalid partial campaign merely to keep devices busy.

## 12. Required return package

Return:

- PRE/POST evidence for the trade-count defect;
- architecture manifest and layer diagrams;
- exact datasets, rows, dates, purges and hashes used by every screen;
- all candidate and rejected topology cells;
- per-seed probe and integrity results without composed headline scores;
- raw economic metrics where applicable: return, trades, exposure, drawdown, Sharpe, turnover and costs;
- compute and memory accounting;
- explicit limitations and the next single recommended experiment;
- a clear separation among `MECHANICS_ONLY`, `REPRESENTATION_DIAGNOSTIC`, `ECONOMIC_SCREEN` and `PROMOTION_ELIGIBLE` evidence.

No result may be called SOTA merely because its plugin name derives from a SOTA paper. The claim must be earned against matched baselines on our data and trading objective.

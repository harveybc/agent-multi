# Selected72 Interior Refinement

2026-11-12. Owner authorizes the next bounded refinement after297 diagnosis.
Source:72-input exported actor295 and296 fresh calibrated critics, seeds19/43.
The297 diagnostic must authenticate to the same actor, support and calibration.

| Requirement | Gate |
|---|---|
| No inherited69-input guard | Recompute strict native-correct FIT actions and margins from current72 actor |
| Corrective targets | Recompute unique native-optimal FIT errors, weighted by native advantage over source action; no ZigZag or extra observations |
| No validation optimization | FIT2020-2022 only,2023 MONITOR diagnostics;2024 account evaluation only |
| Fixed scope | Eight attempted head updates per seed,batch128; representation,critics,alpha and buffers frozen |
| Preserve prior correct decisions | Same1e-7 source-margin tolerance and identical protected greedy actions |
| Interior feasible proposals | Source-derived dtype rounding cushion; absolute linear residual >=-1e-10; nonlinear checks authoritative |
| Trust budget | Step-margin RMS<=0.002,meanKL<=1e-4,maxKL<=1e-3,relative head drift<=0.01;24 backtracks; nontrivial movement>=1e-6 |
| Reject safely | Restore parameters and optimizer moments when candidate fails; no fallback to unconstrained update |
| Reproducibility | Sealed source,immutable runtime,cold checkpoints at1/8,exact full native account parity |
| Attribution | Both seeds and both periods, chronological account deltas; finite automated worker -> report -> hash-verified collection |

This is the same tested interior correction, with fresh sets/schedule from the
new calibrated source. Historical failed guard-only/targeted69 arms are not
rerun as ceremonial prerequisites, nor used to supply current targets. The
new schema explicitly separates this transfer from historical294 provenance.

The actor objective is corrective cross-entropy on native FIT returns,
weighted by advantage; this is not a fresh unconstrained SAC-loss update.
Critic values are frozen and retained for diagnostics, not used as correction
targets. Preserve frozen source TRAIN15.612726%,VAL18.369946%,746/207 orders.
Do not promote any candidate automatically or conflate local objective decrease
with higher trading performance. COT/live/STL qualification remains separate.

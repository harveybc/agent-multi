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

## Completed Results

Runtime e6e34352. Both seeds accepted8/8 full projected steps with no relaxed
constraint.25 current FIT-correct decisions were protected and64 unique-optimum
FIT errors supplied correction targets. The old69-input sets were not reused.
Worker25.01s, report2.37s; finite worker/report/collector all completed.28 tests
and scoped Ruff pass. Independently authenticated164 worker/source/export files
plus15 report files; four checkpoints pass frozen-state,optimizer-count,margin
and target/probability reconstruction. All four account attributions reconcile
with zero residual. Original actor and critic source artifacts remain intact.

| Endpoint | TRAIN return % | VAL return % | TRAIN Sharpe | VAL Sharpe | TRAIN DD % | VAL DD % | TRAIN / VAL trades |
|---|---:|---:|---:|---:|---:|---:|---:|
| Frozen72 |15.612726|18.369946|0.757587|2.692275|9.461544|3.257290|746 /207|
| Interior19,step8 |15.818189|18.369946|0.765395|2.692275|9.461350|3.257290|747 /207|
| Interior43,step8 |15.818189|18.369946|0.765395|2.692275|9.461350|3.257290|747 /207|

Each endpoint gains20.546375USD in the TRAIN account, from one effective
NOP-to-SELL entry change and its chronological balance-dependent continuation.
Two TRAIN signals changed; only one changed the account. One validation signal
changed, but no validation order or account changed. Equivalent endpoint account
results across schedules are not independent market replications.

At endpoint8, TRAIN has227 BUY (152 wins),520 SELL (312 wins),62.1151% wins.
Validation remains22 BUY (14 wins),185 SELL (110 wins),59.9034% wins.
Early-close outcomes are unchanged: TRAIN59 (29 helpful,30 harmful),VAL6
(4 helpful,2 harmful). This does not repair the closing policy or demonstrate
new out-of-sample profit. Minimum protected margin gains remain positive,
1.0878e-6/1.1139e-6 at the endpoints.

FIT desired-action probability improves on57/64 and56/64 error anchors;
none crosses its greedy boundary. The additional TRAIN order comes from the
full-period evaluation, not a corrected sampled error anchor. MONITOR desired
probability improves21/23 in both schedules, again without an argmax correction.
Do not confuse these local improvements with correction of every decision.

## Disposition

Preserve frozen72 and both endpoint checkpoints. Next qualify/export both
fixed-step8 candidates with cold/native parity; primary seed19 by the existing
fixed convention, not an after-the-fact validation tie-break. Then run fresh-
context deletion plus cumulative readmission of all48 phase2 absences on the
qualified actor. This is a new actor context, not a repeat on identical weights.
Do not reset the actor, run a long unconstrained refit or alter live services.
Live eligibility still depends on resolving the COT/source identity separately.

Run identity:`01b7b09848026446685ad813d6f2947b11a387b55775f1406a6235f8c2608da3`.
Private roots:`sac72-interior-fit-20261112-r1` and
`sac72-interior-report-20261112-r1`. Checkpoints are preserved at
`interior_error_project_guard_{19,43}/ACTOR_0008.pt` under the run root.

Evidence:[summary](generalization/selected72_interior_20261112/SUMMARY.json),
[independent audit](generalization/selected72_interior_20261112/AUDIT.json),
[steps](generalization/selected72_interior_20261112/STEPS.json),
[return plot](generalization/selected72_interior_20261112/RETURNS.png).

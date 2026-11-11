# Selected Actor Export And Bounded Refinement

2026-11-11. Owner authorizes export then refinement of289's changed69-input
composition,28 groups. Preserve the selected inference bundle and previous
SAC checkpoints. No source overwrite, data substitution or live change.

## Frozen Design And Gates

| Requirement | Test / measured gate |
|---|---|
| Physical selected inputs | Slice expanded learned channels into an ordinary69-input ThreeLogits architecture; preserve group/fusion order and head |
| No lost constant response | Reject dropping nonzero whole-group responses; reject unsupported centered donor groups |
| Preserve actual policy | Full30234-window parity, zero changed decisions, exact actions/trades/equity/close traces against289 |
| Reloadable SAC | Standard bundle export, cold state/logits/accounts exact; no optimizer training at export |
| Changed-input critic | Initialize critic encoders from exported actor; fresh twin heads, no relabeling old critic channels |
| Same-policy calibration | Recompute151 inherited timestamps' native continuations under new actor; FIT2020-2022 only, MONITOR2023 diagnostic |
| Bounded refinement | Seeds19/43;4096 critic-head calibration updates, then8 actor-head AdamW updates,batch128;284 trust limits; fixed endpoint8 |
| Frozen control | Exported actor at0; frozen representation,critic and alpha during actor updates; cold checkpoints at1/8 |
| Full assessment | TRAIN2020-2023,VALIDATION2024; activity >2/week; all endpoint account effects attributed; no TEST |

Calibration uses realized greedy reward returns, not ZigZag labels or a future
observation. The151 anchors preserve prior support, but every continuation is
rebuilt for the new policy. Encoders stay frozen; both critic heads are fit on
all three action returns. Targets synchronize after calibration. Actor and
temperature remain exact through calibration. Native actor entropy remains.

Keep step margin RMS0.002, mean cumulative KL0.0001,max row KL0.001 and relative
head drift0.01 with24 backtracks. No automatic long training if eight updates
fail. Record training loss and critic error separately from account gains.
Select neither intermediate checkpoints nor budgets by their validation score.

The exported critic is explicitly uncalibrated; calibration is mandatory before
actor updates. Prior critic encoders are not silently mapped onto swapped
columns. Representation initialization is recorded as a change, not claimed
equivalent to the old critic. The actor alone must reproduce289 before launch.

Expected frozen account:TRAIN13.913055%,VAL16.336139%,739/207 orders.
Execution: qualification on CPU, then sequential primary-GPU calibration/actor
arms, independent attribution and finite authenticated collection.31-test prior
qualification remains historical; the new scoped tests and full-data gates
govern this export.

The first CPU export stopped before learning: GPU-saved neutral buffers in
three inactive centered donor branches leave CPU residuals3.5763e-7,2.9802e-8,
2.9802e-8. Record these explicitly; only CenteredDonor roundoff up to1e-5 is
eligible for removal. Ordinary nonzero original-group responses still refuse.
Full logit tolerance and exact decisions/accounts remain mandatory. A regression
rejects a material1e-3 centered response. The first run is retained as stopped;
the corrected export uses a fresh r2 identity, never overwriting its evidence.

The next preflight caught an input-loader issue: projecting through the older
parent's already-reduced recipe removes the readmitted RAW_BAR_LOG_RETURN.
Export, refinement and reporting now project directly from authenticated full
support to the selected recipe; the default old loader behavior is unchanged.
A regression rejects traversal through the old subset. Use fresh r3 outputs;
neither failed preflight produced a trained actor or replaces the saved selection.

The r3 provenance check refused changing the historical loader pinned by289.
Keep that loader byte-identical and place selected-recipe loading in a separate
module; r4 is the export runtime. No historical receipt is relaxed or rewritten.

## Export Verified

41 focused tests and scoped Ruff pass. Runtime01774983 is immutable.
CPU export qualified in22.53seconds:69 physical input channels,28 groups,
313 actor tensors copied exactly after channel/group index mapping.
Compared with the packed selection, max logit difference2.86102294921875e-6
across30234 valid windows, zero action disagreements. Full actions, trade,
equity and close-counterfactual traces match289 exactly; cold reloaded state,
logits and accounts match the new export exactly. No actor learning occurred.

Export identity:`3778d1d1773ab737387c5e9bc979e58434c8ccd7fbe71b8047611677b9d1b32a`.
Model SHA256:`b4b432d7509e38f5c3236fd6be7d8a2d16d5280448bb65dcab97f7e9e0d12531`.
Full standalone bundle:`sac69-selected-export-20261111-r4/bundle/`.
The primary GPU passed readiness and replayed its reference accounts before
rebuilding all151 continuation anchors. Refinement, independent attribution and
finite authenticated collection were chained and completed successfully.

## Completed Results

Both seeds completed4096 critic-head calibration updates and8 accepted bounded
actor-head updates. This is a short controlled refinement, not a full retraining.
Worker runtime220.17seconds; independent account report7.97seconds. Worker and
collector exited0. The independent integrity audit authenticated163 files,
verified FIT-only calibration, actor invariance during calibration, and frozen
encoders/critics/alpha during actor updates. Four full-account attributions
reconcile with exactly zero residual. Seeds vary critic initialization and update
schedule, not the dataset. No TEST read or live promotion.

| Model | TRAIN return % | VAL return % | TRAIN Sharpe | VAL Sharpe | TRAIN DD % | VAL DD % | TRAIN / VAL orders |
|---|---:|---:|---:|---:|---:|---:|---:|
| Frozen export |13.913055|16.336139|0.685588|2.400526|9.548436|3.257322|739 /207|
| Refined19 |14.726061|15.725127|0.723184|2.325083|9.093860|3.257323|738 /206|
| Refined43 |14.095633|15.262069|0.698293|2.265622|9.038206|3.644422|735 /205|

Returns are net account return relative to initial balance, not Sharpe. Full
native costs, side counts, wins and exit diagnostics are in the evidence below.
Validation wins:58.9372%,58.7379%,58.5366%; orders/week:3.8333,3.8148,3.7963.
Frozen validation has23 BUY (14 wins),184 SELL (108 wins); refined19 has23 BUY
(14 wins),183 SELL (107 wins); refined43 has23 BUY (14 wins),182 SELL (106 wins).
All three have6 early closes,4 helpful and2 harmful versus passive protection.
This does not demonstrate an improved closing policy.

## Critic And Decision Diagnosis

Values below use the minimum of twin Q estimates, before/after calibration.

| Seed | FIT RMSE | Monitor RMSE | FIT pairwise accuracy | Monitor pairwise accuracy |
|---|---|---|---|---|
|19|0.751724 ->0.343750|0.616507 ->0.729269|0.5390 ->0.7390|0.5900 ->0.6400|
|43|0.824161 ->0.515893|0.647547 ->0.671427|0.5153 ->0.6712|0.4000 ->0.4900|

Calibration improves FIT accuracy but worsens monitor value RMSE; ranking and
value error must not be conflated. Monitor is2023, separate from2024 validation.
Improved average ranking is not proof that executable opening boundaries improve.

Seed19 gains81.300540USD TRAIN but loses61.101210USD validation by skipping one
profitable SELL. Seed43 gains18.257790USD TRAIN but loses107.407005USD validation
by skipping that same SELL and another profitable SELL. These are chronological
full-account continuation effects, including later balance-dependent sizing,
not isolated trade-P&L sums. All effective changes are listed in SUMMARY.json.

## Disposition And Next Work

Keep the frozen selected export as the research baseline; neither refinement
replaces it. All endpoint weights and calibrated critic snapshots are preserved.
The frozen bundle's critic remains explicitly uncalibrated: do not resume SAC
updates from that critic without calibration. Export parity is an actor guarantee.

Before another refinement or floating pass, diagnose calibrated Q rankings and
shared-head gradient transfer at executable FIT opening boundaries under the
new69-input composition. Do not optimize against the missed validation trades.
No new long run is launched as part of this completed task.

Evidence: [export](generalization/selected_export_refine_20261111/EXPORT.json),
[refinement](generalization/selected_export_refine_20261111/REFINEMENT.json),
[account attribution](generalization/selected_export_refine_20261111/SUMMARY.json),
[integrity](generalization/selected_export_refine_20261111/INTEGRITY.json),
[return plot](generalization/selected_export_refine_20261111/RETURNS.png).

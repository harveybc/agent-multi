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

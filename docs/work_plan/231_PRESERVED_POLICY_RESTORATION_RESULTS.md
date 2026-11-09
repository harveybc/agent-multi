# Preserved Policy Restoration Results

Design:[230](230_CRITIC_REFINEMENT_POLICY_AND_PROBE.md).
Completed2026-11-09 at11:14:04 UTC (06:14:04 America/Bogota),563.25seconds,
exit0. Runtime4b72e549,RTX5090. Both arms completed automatically. Coordinator
independently verified94 terminal-listed files. No source checkpoint changed,
no model promoted. Reference here is preserved70,not the selected69 champion.

## Calibration and Bootstrap

Same preserved epoch0 actor and its own24025 TRAIN transitions;4096 critic-head
updates,batch128,identical replay and empty optimizer state. Actor/encoders/alpha
stay fixed during calibration. The source-policy accounts reproduce exactly.

| Arm | Initial RMSE | RMSE after4096 | Zero-Q reference | RMSE after separate256 bootstrap updates |
|---|---:|---:|---:|---:|
| Retained heads |1.292724|0.617907|0.657385|0.628605|
| Fresh heads |0.802216|0.617996|0.657385|0.619441|

The large advantage of head reset observed on the degraded epoch40 critic does
NOT repeat as a final calibration advantage on the preserved epoch0 critic.
Both arms converge to essentially the same overall error here. Native bootstrap
slightly increases error,less for fresh heads in this bounded single-seed probe;
it does not reproduce the historical collapse over256 updates. These return
labels differ from229,so0.618 must not be compared directly against its0.290.

| Signal | TRAIN rows | Retained final RMSE | Fresh final RMSE | Zero-Q RMSE |
|---|---:|---:|---:|---:|
|NOP|17203|0.626225|0.617115|0.655383|
|BUY|2584|0.589642|0.608091|0.647729|
|SELL|4238|0.600510|0.627488|0.671199|

## Bounded Actor Updates

Actor heads then receive1/8/32 updates with calibrated critics fixed. This branch
reloads the calibrated checkpoint BEFORE the separate bootstrap test,so actor
results are not attributed to bootstrap-modified critics. All checkpoints retain
strictly more than2orders/week in both periods. Native full accounts follow.

| Arm | Actor updates | TRAIN return % | VAL return % | VAL Sharpe | VAL trades | VAL win % | VAL DD % |
|---|---:|---:|---:|---:|---:|---:|---:|
| Preserved reference |0|11.51453|13.80849|2.09706|206|57.76699|3.62493|
| Retained |1|11.51453|13.80849|2.09706|206|57.76699|3.62493|
| Retained |8|11.51453|12.67332|1.89739|208|57.21154|3.64822|
| Retained |32|11.21926|12.67332|1.89739|208|57.21154|3.64822|
| Fresh |1|11.94572|13.80849|2.09706|206|57.76699|3.62493|
| Fresh |8|11.51186|13.23948|1.97910|207|57.48792|3.62492|
| Fresh |32|12.24694|12.71370|1.90953|208|57.21154|3.62494|

Fresh/1 improves TRAIN0.43119percentage points with exactly unchanged validation
actions/accounts. It is not a validation improvement or a demonstrated robust
successor. Fresh/32 improves TRAIN but loses1.09479validation percentage points.
The original69 and70 models stay preserved; these are diagnostic candidates.

## Action Sensitivity

Native ACTIONS.json files were matched by timestamp over24025 TRAIN decisions
and6207 VALIDATION decisions. Against reference:

| Arm/updates | TRAIN signal changes | VAL signal changes | VAL command changes |
|---|---:|---:|---:|
| Retained/1 |1|0|0|
| Retained/8 |3|3|3|
| Retained/32 |24|8|5|
| Fresh/1 |2|0|0|
| Fresh/8 |10|7|3|
| Fresh/32 |39|16|14|

Small logit movements can flip categorical decisions and change subsequent order
history. This is measured sensitivity,not proof of an incorrect decoder or of
absent market signal. Next localize which changed decisions and critic action
gaps account for losses,using native same-state counterfactuals before changing
observation inputs or launching another long joint refinement. Cumulative phase2
readmission remains designed but unexecuted; this probe does not change its scope.

Queue identity:
`7bb2154b3c0e4b76c69666edca8ac56296d3cd8ad66d39379a5541a94109fd1f`.
COMPARISON SHA256:
`0772506fe37f2fa29a9dc31eb053f1733cb77f0369a20ea3324d06eab273a9ec`.

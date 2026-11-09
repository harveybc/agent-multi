# Actor Preservation Results

## Completion

Contrast238 completed2026-11-09T18:34:19.569445Z (13:34 Bogota) in234.44s.
All six arms and automatic collection completed.449 receipt files authenticated.
Identity:`aad33a5915bc28351c2501b6f867d2cac776bc006fb2eca7948304391c671107`.
Runtime418b5fb6;37 focused tests and real RTX5090 gradient canary passed.
Private artifacts:`/var/tmp/sac70-actor-preservation-20261109`.
TRADING.csv contains all84 arm/point/period rows, including reused reference
rows at update0. There are76 separately evaluated full accounts: two periods
per reference/seed plus six nonzero points per arm/seed. All meet activity.
No checkpoint was promoted. Both source69 and source70 hashes remain unchanged.

## Fixed256-Update Endpoint

Unchanged parent:TRAIN11.51453%,VALIDATION13.80849%. Values below are not
selected peaks. Native1/8/32 full accounts reproduced the preceding experiment
exactly in both seeds. Every arm preserved encoders, online critics and alpha.

| Seed | Arm | TRAIN return % | VAL return % | VAL Sharpe | VAL DD % | VAL orders | VAL win % | VAL early closes |
|---|---|---:|---:|---:|---:|---:|---:|---:|
|19|Native|12.77009|12.99522|1.97982|3.78561|198|57.57576|5|
|19|Learning rate/3|12.30802|14.62457|2.17477|3.62494|206|58.25243|6|
|19|Parent KL|12.15072|14.05738|2.08947|3.62493|207|57.97101|6|
|43|Native|10.38354|9.80770|1.51906|3.68927|213|55.86854|13|
|43|Learning rate/3|10.77681|11.92208|1.78255|3.71268|209|56.93780|9|
|43|Parent KL|12.27509|12.83945|1.90388|4.04463|205|57.56098|7|

All six trade in both directions. Validation BUY/SELL counts respectively:
native19:22/176; lower-rate19:23/183; KL19:23/184;
native43:35/178; lower-rate43:27/182; KL43:24/181.

## Mechanism Evidence

| Seed | Arm | Mean validation KL(parent || current) | Changed greedy actions |
|---|---|---:|---:|
|19|Native|0.00205913|65|
|19|Learning rate/3|0.00025577|26|
|19|Parent KL|0.00006036|11|
|43|Native|0.00465287|113|
|43|Learning rate/3|0.00045574|35|
|43|Parent KL|0.00014054|26|

The KL penalty reduced mean validation policy divergence by about97% versus
native in both seeds. This percentage describes probability drift, not a97%
profit improvement or generalization repair. Lower rate and KL each improve
the final validation return over the matched native arm in both seeds.

Neither method consistently beats the unchanged actor: both exceed13.80849%
in seed19 and remain below it in seed43. KL increases TRAIN return over the
parent in both seeds, but validation does not follow in seed43. Bounded policy
movement therefore mitigates drift without proving that the remaining critic
preferences reliably select profitable action changes. KL beta1 is one fixed
diagnostic setting, not an optimized regularization parameter.

The source replay, initial representation and dates are shared; the replicas
vary critic correction and update sampling, not independent market histories.
Early-close counts alone do not establish quality; native diagnostics remain
in each account artifact and no early-close fix is claimed here.

## Disposition

Keep the untouched parent and all new endpoints. Adopt the KL and changed-action
measurements as refinement diagnostics, not automatic promotion criteria.
The next technical question is which residual action changes hurt despite
small KL, and whether they originate in critic action ordering or small
classification margins. Exact account attribution and critic-margin inspection
can use these stored weights and traces without another training run. No new
training is launched by this results report.

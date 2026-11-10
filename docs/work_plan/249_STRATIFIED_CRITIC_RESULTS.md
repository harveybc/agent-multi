# Stratified Critic Results

## Verified Completion

Design248 finished2026-11-10T02:15:11.391368Z, in510.40 seconds (8m30s).
Runtime45b035cf; four arms,4096 updates each. Worker and automatic collector
exited0. All1500 collection files authenticated against receipt identity
`94869a1f0c1462cfd121e293db0943c4c984f8219c016c560e52921348c9bf76`.
Source69/70 checkpoint hashes are unchanged. Full TRAIN actor logits and all
frozen parameters remained exact in every arm. No actor update or promotion.

Evidence: `/var/tmp/sac70-stratified-critic-20261109`.
Public subset: [curves](generalization/stratified_critic_20261110/CURVES.csv),
[plot](generalization/stratified_critic_20261110/FIT_MONITOR.png).
SOURCE_DONE authenticates the full source collection, not only that subset.

## Coverage And Pairing

All16 quarterly episodes completed,4278 native counterfactual suffixes plus
48 witness reruns. Maximum witness return/cash discrepancy0.0. Reference
branches reproduced native terminal cash. Native orders and costs unchanged.

| State | Fit2020-2022 | Monitor2023 |
|---|---:|---:|
| Flat, both directions admitted |384|128|
| Flat, restricted |96|32|
| Open long |230|63|
| Open short |365|128|
| Total |1075|351|

Strata with fewer than the declared quota retain their actual available count,
without duplication or target-based filtering. Fit label support ends before
monitor window support starts. Both scopes use identical minibatch schedule
hashes within each seed and identical initial diagnostic metrics.

## Fixed Endpoint

Pair accuracy is correct ordering of informative action-return pairs using
the minimum of the two critics. It is NOT a trading win rate.

| Seed | Scope | Fit pair accuracy | Monitor pair accuracy | Monitor gap RMSE | Monitor mean regret |
|---|---|---:|---:|---:|---:|
|19|Heads|54.66%|54.70%|0.378132|0.192022|
|19|Representation + heads|66.51%|51.04%|0.394906|0.185488|
|43|Heads|54.92%|54.44%|0.378243|0.191440|
|43|Representation + heads|67.02%|51.31%|0.394975|0.188401|

Gap RMSE and regret are in discounted-reward units, not dollars. Aggregate
mean regret improves modestly with representation training despite worse pair
accuracy and gap RMSE; the objectives do not rank all errors identically.

### Monitor By State

| State | Heads, seeds19/43 | Representation, seeds19/43 |
|---|---:|---:|
| Flat permitted pair accuracy |56.25 /56.51%|56.51 /55.99%|
| Open long pair accuracy |51.59 /50.79%|38.89 /40.48%|
| Open short pair accuracy |53.91 /53.13%|48.83 /49.61%|

The open-long monitor has only63 states; these rows are correlated, not63
independent trades. Restricted-flat labels have no informative ordering and
are reported separately, not counted as correct decisions. Their equivalence
gap RMSE also worsens with representation training (about0.031 to0.115).

## Learning Curves And Interpretation

Both representation arms improve monitor pair accuracy initially:56.40/56.79%
at64 updates versus48.96% before calibration. At4096 it is51.04/51.31%, while
fit accuracy keeps rising. Open-position monitor gap RMSE diverges from fit,
most clearly after1024 updates. These curves support overfitting during this
critic calibration, not a claim that the data contain no useful signal.
No early checkpoint is selected retrospectively; the declared4096 endpoint
remains the comparison.

The32-example result247 demonstrated fitting capability. This broader contrast
does not establish that unfreezing the complete critic representation is a
generalization fix. It also does not identify a single causal defect: temporal
distribution change, state-dependent action values without state inputs, and
which encoder modules drift still require separate controlled diagnoses.

Monitor2023 was excluded from THESE gradient updates but was used previously
by the source policy, representation, scaler and calibrated critic. It is not
an untouched holdout for the whole pipeline. Quarterly reset/settlement is a
diagnostic convention; anchors within a quarter have overlapping continuations.
Do not compare these Q levels directly with full-TRAIN suffix experiments.
No2024 trading evaluation or TEST scoring occurred in this contrast.

## Next Step

Keep the trading champion intact. Inspect critic drift and action-value
conflicts by operational state, focusing on open-long CLOSE/HOLD decisions.
Then compare partial unfreezing of the last critic block against the two
recorded scopes on the same panel, preserving actor, schedule and inputs.
That follow-up requires an explicit pinned module map and gradient tests;
this report does not launch it. Do not start another feature-removal pass
or promote the diagnostic critic checkpoints on the basis of fit accuracy.

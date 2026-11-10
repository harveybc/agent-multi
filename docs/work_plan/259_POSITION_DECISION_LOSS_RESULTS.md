# Position Decision-Loss Results

2026-11-10. [Design258](258_POSITION_DECISION_LOSS_CONTRAST.md) completed:
16/16 new arms,131.89 seconds on the primary GPU,158 collected files verified.
Eight archived geometric-interface SmoothL1 arms from257 are reused as controls.
All source actor tensors, frozen encoders, input windows and native order
mechanics are unchanged. This is a head-calibration diagnostic, not a new
portfolio backtest or a deployed trading policy.

## Comparison Contract

Same2342 positions in866 banks:1781 fit2020-2022,561 monitor2023. The source
representation already used2023; no2024/TEST evaluation here. Both seeds19/43,
progress9/surface81, actual/constant market embeddings,4096 updates per arm.
Every new arm matches its archived control's initial tensors, axes, primary
minibatch schedule and parameter count. No row, window or difficult case removed.

SIGN uses cost-weighted logistic loss for CLOSE/HOLD. SIGN_PAIR gives equal
weight to SIGN and an opposed-position ranking term within the same native
market/direction bank. Fixed fit-bank normalizers: sign0.31864168713385727,
pair0.04446376888093006. All labels remain whole-account native continuation
gaps under the frozen source actor, not a newly invented optimal exit target.

Scores from these objectives are not Q values or calibrated probabilities.
Raw score-to-Q RMSE is deliberately absent. Positive mean-twin score requests
CLOSE, <=1e-8 requests HOLD. Regret and local harmful closes remain separate.

## Main Results

Monitor bank-balanced regret is in discounted reward units; lower is better.
Accuracy is the correct sign of the account CLOSE-minus-HOLD gap. Close counts
below sum the two seed runs for descriptive comparison: they are not additional
independent positions or executed portfolio orders.

| Interface | Loss | Monitor regret, mean | Accuracy, mean | Harmful / effective closes | Beneficial closes |
|---|---|---:|---:|---:|---:|
|progress9|SmoothL1 control|0.133868|54.99%|357 /761 (46.91%)|404|
|progress9|SIGN|0.131140|54.72%|379 /798 (47.49%)|419|
|progress9|SIGN_PAIR|0.132324|60.87%|218 /567 (38.45%)|349|
|surface81|SmoothL1 control|0.132305|51.96%|306 /609 (50.25%)|303|
|surface81|SIGN|0.133515|51.52%|321 /636 (50.47%)|315|
|surface81|SIGN_PAIR|0.128891|60.96%|241 /608 (39.64%)|367|

Surface81 SIGN_PAIR reduces mean monitor regret2.58% versus its archived
SmoothL1 control. Harmful closes fall21.24% by count and beneficial closes
increase21.12%, with almost identical total closes. This is not merely avoiding
all closes. It remains a two-seed descriptive result, not a significance claim.
Progress9 SIGN_PAIR also lowers harmful fractions, but its beneficial closes
fall along with activity. SIGN alone is not a general improvement.

### Surface81 Per Seed

| Seed | Loss | FIT regret | Monitor regret | Harmful | Beneficial | Effective closes |
|---|---|---:|---:|---:|---:|---:|
|19|SmoothL1|0.121509|0.138716|146|134|280|
|19|SIGN|0.119436|0.136103|157|155|312|
|19|SIGN_PAIR|0.141162|0.130983|111|175|286|
|43|SmoothL1|0.116957|0.125894|160|169|329|
|43|SIGN|0.126529|0.130927|164|160|324|
|43|SIGN_PAIR|0.139333|0.126798|130|192|322|

Local harmful/beneficial counts improve in both seeds under SIGN_PAIR, but
whole-account regret improves only in19 versus SmoothL1;43 slightly worsens.
Those outcomes can differ because closing affects subsequent opportunities.
FIT aggregate regret worsens under the paired objective. Do not describe this
as dominance or claim that smaller FIT-monitor separation alone fixes overfitting.

## Constant-Market Controls

Same position queries, with only the frozen embedding replaced by its fit-only
bank-balanced mean. The controls are part of design258, not added afterward.

| Interface | Loss | Actual-market monitor regret | Constant-market monitor regret |
|---|---|---:|---:|
|progress9|SmoothL1|0.133868|0.136326|
|progress9|SIGN|0.131140|0.136017|
|progress9|SIGN_PAIR|0.132324|0.130789|
|surface81|SmoothL1|0.132305|0.129143|
|surface81|SIGN|0.133515|0.135065|
|surface81|SIGN_PAIR|0.128891|0.129421|

Surface81 SIGN_PAIR now beats its constant-market counterpart in both seeds,
but by only0.41% on mean regret. Constant-market SIGN_PAIR also reduces harmful
closes substantially (239/605,39.50%). Much of the benefit therefore remains
compatible with geometry-only learning; useful market contribution is not yet
established robustly. Do not blame the input data or declare them sufficient
from this diagnostic alone.

## Conflict Strata And The Remaining Witness

Post-endpoint reporting partitions complete banks by whether their native
targets contain both positive and negative gaps. It does not select training
rows or change the objective. There are110 fit and29 monitor conflicting banks.

| Surface81 loss | FIT conflicting | FIT other | Monitor conflicting | Monitor other |
|---|---:|---:|---:|---:|
|SmoothL1|0.123920|0.118294|0.128422|0.132938|
|SIGN|0.114615|0.124659|0.106384|0.137935|
|SIGN_PAIR|0.097372|0.148838|0.107684|0.132346|

The pair term improves FIT on the intended conflicting cases but sacrifices
other cases. Its monitor conflict regret improves versus SmoothL1, not versus
SIGN alone. Keep this tradeoff visible instead of selecting one headline.

The archived critical witness requires CLOSE,HOLD,HOLD. No new endpoint passes
all three. With actual market, SIGN_PAIR chooses HOLD,HOLD,HOLD in both interfaces
and both seeds. However its mean scores correctly rank the first position above
the other two:

| Interface | Seed | Mean scores for the three positions |
|---|---:|---|
|progress9|19|[-0.126497,-0.639765,-0.669323]|
|progress9|43|[-0.053540,-0.375880,-0.394333]|
|surface81|19|[-0.042281,-0.103604,-0.103670]|
|surface81|43|[-0.154104,-0.369272,-0.378109]|

This separates relative ranking from the zero-threshold decision. The pair
term depends on z_i-z_j, so it is invariant to adding a common offset. The
SIGN component anchors that offset, but this finite fit has not placed it
correctly for the witness. This is a diagnostic clue, not proof that changing
one offset fixes all cases. No threshold was fitted to this monitor witness.

## Verification And Artifacts

- Runtime8944851b. Experiment identity
  `a14b0d897d0523b53815ed7069d9c16bd0590712a14e0b568260a8a4a490d1e3`.
  Finished2026-11-10T16:03:38Z; both worker and collector exited0.
- Report identity
  `c50b4c376d584d3942724d23903c40f987ee781fc84db14ec6663a4f6478e5a6`.
  Authenticates all three parent collections before comparing their endpoints.
-20 tests passed before launch on coordinator and worker;21 final focused tests
  including the conflict-stratum reporter pass on coordinator. Native original
  position accounting, same-market pairing, no-monitor normalization, extreme
  scores, gradient directions, neutral labels and score semantics are covered.
-Full-TRAIN source actor outputs and source tensors/hash unchanged. Cached
  embedding reconstruction error remains <=2.39e-7. Every trained head changed
  from its initialization and reloads with identical per-instance scores.
-No failure/retry or silent arm omission. The vendored ruff executable is absent
  in this worktree; lint was not run. No lint-clean claim is made.

Public compact evidence: [summary](generalization/position_decision_report_20261110/SUMMARY.csv),
[all endpoints](generalization/position_decision_report_20261110/ENDPOINTS.csv),
[conflict strata](generalization/position_decision_report_20261110/STRATA.csv),
[witness scores](generalization/position_decision_loss_20261110/WITNESS.json),
[surface market curves](generalization/position_decision_loss_20261110/plots/surface81/market/CURVES.png).
Other per-interface/context plots, configuration, constants and DONE receipts
are in the same folders. Full weights/schedules remain in the sealed external
collection; the repository copies are intentionally compact subsets.

## Decision And Next Step

Preserve the paired objective and its evidence as a promising diagnostic,
without replacing the source actor or the phase2 evaluator. No portfolio return
or Sharpe is newly claimed. No further training is running from this queue.

Next narrowly scoped experiment: freeze these heads and measure whether an
intercept/decision-threshold calibration determined exclusively on fit data
improves offset errors while preserving the gains. Compare the zero-threshold
baseline and constant-market controls, publish all seeds and conflict strata,
and inspect the monitor only after fixing the threshold rule. Never choose an
offset from the known monitor witness. This separates calibration from ranking
and representation before any further network growth or actor update.

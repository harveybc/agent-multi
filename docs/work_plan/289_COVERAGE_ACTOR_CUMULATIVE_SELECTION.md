# Coverage Actor Qualification And Cumulative Selection

2026-11-11. Owner authorizes qualification of288 narrow19/43 and a fresh
primary19 cumulative phase2 deletion/return pass. Fixed endpoint8, no fitting
or intermediate checkpoint search. Keep287 and all288 sources unchanged.

## Contract And Evidence Gates

| Requirement | Verification |
|---|---|
| Correct source | Coverage experiment schema, narrow arm, endpoint8, seeds19/43, source receipts |
| Preserved learning | Only actor head tensors differ; export and cold state/logits/accounts exact |
| Economic identity | Full actions, trades, equity and close counterfactuals reproduce288 |
| Full phase2 return | Start69 inputs/28 groups; restore all51 historical removals within120 admitted inputs |
| Sequential rules | Ordered deletion accepts improvement or exact tie if eligible; return requires strict improvement |
| No silent weight loss | Donor expansion control parity, retained tensors and packed cold replay |
| Automation | Qualification first, selection and independent report in finite sequential jobs; authenticated collection |

Keep fixed TRAIN scaling, window13/stride2, original full data support,
TRAIN2020-2023 and VALIDATION2024. No TEST, rolling scaling, state feedback,
source-data substitution or live changes. Score is validation return/week;
both periods require more than2 orders/week. Preserve coherent encoding blocks.
Archived donor-assisted readmission is a frozen screening experiment, not
proof against improvement after joint retraining.

Expected baseline:TRAIN13.201811650001527%,VALIDATION14.693261850000072%,
741/206 trades. Qualification must reproduce those accounts before selection.
Retain seed43 as a comparison bundle. If no features change, do not refit or
repeat an identical pass by default. Publish decisions, all metrics, timing,
weights and the exact next action in the master plan.

## Qualified And Launched

31 focused tests and scoped Ruff passed. Runtime2160a100 is immutable.
Both candidates qualified in29.80seconds, with exact full action/trade/equity
and close-counterfactual replay. Only four head tensors changed versus287;
337 other actor tensors remain exact. Cold state, logits and accounts match.
Six TRAIN signals and no validation signals changed in each candidate.

| Seed | Exported model SHA256 |
|---|---|
|19|`c0d681205252af731f68ad5725428ce9ca993431c00949482d39d8113d004bc3`|
|43|`cf4ef9d964fb22c7cd7277103dfb4931f1589685e7cb4447d208950cfc1ab7cf`|

Qualification:`5c6d28234200f58141ccbccec53b1d97d89ee7642f0cc155bfe559ba090a1943`.
Fresh selection:`b04297ef6c108ae4be4d62e6d95995add99c5af28819bb0ede3ecda31c25ef96`.
The primary RTX5090 passed readiness; selection, independent weight/decision
report and finite authenticated collection are sequentially automated.

## Completed Selection

The selection took138.80seconds (145.41 including startup); both sequential
tasks and the finite collector completed successfully.135 selection/report files
and68 qualification files are authenticated. No optimizer update occurred.

| Item | Result |
|---|---|
| Initial / after deletion / final |69 /68 /69|
| Retained input groups |28|
| Evaluations |113: reference +65 deletion blocks +47 return blocks|
| Return coverage |51 previously absent features plus the one just removed|
| Accepted deletion |`px.range_12`|
| Accepted historical readmission |`RAW_BAR_LOG_RETURN`|
| Independent decision replay |exact|
| Current actor tensors retained |313 exact; index buffers checked separately|
| Packed cold logits and accounts |exact|

Reinsertion of `px.range_12` was tested and rejected. All historical phase2
removals were considered, not just this pass's deletion. There were no ties;
deleting `cot.eur.net_share` failed the activity gate. The same final feature
count does not mean no change: a historical channel replaces a current one.

Validation return/week progresses0.272097441666668 ->0.2724504638888902
after deletion ->0.30252109722222015 after readmission. Native full accounts:

| Metric | Before TRAIN | After TRAIN | Before VAL | After VAL |
|---|---:|---:|---:|---:|
| Return % |13.201812|13.913055|14.693262|16.336139|
| Net USD, initial10000 |1320.18|1391.31|1469.33|1633.61|
| Sharpe |0.651837|0.685588|2.179247|2.400526|
| Maximum DD % |7.930285|9.548436|4.044574|3.257322|
| Orders |741|739|206|207|
| Orders/week |3.5455|3.5359|3.8148|3.8333|
| Win rate % |61.8084|61.7050|58.2524|58.9372|

Final direction and closing detail:

| Metric | TRAIN | VALIDATION |
|---|---:|---:|
| BUY orders / winners |224 /152|23 /14|
| SELL orders / winners |515 /304|184 /108|
| Early / TP / SL / forced exits |54 /403 /220 /62|6 /110 /66 /25|
| Helpful / harmful early exits |25 /29|4 /2|

Validation gains1.6428774 percentage points, with better Sharpe and lower DD.
TRAIN return increases but DD worsens1.618151 percentage points. Early-close
counts do not establish improved close decisions: harmful TRAIN exits remain29
and helpful decrease28 to25. Preserve the prior reference as an alternative,
not just the higher-return candidate. This remains a development selection on
the stated score, not a claim of independent generalization or global optimality.

The expansion control's maximum logit difference is1.9073486328125e-6,
zero action disagreements across30234 valid windows and exact native accounts.
This is tolerance-qualified donor expansion, distinct from the later bit-exact
packed cold reload. The current head was never overwritten by archived weights.

## Saved Artifacts And Next Action

Standalone qualified pre-selection SAC bundles remain in the private
`sac69-coverage-qualification-20261111/critic_narrow_{19,43}/bundle/` roots.
The changed-input candidate is saved separately in
`sac69-cumulative-coverage-20261111/bundle/`, with its69-column recipe, selected
order, scaler and restored learned connection. Its packed artifact is for
inference; it is not yet a trainable SAC replacement.

Next: export the new input composition into a compatible trainable bundle,
first proving exact/tolerance-qualified signal and full-account parity while
preserving the learned actor. Do not silently reuse the old critic as though
the input identity had not changed. Qualify same-policy TRAIN calibration before
further actor learning; keep a frozen new-candidate control. Do not default to
from-scratch training or repeat this identical frozen selection. Live services
are unchanged.

Evidence: [manifest](generalization/cumulative_coverage_20261111/MANIFEST.json),
[qualification](generalization/cumulative_coverage_20261111/QUALIFICATION.json),
[full results](generalization/cumulative_coverage_20261111/SUMMARY.json),
[feature ledger](generalization/cumulative_coverage_20261111/FEATURE_LEDGER.csv),
[selection trace](generalization/cumulative_coverage_20261111/SELECTION_TRACE.json),
[weight audit](generalization/cumulative_coverage_20261111/WEIGHT_SPACE.json).

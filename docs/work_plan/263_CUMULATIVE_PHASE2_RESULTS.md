# Cumulative Phase2 Return Results

2026-11-10. [Design262](262_CUMULATIVE_PHASE2_RETURN.md) completed on the
physically verified RTX5090. Selection runtime6c01d718. No training or TEST.

## Result

**69 ->69 ->69 inputs; the same28 learned groups remain selected.**
The current inference checkpoint stays unchanged. No candidate was promoted.

| Scope | Count |
|---|---:|
| Original phase2 universe |120 columns|
| Starting actor |69 columns|
| Previously removed phase2 columns |51|
| Deletion evaluations |65 coherent blocks|
| Cumulative reinsertion evaluations |46 coherent blocks, all51 absent columns|
| Total distinct account evaluations |112, including control|
| Accepted deletions / insertions |0 /0|
| Runtime inside worker |137.20 seconds|
| Native full-support windows |30,234|

Coherent calendar and probability blocks explain why block counts differ from
column counts. Nothing from the phase1-invalid pool was silently reintroduced.
Every deletion lowered the declared validation score; one also failed activity.
Of the46 return candidates,45 lowered the score and one tied it. Insertion
requires strict improvement, so the tie remains absent. There was no deletion
tie suppressed by an undeclared tolerance.

## Preserved Trading Results

These reproduce the existing selected69 actor, not a new gain from this pass.
TRAIN2020-2023; VALIDATION2024; unchanged real historical inputs and costs.

| Metric | TRAIN | VALIDATION |
|---|---:|---:|
| Net return |10.7921%|14.5911%|
| Net profit, initial USD10,000 |1,079.21|1,459.11|
| Sharpe |0.5396|2.1656|
| Maximum drawdown |8.3906%|4.0447%|
| Trades |750|206|
| Orders/week |3.5885|3.8148|
| Win rate |61.20%|58.25%|
| Buy / sell trades |232 /518|24 /182|
| Buy win rate |65.95%|62.50%|
| Sell win rate |59.07%|57.69%|
| Agent early closes |60|7|
| Helpful / harmful early closes |30 /30|4 /3|

Validation remains short-heavy:182/206 trades. Early closes remain7/206;
the selection pass did not repair or change the separate close-controller
research question. Do not infer a desired early-close rate from these counts.
The fixed-position counterfactual diagnostic is not full account continuation.

## Most Competitive Changes

Score = validation return percentage points/week; reference0.2702053639.

| Intervention | Score | Decision |
|---|---:|---|
| Remove px.range_12 |0.2621488759|Keep feature|
| Remove wv.r1.energy_D1@diff |0.2618858972|Keep feature|
| Remove BB_position |0.2582841148|Keep feature|
| Restore RAW_ELAPSED_HOURS |0.2702053639|Tie: do not add|
| Restore OPEN@diff |0.2641160194|Do not add|
| Restore fac.fx_f1.ret_1 |0.2584468009|Do not add|
| Restore fac.fx_f1.kalman_slope |0.2568405093|Do not add|
| Restore RAW_CLOSE_LOG_RETURN |0.2557127528|Do not add|

Removing cot.eur.net_share also failed the unchanged activity gate and reduced
the score to0.0518183167. The full ledger includes all changes, not only this
small explanatory table.

## Restoration And Verification

Seven absent columns use the compatible76-input ancestor;44 use archived
original120 learned connections. Three absent groups were reconstructed:
px.rv, px.close_loc/px.close_loc_2 and px.close_loc/px.close_loc_6.
Donor scaling conversion and neutral-response preservation are explicit in262.
This is a frozen donor-assisted screen, **not jointly retrained readmission**.
A donor failure does not establish that a feature cannot help after matched fit.

- All313 retained actor tensors compared exactly against the preserved actor,
  including surviving first-convolution slices and original fusion columns.
  All28 original index buffers were verified against the expanded ordering.
- All30,234 native windows: zero action disagreements; maximum logit error
  1.9073486328125e-6. Original TRAIN/VALIDATION account metrics reproduce exactly.
- Final packed inference export cold-reloads exactly; full accounts reproduce.
  The original physical69 bundle remains preferred since no input changed.
- All130 collected files pass their hashes. A separate CPU report independently
  replays all112 selection decisions from sealed cache measurements and obtains
  exactly the same trace and final selection.
-14 focused tests pass, including historical return without new deletions,
  latest-baseline decisions, tie/activity rules, coherent blocks, changed scaler
  refusal, zero/missing donor refusal, removed-group bias handling and cached
  restart. The real GPU pass verifies full production support, not just probes.
- Worker and automatic collector exited0. Source checkpoints were unchanged.
  This is a completed short job; no long-running training is implied.

The preparatory command initially supplied the RESULT digest while checking
SOURCE_RECEIPT. It stopped before creating a run or evaluating anything. The
checker now authenticates RESULT and follows its explicit artifact hashes.
The executed run itself had no failed unit or retry.

Run identity:
`20a5c4ece3730d68ad0c9ee0e5aec1eb63af3d4ad39a1c4f2349fa9bc94920d8`.

Evidence:
[summary](generalization/cumulative_phase2_20261110/SUMMARY.json),
[every column](generalization/cumulative_phase2_20261110/FEATURE_LEDGER.csv),
[full decision trace](generalization/cumulative_phase2_20261110/SELECTION_TRACE.json),
[weight audit](generalization/cumulative_phase2_20261110/WEIGHT_SPACE.json),
[copy hashes](generalization/cumulative_phase2_20261110/MANIFEST.json).

## Exact Resume Point

Do not run the same unchanged mask pass again expecting a different result.
No new subset needs physical compaction or retraining after this pass. Preserve
the69 checkpoint and its28 groups as the comparison baseline and Retsu's model.

The next model experiment is qualified bounded refinement of this preserved
policy, not the previously degrading long head-only schedule. New refinement
must follow the adopted fresh-critic-head/own-policy-calibration rule; never
reset the actor or reset anything during inference or resumed evaluation.
Require alternate-action ranking and native trading/activity checks, not just
smaller critic RMSE. The experimental close fields and rejected threshold
calibration remain separate and are not quietly substituted into this baseline.

Matched retraining of restored features is a separate possible contrast,
because this screen qualifies frozen donor reuse only. Real historical coverage
and synthetic-data compatibility remain in261; neither is a reason to replace
missing real-time values with fabricated inputs. Retsu owns live acquisition
under the published handoff; it does not displace the research resume point.

# Interior FIT Projection

Owner-approved after293. Same physical69 inputs,28 groups, frozen actor
reference, calibrated critics,63 fixed FIT error targets,26 protected FIT
decisions, two schedules19/43 and eight attempted head updates per schedule.
Reuse the sealed293 boundary-projection control; do not retrain controls.

## Frozen Change

Only tighten the **linear proposal** constraint. For each protected source
action a versus b, calculate once from source FIT logits:

`buffer[a,b] = 8 * finfo(source_dtype).eps * (1 + abs(z[a]) + abs(z[b]))`.

Project onto `J_margin * delta >= source_margin + buffer - current_margin`,
with the same nonpositive corrective-loss slope constraint. Eight rounding
units is a prespecified engineering cushion, not an optimized coefficient or
a proven bound on the entire nonlinear network. It uses no monitor/validation
outcome. It does not change logits, targets, inputs or native acceptance checks.
Require the full-space linear inequalities to satisfy their buffered bounds
within absolute1e-10 before nonlinear search; solver success alone is insufficient.

The nonlinear gate remains: unchanged protected actions, original source
margins within1e-7, correction loss nonincrease and all existing trust budgets.
Backtracking still has24 trials and minimum FIT margin movement1e-6. No tolerance
relaxation, multiplier search, extended training or fallback to unguarded steps.
If projection is infeasible it rejects. Log every trial, not just the last one.
The fixed source buffer is not accumulated over updates. Source and candidates
use identical actor-only trainability and full-panel batch shape.

Record first-step feasibility and requested/actual correction before proceeding
to the remaining fixed attempts. Evaluate the fixed endpoint8 and snapshots0/1/8;
do not select by validation. Critics, feature extractors, fusion, core, alpha
and normalization stay frozen. Rejections restore weights and optimizer moments.

## Requirement To Evidence

| Requirement | Test / evidence |
|---|---|
| Numerical cushion is fixed and FIT-only | Exact float32/float64 formula, positive finite bounds, unprotected perturbation invariance, invalid inputs refuse |
| Proposal tightens rather than relaxes | Known halfspace interior solution, original preservation test unchanged, infeasible constraints reject |
| Nonlinear enforcement remains authoritative | Toy nonlinear boundary failure/interior success, full trace of real backtracking with unchanged budget and floors |
| One changed experimental variable | Exact293 error labels/weights, guard, actor/support/schedules/critics; independent checkpoint audit |
| No numerical movement mislabeled as learning | Minimum movement, loss/probability/error changes, preserved margins, accepted optimizer counters |
| Economic outcome | Full TRAIN/VALIDATION accounts, directions, activity, wins, drawdown, early-close and chronological attribution; exact cold replay |

This remains an experimental own-policy, advantage-weighted offline actor
correction, not an unchanged SAC loss or an expert-guided ZigZag model.
No TEST, live promotion or repeated floating pass on an unchanged actor.

## Completed Results

Runtime `3fac45e8`; identity
`4cca31b445a8a069a1ad59fec11ee4e0b772c6998a558f7713d08dbeeb5fbfc8`.
The finite queue ran both arms in28.13seconds on the primary GPU, then full
chronological account attribution in5.47seconds.114 worker files and15 report
files passed collection hashes.41 focused tests and scoped Ruff pass.

The fixed buffer ranges from9.752604e-7 to2.001812e-5. All16 projections
and all16 nonlinear checks accept the full proposed step, without backtracking.
Worst protected margin change across accepted checkpoints/steps is **positive**
9.462237e-7; all26 protected actions stay unchanged. The1e-7 acceptance tolerance
was not relaxed. FIT margin RMS movement stays0.001609-0.001879 per step, below
the0.002 ceiling. Maximum cumulative KL0.000252603 and relative parameter
drift0.00170810 remain below their0.001 and0.01 ceilings.

| Arm, fixed endpoint8 | Accepted / attempted | TRAIN return % | VAL return % | TRAIN / VAL orders | VAL Sharpe |
|---|---:|---:|---:|---:|---:|
| Frozen reference /293 boundary control |0 /8 for293|13.913055|16.336139|739 /207|2.400526|
| Interior19 |8 /8|14.756076|16.342145|741 /207|2.401048|
| Interior43 |8 /8|14.756076|16.342145|741 /207|2.401048|

Both endpoint8 policies produce identical native accounts. The schedule seeds
are matched sampling variations, not independent market samples. Point1 still
has unchanged accounts in both arms; endpoint8 was fixed before training and
was not selected from validation.

## What Changed

For both arms, the preferred-action probability rises on53 of63 fixed FIT
errors and falls on10; diagnostic MONITOR has19 increases and4 decreases out
of23 errors. None of these original panel errors crosses its greedy boundary.
Mean preferred probability changes are0.000378970/0.000376635 in FIT and
0.000171620/0.000168338 in MONITOR for schedules19/43 respectively.
Thus correction is partial at the probability level, not a claim that all
63 panel errors have been repaired. The shared head also changes other rows.

Full-period signals change on6 TRAIN and2 VALIDATION rows. Native chronological
attribution identifies2 effective TRAIN changes and1 effective VALIDATION
change; all four arm/period delta reconciliations have exactly0 USD error:

- TRAIN: one additional SELL and one additional BUY, both profitable. Account
  contributions including subsequent effects are40.015380 and44.286715 USD,
  summing84.302095 USD (+0.84302095 percentage points of initial balance).
- VALIDATION: a changed SELL entry accounts for0.600545 USD
  (+0.00600545 percentage points). Total trades stay207 because action sequences
  and position occupancy affect later entries. This tiny gain is not evidence
  of broad generalization improvement.
- Early-close behavior/quality does not improve:54 closes in TRAIN
  (25 helpful/29 harmful) and6 in VALIDATION (4 helpful/2 harmful).

No harmful **effective account interventions** were found in these paired
chronological comparisons. This does not mean every probability change, every
standalone hypothetical trade or every future use of this correction is helpful.

| Endpoint metric, both arms | TRAIN2020-2023 | VALIDATION2024 |
|---|---:|---:|
| Profit USD |1475.607640|1634.214470|
| Sharpe |0.718355|2.401048|
| Max drawdown % |9.548364|3.257301|
| BUY / SELL orders |225 /516|23 /184|
| BUY / SELL wins |153 /305|14 /108|
| Total win rate % |61.8084|58.9372|
| Orders/week |3.5455|3.8333|
| Early / TP / SL / forced closes |54 /405 /220 /62|6 /110 /66 /25|

## Audit And Disposition

An independent audit authenticates262 files and all four saved checkpoints.
It reconstructs the26 protected rows,63 FIT targets/weights and numerical
buffer without the training helpers; verifies no monitor targets, identical
source floor/tolerance, full backtracking traces, correct accepted optimizer
counters and frozen components; reproduces endpoint probabilities/margins.
Canonical cold reloads match emitted logits and full accounts exactly. Source
weights and all earlier sealed evidence remain unchanged.

The local feasibility obstruction is resolved in this contrast. Adopt the
fixed interior proposal for this corrective refinement path, retaining all
nonlinear gates; do not generalize it to unrelated optimizers automatically.
Keep both endpoint8 checkpoints, using schedule19 as the conventional primary,
43 as replication, and keep the original model as control. No live promotion.

Next: qualify/export the changed actors with exact cold/native parity, then
perform a fresh phase2 deletion-and-cumulative-readmission pass using primary19.
Readmission includes the historical phase2 removals, not just this pass's
deletions. No default retraining or restarting from random weights. Refresh
own-policy continuation targets before any later corrective training, since
the policy has now changed. Do not extend this fixed experiment's step budget.

Evidence: [manifest](generalization/interior_fit_20261111/MANIFEST.json),
[summary](generalization/interior_fit_20261111/SUMMARY.json),
[independent audit](generalization/interior_fit_20261111/AUDIT.json),
[all steps](generalization/interior_fit_20261111/STEPS.json),
[return plot](generalization/interior_fit_20261111/RETURNS.png).

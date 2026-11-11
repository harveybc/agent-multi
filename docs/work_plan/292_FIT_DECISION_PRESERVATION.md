# FIT Decision Preservation Contrast

Owner-approved2026-11-11 after291. Preserve the physical69-input/28-group export.
Reuse290's two calibrated critics,151 native continuation anchors and two
eight-step batch128 schedules. No critic recalibration or dataset change.

## Prespecified Design

Protect only FIT2020-2022 rows where the source actor's greedy action is the
strictly best of its three measured frozen-policy continuations, by a native
reward gap above1e-8. Include correct NOP/close decisions, not only openings.
MONITOR2023 and VALIDATION2024 never define protection, targets or step acceptance.
Seal the protected row/action list before any update. No agent-state input.

Two arms per seed19/43, both using the same sealed critic SAC objective:

1. `guard_only`: ordinary AdamW proposal plus the new nonlinear preservation
   check, with existing backtracking. This is the immobility control.
2. `project_guard`: nearest Euclidean parameter step to that same AdamW proposal
   satisfying linearized protected margins and a nonpositive objective slope.
   Then apply the same nonlinear protection, objective and trust checks.

For each protected action a and other action b, require
`logit(a)-logit(b) >= source_margin(a,b)-1e-7` and unchanged greedy action.
The reference is the original source margin, not a moving floor that can erode
over steps. Preserve the previous284 trust budgets and24 backtracks. A useful
accepted step must have FIT margin RMS at least1e-6, real parameter movement,
and sampled objective not increasing beyond1e-10. This extra floor prevents
near-zero numerical updates from being counted as useful refinement.

Solve the strictly convex Euclidean projection with SciPy SLSQP in the constraint
row space obtained by SVD. Row normalization and proposed-step scaling keep the
problem numerically conditioned; the original full-space inequalities are checked
afterwards. Use SVD relative cutoff1e-10, SLSQP ftol1e-12/maxiter300, normalized
feasibility tolerance2e-8. Failed, infeasible or zero solutions are recorded as
rejections; they never authorize an unconstrained fallback. The nonlinear gate
is authoritative. Rejected proposals restore optimizer moments and parameters.

Check first-step feasibility with the real panel before executing the remaining
seven updates. Do not extend training if protection is inactive or all steps
reject. Assess the fixed endpoint8, not a validation-selected checkpoint. Save
points0/1/8 and cold-reload evidence. Old290 controls are reused from sealed files,
not retrained. Sequential GPU arms then symmetric account attribution on CPU.

## Test Traceability

| Requirement | Check |
|---|---|
| Correct protection membership | Strict source-optimal FIT only; ties/monitor excluded; future target perturbation cannot alter membership |
| Stable preservation floor | Source margins remain the floor across repeated updates; decisions cannot flip within tolerance |
| Correct projection | Known halfspace solution, already-feasible identity, duplicate/rank-deficient constraints, infeasible zero row, zero-only feasible solution, loss-descent inequality |
| No hidden learning bypass | Failed/zero projection rejects; rejected steps restore optimizer moments; frozen actor representation/critic/alpha and buffers exact |
| Matched causal contrast | Source/export/critic/support/schedule hashes; exact initial native accounts; identical Q to290 |
| Nontrivial update | Accepted/rejected counts, parameter displacement, FIT margin RMS, preserved margin minima, objective and trust budgets |
| Real trading consequence | Full TRAIN/VALIDATION return, Sharpe, DD, activity, directions, wins, early closes; chronological account attribution and cold replay |

Protection applies only to sampled FIT continuations of the source policy, not
to every market state or arbitrary later policy. Preserved margins alone do not
guarantee account profit or generalization. No TEST access or live promotion.

## Baseline Preflight Correction

The first run stopped before accepting any update. Fixing the guard floor before
setting actor-only trainability gave a different GPU kernel path: max logit
change9.536743e-7, zero action changes. Loading the calibrated critic under the
same actor-only scope gives exactly zero difference. Set actor-only scope before
capturing the source panel and require bitwise-equal panel logits after each
critic load. Compute constraint Jacobians on the same full panel, not a smaller
protected-only batch. Keep the1e-7 preservation tolerance unchanged. The failed
run is retained; the corrected runtime and output have new identities.

## Completed Results

Runtime `d8fc65c9`; identity
`2bd36cbeb968ea1f3b16f5e7333c732d6e9762750ae726a3e835d430578ebafb`.
The four arms finished in53.00seconds on the primary GPU; symmetric account
reporting took1.85seconds. Worker and collector exited0.27 tests preceded launch;
28 tests pass after the diagnostic-only consistency correction described below.
Scoped Ruff passes. No new critic fit, input change or live promotion.

The fixed protection contains26 FIT anchors:6 source NOP,8 BUY and12 SELL
symbols that are strictly best under their same-state continuation returns.
There are52 protected relative-logit inequalities plus the objective-slope
inequality. All16 projected QPs solved successfully at rank53 with nonzero
proposals. The nonlinear gate accepted only9 of the16 proposed steps.

| Arm | Accepted / attempted | TRAIN return % | VAL return % | TRAIN / VAL orders | VAL Sharpe |
|---|---:|---:|---:|---:|---:|
| Frozen export |0 /0|13.913055|16.336139|739 /207|2.400526|
| Historical unprotected19 (290) |8 /8|14.726061|15.725127|738 /206|2.325083|
| Historical unprotected43 (290) |8 /8|14.095633|15.262069|735 /205|2.265622|
| Guard only19 |0 /8|13.913055|16.336139|739 /207|2.400526|
| Guard only43 |0 /8|13.913055|16.336139|739 /207|2.400526|
| Projected guard19 |4 /8|13.913055|16.336139|739 /207|2.400526|
| Projected guard43 |5 /8|13.913055|16.336139|739 /207|2.400526|

Every new endpoint has zero changed greedy signals in both periods. All eight
account comparisons reconcile with exactly zero delta and zero residual. Thus
the projected weights actually move and the sampled objective improves, but
there is no new trade, no improved return and no demonstrated generalization
gain. Keeping the same validation score is not evidence of learned improvement.

Unchanged validation details:207 orders,23 BUY/184 SELL,14/108 respective wins,
58.9372% total wins,3.8333 orders/week,3.257322% maximum drawdown. Six early closes
remain:4 helpful and2 harmful versus passive protection. TRAIN drawdown remains
9.548436%. The account includes the same native trading costs as290.

## Feasibility And Integrity

The guard-only arms accept nothing and preserve head tensors exactly, exposing
the immobility risk directly. The projected arms accept steps1/2/3/4 for19 and
1/2/3/5/8 for43. Accepted FIT step-margin RMS spans3.9079e-6 to4.5114e-4,
above the declared1e-6 movement floor. Final protected margin decrease is no
more than5.9605e-8; the worst accepted intermediate is8.9407e-8, both below1e-7.
No protected greedy action changes. These small steps do not cross any other
decision boundary in the full data either.

The independent GPU audit verified361 sealed files across the run and its
sources, recomputed protection membership directly from FIT outcomes, checked
all eight saved checkpoints, and confirmed exact frozen representation/critic/
alpha state. Optimizer counters equal accepted steps, including empty state
for the entirely rejected controls. It recomputed protected margins using the
same GPU dispatch and reproduced stored minimum changes exactly. The collector
separately verified213 worker and27 report files; these counts overlap and must
not be added as independent evidence samples. Cold logits/accounts match in
every arm. All source weights remain unchanged.

The run's non-governing expected-return diagnostic compared actor-only panel
logits against canonical production emissions; the zero-update controls therefore
showed rounding deltas up to1.49e-9 despite exact weights/accounts. Future runner
diagnostics now compare production emission against production emission, with a
zero-change regression test. This does not affect protection, fitting, accounts
or any decision; sealed run results remain intact, not rewritten.

## Disposition And Next Step

The preservation mechanism is feasible, not yet a profit-improving refinement.
Keep the frozen export as the research reference. Preserve both projected
checkpoints as separate experimental snapshots, not automatic live replacements.
Do not repeat the unchanged floating pass just because weights moved while
actions stayed identical. No further training is running from this step.

Next contrast should direct the update toward the source's incorrect FIT
decisions while retaining these source-correct constraints, rather than only
decreasing the aggregate critic objective. First measure requested versus actual
margin changes on those FIT errors; use current-policy continuation advantages,
not validation outcomes or an input carrying future data. Compare with the saved
292 results under the same finite budget. A successful constraint check alone
must never substitute for full account improvement. This next contrast remains
designed, not executed in292.

Evidence: [summary and account attribution](generalization/fit_preservation_20261111/SUMMARY.json),
[update steps](generalization/fit_preservation_20261111/STEPS.json),
[protected FIT set](generalization/fit_preservation_20261111/GUARD.json),
[independent GPU audit](generalization/fit_preservation_20261111/AUDIT.json),
[return plot](generalization/fit_preservation_20261111/RETURNS.png),
[manifest](generalization/fit_preservation_20261111/MANIFEST.json).

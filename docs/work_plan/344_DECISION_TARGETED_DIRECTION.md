# 344: Decision-Targeted Feasible Direction

Status: IMPLEMENTED_TESTED_BEFORE_EXECUTION.69 focused tests and scoped Ruff pass.
The new test first failed because the implementation did not yet exist.

## Question And Frozen Scope

Can an explicitly decision-targeted direction correct an erroneous FIT action
where the saved AdamW proposal only lowers soft cost? Use the ten immutable340
first-crossing witnesses and their341 protected controls. No actor, encoder,
input, target, split, optimizer history or held-label change. This is a finite
weight-space diagnostic, not ten new training runs or a promoted recipe.

For each witness, retain the same AdamW displacement d as the projection's
reference and the same maximum norm4*||d||, original anchors, expanded optimal
sets, three relinearizations and eight scales. Add complete-twin inequalities
requiring ONE currently wrong FIT row to choose a target-optimal action.
The chosen target is the highest-current-Q member of the label's optimal set;
equal optimal labels are not ordered against one another. Target margin is
64*float32_epsilon*max(1,max_abs_Q_at_row), frozen before execution.

Only wrong rows with the same greedy action across all five geometries are
eligible. Original anchors are excluded because their frozen actions cannot
change. Rank all remaining rows by the largest individual halfspace distance
max((margin-gap)+/||gradient||), then by global row id. This is only a LOWER
BOUND on movement, not a feasibility certificate. Examine at most the first8
within4*||d||. Record excluded, too-far and examined counts. No adaptive increase
of radius, candidate count or thresholds after results.

Accept only if actual float32/64 single/batch inference preserves original
guards and expanded optimal sets, corrects the target in all five geometries,
strictly lowers both full-FIT soft cost and regret in all five, and has positive
bounded motion. Report every candidate, including failures. Best diagnostic
candidate is minimum NumPy FIT regret, then cost, then row id; no held data.
Save candidate state without an optimizer: this counterfactual is NOT a valid
continuation checkpoint. Do not silently reuse AdamW moments after this probe.

## Test Design And Evidence Gates

|Requirement|Test or verification|
|---|---|
|Only FIT, wrong consensus rows, no original anchor|Unit fixtures with correct, tied, inconsistent and excluded rows|
|No ordering among tied optimal targets|Exact candidate-pair assertions|
|Rank and cap deterministic|Known affine fixture and radius rejection|
|Correct analytic gradients|Torch64 autograd on actual target pairs before projection|
|No surrogate-only acceptance|Finite test rejects unchanged action or higher regret|
|Same control, source and constraints|341 endpoint/receipt/quality exact replay; parent seals|
|Actual candidate behavior|Five-view finite checks, cold checkpoint and full repeat|
|No mutation/promotion|Source hashes and actor bundle hash before/after|

After completion publish counts, movements, actual improvements/worsenings and
matched FIT metrics. A null result bounds this neighborhood; a successful
direction is local feasibility evidence requiring a separate continuation
contract. Neither triggers a long fit or a floating-selection rerun by itself.

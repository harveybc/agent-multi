# 344: Decision-Targeted Feasible Direction

Status: COMPLETE_INDEPENDENTLY_VERIFIED.69 focused tests passed before launch;
the expanded related suite now has73 passes. Scoped Ruff passes.
The new test first failed because the implementation did not yet exist.

## Measured Result

Producer e1a41836, independent verifier03b854ea; identity
`1273e12e6e3c44ec6a6238a497a32cd25bd205f5ae455d231903c49b7397af23`.
Elapsed118.92seconds. All ten341 controls reproduced exactly.

Across the ten witnesses there were1033 eligible wrong-row occurrences, not
unique independent samples. The linearized distance screen admitted43; all43
were tested, with10 accepted candidates across5 witnesses and3 of4 cases.
All10 accepted states passed cold verification and repeated projection exactly.
Target-pair derivatives independently agree with Torch64 within7.11e-15.

|Witness|Candidates tested / accepted|Control FIT regret|Best feasible regret|Reduction|
|---|---:|---:|---:|---:|
|A19/10|3 /0|0.156435285|none|none|
|A19/32|3 /2|0.150951733|0.148733808|1.4693%|
|A19/51|2 /0|0.149094201|none|none|
|A43/238|4 /3|0.131713470|0.128855336|2.1700%|
|B19/5|6 /0|0.193833237|none|none|
|B19/33|8 /0|0.193875502|none|none|
|B43/67|2 /0|0.229738595|none|none|
|B43/88|4 /2|0.230927354|0.229738595|0.5148%|
|B43/144|5 /1|0.226259494|0.225138172|0.4956%|
|B43/152|6 /2|0.226325743|0.224879363|0.6391%|

The five best diagnostic candidates improve6 case-row decisions and worsen0.
They move1.17-3.01 times the raw AdamW proposal norm, below the common4x cap.
This is actual FIT action improvement, not just a lower surrogate loss.
It is not account profit or proof of end-to-end generalization.

Of33 rejected candidates, the final recorded attempt failed because23 did not
strictly lower regret,8 exceeded the norm cap,1 did not lower cost and1 failed
the solver check. This classifies the last attempted branch, not a proof that
every earlier trial had the same failure. The990 distance-screen exclusions
refer to the linearized model only; they do not prove nonlinear impossibility.
No radius or threshold was enlarged after seeing the results.

## Decision

A feasible decision-improving direction exists in several neighborhoods where
the original protected AdamW step changes no actions. Layer swapping was not
enough; targeting an action boundary can help. Keep the scope precise: these
are FIT-selected, bounded counterfactual corrections on correlated witnesses.
Their saved checkpoints have no optimizer and cannot silently resume training.

The next bounded [345 continuation](345_BOUNDED_TARGETED_RESCUE_CONTINUATION.md)
explicitly defines projected-AdamW moments, sixteen matched attempts per case,
four targeted searches, exact restart and automatic independent verification.
It is already implemented and dispatched, not an actor promotion.

[All ten results](generalization/targeted_direction_20261114/SUMMARY.csv),
[verification](generalization/targeted_direction_20261114/VERIFICATION.json),
[configuration](generalization/targeted_direction_20261114/CONFIG.json),
[comparison plot](generalization/targeted_direction_20261114/PAIRED_REGRET.png).

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

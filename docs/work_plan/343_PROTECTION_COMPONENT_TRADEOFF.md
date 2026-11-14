# 343: Saved-Weight Protection Tradeoff

Status: COMPLETE_REPEATED_EXACT. Post-hoc diagnostic, separate from342.
Protocol committed before execution at b959e90d; no fitting or promotion.

342 prevents harmful FIT switches but is not uniformly better than its saved
control. Investigate without a longer fit: enumerate all16 combinations of
the four native critic blocks (each twin's hidden and output layer, including
bias) between the protected64 and control64 endpoints for each of four cases.
No new optimizer, actor, encoder, inputs, labels or split. Existing hybrid and
exact-Shapley tests cover this operation.

Authenticate342 and its independent verifier plus original338/330/324 parents.
Masks0/15 must exactly reproduce their source states and saved NumPy FIT
metrics. Every hybrid must pass original guards and preserve all target-optimal
FIT sets eligible in all five protected-endpoint geometries. A useful recovery
also requires lower FIT regret AND lower FIT soft cost in all five geometries
than protected64. All-tied labels impose no ordering. No held score chooses
a mask. Enumerate everything; do not tune coefficients after seeing results.

Report64 finite interventions, exact Shapley attribution of the regret and
cost difference, feasible recovery masks and all violations. Any recovery is
local evidence for a later update design, NOT a deployable checkpoint or an
automatic trading-model promotion. No full-training run follows automatically.

## Findings

Identity `3efba74f2d83248cd889f8886cd4e66380fa463b800a2e1424c03c20fc325a25`.
All64 interventions completed and the full result repeated byte-exactly in a
fresh process/output. ZERO satisfy the recovery rule. Both endpoints reproduce
exactly in every case. A43/B43 control and protected endpoints are identical;
exchanging those blocks adds no new state.

In A19, every partial exchange (masks1..14) violates at least one original
guard. The full control passes original guards but loses2 protected optimal
decisions. Some hybrids have lower aggregate regret, so looking only at their
mean would miss the damage. In B19, the control itself is worse than protected.

Exact Shapley attribution (protected to control, negative means lower regret):

|Case|Twin0 hidden|Twin0 output|Twin1 hidden|Twin1 output|
|---|---:|---:|---:|---:|
|A19|+0.000420320|+0.000312451|-0.003407121|+0.000474996|
|B19|+0.000331145|+0.000341837|-0.000588751|+0.001297036|

Attributions include interactions and sum to the observed endpoint difference;
they do not certify that an individual block can be transplanted safely.
Both other cases have exactly zero attribution. This rules out a simple saved
block exchange in this finite contrast, not all weight-space methods.

Next bounded contrast: construct a FIT-only decision-aware update direction
for currently wrong actions, retaining original and expanded optimal-set
guards. Compare against the saved AdamW proposal at identical states before
any long fit. Require finite decisions, nonzero motion, independent derivatives
and cost/regret measurements; no held labels, new inputs or actor changes.
The question is whether a better feasible direction exists, not whether to
relax protections or keep shrinking the same harmful direction indefinitely.

[All interventions](generalization/protection_tradeoff_20261114/SUMMARY.csv),
[five-view details and attribution](generalization/protection_tradeoff_20261114/RESULT.json),
[identity](generalization/protection_tradeoff_20261114/CONFIG.json).

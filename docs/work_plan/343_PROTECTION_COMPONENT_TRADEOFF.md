# 343: Saved-Weight Protection Tradeoff

Status: DESIGNED_BEFORE_EXECUTION. Post-hoc diagnostic, separate from342.

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

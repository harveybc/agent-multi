# 340: First Harmful Crossing Replay

Status: IMPLEMENTED_TESTED_BEFORE_EXECUTION.52 focused tests pass; scoped Ruff passes.

Replay all four complete338 trajectories from334 through256 using the original
batch schedule, optimizer state, CUDA device class, primal solver and complete
twin plus single/batch guards. No new algorithm, actor update, feature change,
held-label gradient or2024/TEST access. This is authenticated replay, not a new
candidate. Record first harmful change of every expanded FIT row, not only339's
nine harmful terminal occurrences. Preserve all original artifacts.

## Acceptance And Traceability

| Requirement | Test / evidence |
|---|---|
| Same trajectory | Every transaction receipt exactly equals338; weights,optimizer,RNG at64/256 equal saved checkpoints |
| Causal stage attribution | Saved before/proposed/accepted weights and actual clipped batch gradient; finite margins and local tangents separately |
| Optimizer decomposition | Reconstruct AdamW proposal from updated moments plus weight decay; separate old/current numerator contributions at the actual updated denominator |
| No fake momentum ablation | Label decomposition as algebraic attribution, not a counterfactual optimizer without momentum |
| Shared gradient | Own multiplicity in actual batch plus other examples; compare full-FIT gradient and role contributions at crossing |
| Numerical witnesses | NumPy64 and Torch32 actions; analytic derivative versus actual CUDA batch gradient, with minimum-twin branch differences reported |
| Frozen sources | Parent seals and source actor hash before/after; witness files independently reloadable |
| Nontrivial regression | Known AdamW recurrence; duplicate-batch contribution accounting; accepted projection can reverse unprojected margin; tampered replay rejection |

Do not infer the whole training cause from a source tangent. The measured
step is a local causal intervention with fixed source weights and batch. A
protective projection can help or hurt an unprotected row without violating
the original protected contract. Quantify, do not assume that it is the cause.

Sequential unattended execution: four resumable case units with exclusive
lock, progress records and failure evidence. A failed unsealed case restarts
from its authenticated source; sealed completed cases are reused. Replay
receipt mismatch fails closed. No historical result is overwritten.

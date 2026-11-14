# 336: Stalled Proposal Solver Contrast

Status: IMPLEMENTED_VERIFIED_BEFORE_EXECUTION.

334 B43 cost-only has no accepted update after457; its saved1024 weights and
AdamW state therefore reconstruct later rejected attempts. Authenticate that
claim by exactly replaying original controls before interpreting a new solver.
No source299 actor, encoder,71-feature recipe or calibrated335 copy is changed.

## Paired Probe

Select8 equally spaced trailing rejections from each terminal reason
(`solver_or_norm_cap`, `finite_guard`),16 in total, using attempt order only.
For each restore the SAME final334 weights/optimizer, recover the original
scheduled minibatch, and compare the old dual L-BFGS-B projection with a
reduced-primal SLSQP projection. Save both proposed/end states and optimizer
receipts. Old transaction records must reproduce exactly. The head is
deterministic; no replay claim is inferred from approximate metrics.

New projection: normalize constraint rows as before; QR of the transposed
constraint matrix spans its row space, with no rank truncation or dropped
constraints. Optimize the minimum-norm correction in that span using SciPy
SLSQP, max2000 iterations, ftol1e-14. Check the ORIGINAL normalized residual
<=1e-7, solver success and KKT residuals<=1e-7. All existing finite guards,
2x interior target,3 relinearizations,8 scales and4x norm cap remain unchanged.
No coefficient/tolerance search and no relaxed margin to make a trial pass.

## Automatic Continuation

After all16 probes and exact control replay, run a paired256-attempt continuation
only if at least4 new proposals are accepted, robust across recorded execution
views, non-worsening on stable-added FIT regret versus the frozen start, and at
least one strictly improves it by>1e-8. The continuation starts from saved1024
weights/AdamW, uses scheduled attempts1025..1280 in both arms and records all
outcomes. No promotion from the probe, no further automatic extension.
Otherwise finish with the measured feasibility result and rejected gate.

Report solver feasibility separately from finite-network acceptance and from
actual decision learning. A tighter linear solution is not assumed to solve
nonlinear/float32 safety. Scores use a label-free frozen stable-added mask;
original FIT decisions are protected, held labels are not fitted or selected.

## Test Design

| Requirement | Evidence |
| --- | --- |
| Correct independent projection | Known analytic halfspaces, row scaling, duplicates, infeasibility and KKT witnesses |
| Unchanged legacy behavior | Default versus explicit legacy projector parity |
| Same constraints | No rank truncation; original-space residual and unchanged norm/finite guards |
| Authentic rejected proposals | Exact prior record replay; source/optimizer hashes unchanged after rejection |
| Nonzero meaningful recovery | No extension on feasibility-only, loss-only, geometry mismatch or worsening regret |
| Autonomous bounded work |16 paired probes; conditional two256 continuations; checkpoints and cold/resume checks |

No2024/TEST, actor updates, live promotion or unchanged floating pass. Keep
49 cumulative return candidates and paused STL. This is a local fitting
diagnostic, not a claim about financial signal absence or trading returns.

Prelaunch:45 focused CPU tests pass in3.74s; scoped Ruff checks pass. The
default projector remains unchanged, with an explicit parity test. Completed
probe units are sealed and reused on restart; conditional fitting checkpoints
both weights and optimizer/RNG, without overwriting sealed units.

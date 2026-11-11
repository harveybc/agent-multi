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

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

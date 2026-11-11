# Bounded Functional Actor Contrast

2026-11-11. Owner approves the next contrast after283. Six new arms use the same
qualified69 source actor, sealed282 absolute critics (seeds19/43),151 anchors,
113 FIT2020-2022 /38 MONITOR2023, batch128 and eight-update schedules. No critic
refit or source change. Full native TRAIN2020-2023 and VALIDATION2024 accounts
are scored at0/1/8, fixed endpoint8. TEST stays closed.

## Arms And Shared Limits

For each seed:

1. Bounded native head AdamW (four head tensors).
2. Bounded last-layer AdamW (final weight and bias only).
3. Bounded function-space projection (four head tensors, no Adam moments).

The original unbounded absolute/native endpoints from282 remain an authenticated
reference; they are not retrained. The bounded native arm is necessary to isolate
the common drift limits from the choice of update geometry.

All new arms have a maximum per-step FIT margin RMS of0.002, mean cumulative
KL(source || candidate)1e-4, maximum per-row cumulative KL1e-3 and cumulative
head parameter L2 drift/source L2 of0.01. These are prespecified diagnostic trust
budgets, not market-optimal coefficients. Every trial must also not increase
the sampled native SAC actor objective beyond1e-10. Backtrack by halves at most
24 times; otherwise retain the previous weights and explicitly record a rejected
step. Only FIT controls budgets/acceptance. MONITOR and validation never do.

## Projection Definition

At each step compute the negative independent-logit gradient of the same native
categorical SAC objective, including frozen alpha, using the sealed critic Q.
Convert it to direction-minus-NOP margin changes, scaled to RMS0.002 over the
actual sampled batch. No realized-return targets enter this update.

Form margin Jacobian J against actor-head parameters for unique sampled rows,
weighted by the square root of their actual batch multiplicity. Solve
`delta = J.T solve(J J.T + lambda I, desired)` in float64, with
`lambda = 1e-3 * trace(J J.T)/2`, where row weights sum to one. This is the
weighted mean squared sensitivity per margin and is invariant to expressing
sample multiplicity as repeated rows. Add the same decoupled decay term
`-learning_rate * weight_decay * current_head`. Apply the common nonlinear
backtracking checks. Recompute J every step. Save linear projection residual,
actual target agreement, accepted fraction, budget metrics and actual movement.
This is a tested alternative optimizer, not a claim of a Pareto/global optimum.

## Requirements And Tests

| ID | Contract | Prespecified check |
|---|---|---|
|P1|Damped projection|Known independent/shared linear fixtures, zero desired direction, finite inputs, rank-deficient stability, common scaling and duplicate weighting|
|P2|Correct functional direction|Analytical categorical gradient already verified in283; finite-difference Jacobian and actual objective descent|
|P3|Uniform FIT-only trust rules|Budget boundary tests, monitor exclusion, reject-and-restore including optimizer moments, no silent zero accepted as learning|
|P4|Same starting point|Parent/store/bundle/282 receipts and exact baseline accounts; same Q and schedules by seed|
|P5|Scope and reproducibility|Frozen components and buffers exact; canonical cold logits/accounts exact at1/8; last-layer-only leaves first head layer exact|
|P6|Trading consequences|All six endpoints, activity/wins/buy/sell/DD/early exits and chronological account attribution, not only diagnostic loss|

## Disposition

Sequential primary-GPU fit, then independent all-endpoint attribution and
authenticated collection, bounded by finite supervisors. Preserve every source
and checkpoint. No live promotion, new feature, rolling transform or agent-state
input. An improved snapshot needs export qualification before a changed-policy
floating pass. Do not repeat the unchanged subset screen by default.

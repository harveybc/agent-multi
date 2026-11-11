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

## Verified Launch

40 focused tests and scoped lint pass. Runtimeaacfb61a was exported immutably.
GPU readiness passed on the primary RTX5090; reference native accounts match
282 exactly and the first bounded/native arm has started. The sequential queue
includes independent endpoint attribution and automatic hashed collection.
Identity:
`ed6134dc0327bf1ad630166d23765fc430304a1b3078edd173c1e9e08abe1846`.

## Completed Results

All six arms completed in74.20seconds, independent reporting in23.53seconds.
The collector authenticated305 worker files and39 report files. All48 steps
were accepted, had real parameter movement, respected every FIT drift budget
and decreased the sampled actor objective. No critic updates were performed.
Six cold-reload and frozen-component checks pass; all12 TRAIN/VALIDATION
endpoint account differences reconcile with zero USD attribution error.
Both queue and collector exited0 and are inactive.

| Arm | Seed | TRAIN return % | VAL return % | TRAIN orders | VAL orders | VAL Sharpe |
|---|---:|---:|---:|---:|---:|---:|
| Qualified starting actor |19|11.565540|14.591090|747|206|2.165596|
| Prior unbounded absolute calibration282 |19/43|12.392145|14.693262|743|206|2.179247|
| Bounded head AdamW |19|11.342817|14.591090|749|206|2.165596|
| Last layer only |19|11.342817|14.591090|749|206|2.165596|
| Functional projection |19|11.287557|14.693262|744|206|2.179247|
| Bounded head AdamW |43|11.264796|14.591090|746|206|2.165596|
| Last layer only |43|11.342817|14.591090|749|206|2.165596|
| Functional projection |43|12.109487|14.169750|744|205|2.111990|

The prior282 candidate is not superseded: it has higher TRAIN return and equal
or higher validation return than every new endpoint. This comparison does not
imply dominance on every metric (for example DD), nor rejection of every
possible projection/last-layer configuration. The common ceilings do not make
realized displacement equal across arms.

Validation details: all but projection43 execute24 BUY /182 SELL, with15 /105
winners,58.2524% total win rate,3.8148 orders/week and approximately4.0446%DD.
Projection43 executes24 BUY /181 SELL,15 /104 winners,58.0488% wins and3.7963
orders/week with4.044617%DD. Projection19/43 have6 early closes (4 helpful,
2 harmful); the bounded-AdamW/last-layer controls retain7 (4 helpful,3 harmful).
All six pass the existing activity requirement. Full TRAIN/VAL counts, costs,
drawdowns, directions and early-close diagnostics are in the summary.

## Mechanism And Accounting

The projection actually followed its proposed local FIT direction: cosine
agreement of actual versus requested margin changes is0.9322-0.9452 for seed19
and0.9554-0.9623 for seed43 across eight steps. This is approximate translation,
not exact independence of every observation and not a profit guarantee. The
damped linear residual and nonlinear target RMSE are retained for every step.
No realized-return oracle labels were used to update an actor.

| Arm / seed | Accepted fraction each step | Final mean FIT KL | Final max FIT KL | Final relative head drift |
|---|---:|---:|---:|---:|
| Bounded AdamW19 |0.25|4.8302e-6|1.1891e-4|4.8524e-4|
| Bounded AdamW43 |0.50|1.7835e-5|4.4513e-4|9.9448e-4|
| Last layer19 |1.00|7.8993e-7|1.2301e-5|2.5463e-4|
| Last layer43 |1.00|8.5661e-7|1.4407e-5|2.6102e-4|
| Projection19 |1.00|2.6447e-5|6.7491e-4|2.9007e-3|
| Projection43 |1.00|2.7869e-5|8.7459e-4|2.9430e-3|

Chronological account attribution versus the qualified starting actor:

- Bounded19 and both last-layer arms lose22.272290USD in TRAIN, entirely from
  one flat-state SELL-to-BUY switch. Their validation accounts are unchanged.
- Bounded43 loses30.074310USD in TRAIN through five effective opening changes
  (two helpful,three harmful). Validation is unchanged.
- Projection19 loses27.798230USD in TRAIN through five changes (two helpful,
  three harmful). One is a new short-position early close costing19.936245USD;
  the other changes are openings skipped or retained downstream. Its validation
  gain10.217220USD is the already-known avoided harmful early close before
  forced-flat, not a new broad timing improvement.
- Projection43 gains54.394705USD in TRAIN through eight effective opening
  changes (four helpful,four harmful). In validation it gets that same10.217220USD
  early-close benefit but skips a profitable short opportunity, costing
  52.351235USD; net validation effect is-42.134015USD.

Both projections modestly improve greedy native-return regret on the38-anchor
TRAIN monitor (0.245491 ->0.242392), but this does not imply better full-period
trading. The sampled SAC objective, diagnostic regret and real account outcomes
are distinct checks; none may substitute for another.

## Disposition And Exact Next Step

No new optimizer is adopted as the default and no model is promoted live.
Preserve all284 snapshots and the earlier282 absolute-calibrated19/43 candidates.
This run shows that translating the critic's local direction more accurately
does not, by itself, establish a trading improvement. It neither disproves the
input signal nor settles all causes of harmful refinement.

Return to the productive phase2 sequence: qualify/export the saved282
absolute19 actor (the prespecified seed, with43 as paired replication), verify
full native account and cold tensor parity, then a fresh-cache frozen deletion
and cumulative historical-phase2 readmission pass on that changed policy.
Do not refit the critic or retrain that actor merely to recreate saved weights.
Do not perform another identical pass on the unchanged prior69 policy. New
optimizer experiments can remain a separate lane after this useful candidate
is incorporated into the selection workflow. That export/pass has not yet
been launched in this step; the six-arm contrast and its audit are complete.

Evidence: [manifest](generalization/bounded_functional_20261111/MANIFEST.json),
[full results and attribution](generalization/bounded_functional_20261111/SUMMARY.json),
[all update budgets and prior reference](generalization/bounded_functional_20261111/UPDATE_STEPS.json),
[curves](generalization/bounded_functional_20261111/RETURNS.png).
Report identity:
`0173d69d8fd1a000e0f723a08e68b67651b5d102fdab9cae8802f7dc016d612a`.

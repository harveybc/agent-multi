# TRAIN Coverage Contrast

2026-11-11. Continue after287 using qualified `critic_adamw_19`,69 features,
28 groups, window13/stride2 and unchanged business rules. Preserve all sources.

## Frozen Experiment

Four arms: `critic_narrow` versus `critic_broad`, update-schedule seeds19/43.
All start from the same287 actor and frozen critic. Native AdamW, batch128,
eight head-only updates, source alpha and284 drift limits remain unchanged.
Save0/1/8; endpoint8 is fixed. No critic training/reset, architecture change,
validation target fitting, TEST access, feature selection or live promotion.

Narrow uses the113 inherited FIT anchor timestamps. Broad retains them and adds
up to three times their number within each stratum: UTC year, current position,
actual entry availability, and flat-state greedy action. Build strata using the
current frozen policy's2020-2022 trace, before outcome targets are calculated.
Select additional timestamps uniformly without replacement using fixed seed288;
sort strata and candidates for deterministic ordering. Never use returns to
choose rows. Record shortages explicitly instead of fabricating/repeating rows.
Expected broad size452 if all strata have enough candidates. Preserve the38
MONITOR timestamps in2023 exactly. These anchors are not independent market draws.
Only strata already represented in the narrow panel are expanded; this is not
a claim of exhaustive coverage of every possible market or position state.

Rebuild all native continuations under this actor, restricted to the anchor's
FIT/MONITOR interval. Those returns are diagnostics only here: neither arm
replaces the critic target with them. The hypothesis is coverage under the
unchanged actor objective, not a simultaneous objective/critic change.

Both arms use the SAME broad FIT support for trust checks, avoiding a confound
where only broad updates face additional drift checks. Gradients use their own
training support, with the same1024 total sampled examples and8 optimizer steps.
Covering schedules guarantee every eligible point is visited. The larger set
therefore receives fewer repeats per point; report that budget distinction.
Compare full native TRAIN2020-2023 and VALIDATION2024 accounts, unchanged support.

## Requirements And Tests

| Requirement | Evidence |
|---|---|
| Nested/outcome-blind coverage | Unit fixtures: narrow subset, exact fourfold count when available, identical selection under target perturbation and trace reorder |
| Temporal isolation | Reject non-FIT trace rows/duplicate timestamps; monitor inherited unchanged |
| Scarcity | Explicit requested/available/selected counts, no replacement or invented rows |
| Paired training | Same source/Q/alpha/budgets, common broad trust support, seed schedules authenticated; only FIT indices enter gradients |
| Current-policy labels | Fresh native continuations, parent unchanged, HOLD equivalence, exact source account replay |
| Scope/export | Frozen representation/critics/buffers; canonical cold logits and accounts at1/8 |
| Economic audit | All eight endpoint account differences reconciled, full orders/wins/direction/DD/early-close reporting |

Sequential primary-GPU worker, independent endpoint attribution and finite
authenticated collection. No default long refinement or automatic promotion.
An improved candidate requires saved-weight qualification before another
changed-policy floating pass. If unchanged, do not repeat the same screen.

Preflight:39 focused unit tests passed; scoped Ruff and whitespace checks passed.
The first execution was stopped before actor updates: selection expanded the
anchors but the continuation call still received inherited records. The call
now receives expanded anchors, with a routing regression test. Preserve the
initial artifacts as interrupted, not results; execute the correction in a
fresh `r2` runtime/output identity.

## Completed Results

Corrected runtime6752d251 completed four arms in158.06seconds; independent
account attribution took6.71seconds. Worker and collector exited0. Collection
authenticated215 worker files and27 report files. All32 updates accepted real
movement, strictly reduced their own batch loss and passed the frozen budgets.
All eight endpoint account differences reconcile with exactly zero error.

The outcome-blind expansion produced437 FIT anchors, not452. The2021 flat,
entry-allowed, BUY-greedy stratum had only3 additional candidates for18 requested:
shortfall15, no replacement. The original113 FIT anchors and38 MONITOR timestamps
were retained. Every training row was visited within each arm's1024 samples.
Critic values, alpha, architecture, budgets and437-row trust support match across
all arms. Effective targets equal predicted Q exactly; native returns are only
diagnostics. Cold inference/accounts and frozen components match exactly.

| Arm | Schedule seed | TRAIN return % | VAL return % | TRAIN orders | VAL orders |
|---|---:|---:|---:|---:|---:|
| Qualified287 reference |19|12.708707|14.693262|743|206|
| Narrow113 |19|13.201812|14.693262|741|206|
| Broad437 |19|13.201812|14.693262|741|206|
| Narrow113 |43|13.201812|14.693262|741|206|
| Broad437 |43|13.201812|14.693262|741|206|

Broader coverage adds no economic benefit in this bounded, equal-update probe.
The common additional refinement gains49.310465USD TRAIN versus287, without a
new validation gain. This is not evidence that coverage cannot help at another
budget or with a better critic. The two seeds change update sampling, not the
pretrained representation, critic or market sample.

### Decision Attribution

Every arm has the same three nonzero chronological TRAIN contributions:

| Change | Account contribution USD |
|---|---:|
| Skip a profitable SELL | -36.462985 |
| Skip a losing SELL | +56.446610 |
| Skip a losing BUY | +29.326840 |
| Total | +49.310465 |

These are full account-continuation contributions in chronological order, not
independently additive trade profits. Downstream positions and balance explain
why three changed opening decisions produce a net reduction of two trades.
Broad19 also changes one validation raw signal with no economic consequence.
The other arms change no validation signals. Do not mistake changed logits or
symbols for executed trades, or a net gain for improvement of every action.

All four endpoint accounts have these metrics:

| Metric | TRAIN | VALIDATION |
|---|---:|---:|
| Net return |13.201812%|14.693262%|
| Net USD, initial10000 |1320.18|1469.33|
| Sharpe |0.651837|2.179247|
| Maximum DD |7.930285%|4.044574%|
| Orders/week |3.5455|3.8148|
| Win rate |61.8084%|58.2524%|
| BUY orders/winners |229/154|24/15|
| SELL orders/winners |512/304|182/105|
| Early/TP/SL/forced exits |57/402/221/61|6/107/67/26|
| Helpful/harmful early exits |28/29|4/2|

Early-close quality has not improved: TRAIN helpful closes decrease29 to28,
harmful remain29. The change in the set of entered positions must not be
presented as an independently improved closer. Validation remains unchanged.
Backtracking fractions were0.25 throughout except some broad19 steps at0.5;
equal budget ceilings are not equal realized displacement.

## Disposition And Next Step

Keep all four saved endpoints plus287 unchanged. Choose narrow19 as the next
research qualification candidate, with narrow43 as replication: broader input
coverage did not improve these accounts. This choice does not promote a live
model or prove global optimality. First export/qualify the exact saved weights
with recipe, frozen tensors, cold logits and full account parity. Only then
perform a changed-policy cumulative floating pass, including all previously
removed phase2 inputs. Do not launch another long refit to recreate the weights.

Evidence: [manifest](generalization/train_coverage_20261111/MANIFEST.json),
[full metrics and attribution](generalization/train_coverage_20261111/SUMMARY.json),
[coverage](generalization/train_coverage_20261111/COVERAGE.json),
[integrity](generalization/train_coverage_20261111/INTEGRITY.json),
[update steps](generalization/train_coverage_20261111/UPDATE_STEPS.json),
[curves](generalization/train_coverage_20261111/RETURNS.png).

Worker identity:`10670f6c027d39f0e776235aae72246333cdc8d6bd8230530f161b93adb0d405`.
Report identity:`999fa516a0ae6d8ac6d2881b56060baa150def826ca31163db169048dfddc4b0`.

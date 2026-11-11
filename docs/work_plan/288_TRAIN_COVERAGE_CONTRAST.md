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

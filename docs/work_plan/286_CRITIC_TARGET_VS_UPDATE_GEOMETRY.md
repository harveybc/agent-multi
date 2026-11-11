# Critic Target Versus Update Geometry

2026-11-11. Continue after285 with the qualified absolute19 actor,69 inputs,
28 groups, unchanged features, frozen TRAIN scaling and full native accounts.
The285 floating cycle is complete; no identical screen is repeated here.

## Question And Frozen Design

284 translated the critic's requested FIT changes more accurately without a
joint trading improvement. Separate target estimation from update geometry:

| Target | Head AdamW | Damped functional projection |
|---|---|---|
| Frozen current critic |critic_adamw|critic_project|
| Native frozen-policy continuation, FIT only |return_adamw|return_project|

Two update-schedule seeds19/43 give eight arms, all starting from the same
qualified absolute19 bundle and its same frozen critic. These are optimizer
schedule replications, not independent critic fits or independent market samples.
No critic training, reset, new model inputs or broker changes. The source is285,
not the earlier279 actor. Record at0/1/8; endpoint8 is fixed in advance.

Reuse the151 outcome-independent anchor timestamps declared in282, but replay
the NEW source policy and rebuild each position/state and all three native
continuations. Never reuse old-policy returns or positions. The counterfactual
first action is NOP/BUY/SELL followed by the frozen current policy, to the end
of the same role interval: FIT2020-2022 or MONITOR2023. This is an offline RL
return diagnostic, not ZigZag supervision, an ideal exit, or a live future input.
It is not a full-TRAIN actor fit; all full TRAIN/VALIDATION data are evaluated.

Only FIT realized reward returns replace the critic values in return arms.
MONITOR returns are diagnostic only; validation does not generate targets or
accept updates. Keep native alpha and all other actor-loss terms unchanged.
Native greedy returns and a soft/stochastic critic can differ by objective as
well as estimation error, so a target contrast alone does not prove a code bug.

Both geometries use identical seed schedules, batch128, eight updates and the
prespecified284 trust limits/backtracking. Four head tensors train; encoders,
critics, temperature and buffers stay exact. Do not select intermediate steps
or retune budgets after reading results. Cache finite substeps with receipts;
run sequentially on the primary GPU then independent account attribution and
authenticated collection. No automatic promotion or further training.

## Acceptance And Test Design

| Requirement | Test / evidence |
|---|---|
| Correct source |285 qualification receipt, saved endpoint and model hash, full source account replay|
| Fresh native targets |Rebuild state at fixed timestamps; all453 continuations preserve parent, stay in role interval, HOLD equivalence holds|
| FIT-only replacement |Unit tests perturb MONITOR outcomes without changing training values; reject unsupported target kinds/nonfinite FIT values|
| Same experiment except target/geometry |Same source,Q,schedules,alpha,budgets; explicit per-arm target and critic hashes|
| Translation |Existing Jacobian/projection/duplicate-weight/trust tests plus measured residuals, movement and acceptance counts|
| Frozen components and export |Exact fixed state/buffers, canonical cold logits and full accounts at1/8|
| Economic consequence |All16 endpoint accounts attributed chronologically, including direction/activity/wins/DD/early-close harm|

If even native FIT targets fail, examine transfer from FIT anchors to held-out
states before increasing capacity. If native targets help but predicted targets
do not, prioritize critic objective/ranking on the current policy. These are
conditional next actions, not claims about the result or absence of input signal.

## Verified Launch

38 focused tests and scoped lint pass. Runtime `40dd908f` is immutable.
The primary RTX5090 passed physical readiness and started the sequential queue.
Reference accounts reproduce285 exactly. All151 anchor states and453 native
continuations were rebuilt under the current actor by32.63seconds; the first
paired actor arm then began. Original target records and checkpoints were not
modified. The post-run attribution and finite authenticated collector are queued.

Run identity:
`d700a518254a5962841c85de578bcdf0259516dc9aaaff6a3a504cf026692c43`.

## Completed Results

Eight arms completed in132.10seconds; the independent report took21.94seconds.
Both services exited0 and are inactive. Collection authenticated411 worker files
and51 report files. All64 actor steps accepted real movement, strictly decreased
their declared FIT loss and passed drift limits. Frozen components and cold
logits/accounts reproduce. All16 endpoint account attributions reconcile exactly.

| Target / geometry | Seed | TRAIN % | VAL % | TRAIN orders | VAL orders |
|---|---:|---:|---:|---:|---:|
| Qualified285 reference |19|12.392145|14.693262|743|206|
| Critic / AdamW |19|12.708707|14.693262|743|206|
| Native FIT / AdamW |19|12.708707|14.693262|743|206|
| Critic / projection |19|13.129000|14.096345|741|205|
| Native FIT / projection |19|12.576383|14.693262|742|206|
| Critic / AdamW |43|12.708707|14.693262|743|206|
| Native FIT / AdamW |43|13.275160|14.693262|742|206|
| Critic / projection |43|13.129000|13.526164|741|206|
| Native FIT / projection |43|12.576383|14.693262|742|206|

### What Changed

The bounded critic/AdamW arm gains31.656215USD in TRAIN in both schedule seeds,
from skipping one harmful BUY and its downstream account consequences.
Validation actions and accounts are unchanged. This is a replicated local
TRAIN improvement from one effective decision, not a new validation gain or a
general solution to refinement. Native-FIT/AdamW19 makes the same account change;
seed43 additionally avoids one losing short and gains88.301545USD overall.
Preserve that endpoint, but its larger gain is not replicated by seed19.

Critic/projection improves TRAIN but skips profitable validation shorts:
net losses59.691690USD and116.709825USD. Native-FIT/projection avoids those
validation losses in both seeds, but creates a harmful TRAIN early close
costing20.197485USD within a net TRAIN gain18.423825USD. Thus better FIT targets
change how projection transfers, but do not remove all harmful shared-head
effects. Neither loss descent nor an exact FIT return implies improvement of
each held-out action. This is not a reason to blame the input information.

All counterfactual target values were recomputed:86 of151 anchors have at least
one changed native reward return versus282, despite unchanged position direction
at all151 timestamps. Policy-continuation targets must be refreshed when the
actor changes; the old records remain valid evidence for their own actor only.

### Paired Target Audit

The post-run audit confirms identical predicted Q across all eight arms and
identical batch schedules within each seed. Native returns replace ONLY FIT
rows, exactly at float32 precision. MONITOR effective values remain predicted Q
and never enter the optimizer because all batch indices are FIT. Fresh role
intervals, full account support and all source hashes remain valid.

Current critic RMSE versus the refreshed native returns is0.550888 on113 FIT
anchors and0.624431 on38 MONITOR anchors. Its greedy action is native-optimal
(including ties at1e-8) in56/113 and15/38 respectively. These are diagnostics of
this panel, not an overall market classification accuracy. Critic/objective
differences and shared update effects both remain relevant.

AdamW backtracking accepted0.25 of each critic-driven proposal and0.5 of each
native-return proposal; both projection variants accepted1.0. Common drift
ceilings do not equalize actual displacement. Do not interpret the factorial
as an isolated direction-only effect at identical step magnitude.

### Replicated Research Candidate

Retain `critic_adamw_19/ACTOR_0008.pt` as the first candidate for qualification,
with43 as its paired replication. This retains the existing actor objective and
has the same economic result in both seeds; it is not a claim of dominance over
all other endpoints or all metrics. All other snapshots remain available.

| Metric | TRAIN | VALIDATION |
|---|---:|---:|
| Net return |12.708707%|14.693262%|
| Net USD, initial10000 |1270.87|1469.33|
| Sharpe |0.629028|2.179247|
| Maximum DD |7.930291%|4.044574%|
| Orders / week |3.5550|3.8148|
| Win rate |61.6420%|58.2524%|
| BUY orders / winners |230 /154|24 /15|
| SELL orders / winners |513 /304|182 /105|
| Early / TP / SL / forced closes |58 /402 /222 /61|6 /107 /67 /26|
| Helpful / harmful early closes |29 /29|4 /2|

## Exact Next Action

Qualify/export the saved bounded critic/AdamW endpoint in both seeds with285
as unchanged reference, then use primary19 for a fresh changed-policy cumulative
phase2 pass. No refit is needed to recreate saved weights, and no full training
schedule or optimizer-default switch is justified here. Refresh same-policy
return targets before any later critic fitting. Do not transfer diagnostic
native returns into model observations or deploy unqualified snapshots live.

Evidence: [manifest](generalization/native_target_geometry_20261111/MANIFEST.json),
[results and account attribution](generalization/native_target_geometry_20261111/SUMMARY.json),
[target integrity](generalization/native_target_geometry_20261111/TARGET_INTEGRITY.json),
[all steps](generalization/native_target_geometry_20261111/UPDATE_STEPS.json),
[curves](generalization/native_target_geometry_20261111/RETURNS.png).
Report identity:
`b013cf90906f728b8815e795d3883a3b5fc1e44083a4efaa33d26ada7b8f7566`.

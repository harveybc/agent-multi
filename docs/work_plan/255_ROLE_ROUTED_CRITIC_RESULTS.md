# Role-Routed Critic Results

## Completion And Identity

Design254 completed2026-11-10T11:04:07.447515Z (06:04 Bogota),106.44 seconds.
Runtime6a6ca141;561 files authenticated in the independently collected output.
Identity `542ed938fedd5699335a603ace861d6da574875afa1f73dbe7ab09d18e8676e5`.
Evidence root `/var/tmp/sac70-role-heads-v2-20261110`.

Six arms completed4096 updates each, batch128, seeds19/43, AdamW LR3e-5,
WD0.01, existing pairwise-plus-mean loss. Critic encoders and actor unchanged.
No account-state feature was added, no strategy or checkpoint promoted.
Network computes every head from market embeddings; native direction selects
the role output externally. This is an explicit conditional-output diagnostic,
not a new unconditional market-only policy.

[Complete metrics](generalization/role_heads_20261110/METRICS.csv),
[curves](generalization/role_heads_20261110/CURVES.png),
[summary and trajectories](generalization/role_heads_20261110/RESULT.json).
SOURCE_DONE authenticates the full source collection, including retained head
weights, initialization tensors, schedules and native counterfactual receipts.

## Paired Endpoint Comparison

Reuse844 native banks,2435 state/action-target vectors, unchanged from252:
1840 fit vectors across642 banks,595 monitor vectors across202 banks. Fit is
2020-2022; monitor is2023. Encoder pretraining previously used2023; this is not
a new untouched holdout. No2024 or TEST evaluation.

Shared single has3398 trainable parameters. Shared mean-of-three and externally
routed three-head have10194 each, identical independently initialized tensors
and sample schedules within seed. Sampling weights each bank equally, then
each surviving state equally. All endpoints shown; no checkpoint chosen from
intermediate curves. Cached encoders reproduce native Q within2.3842e-7.

Regret below means lost discounted reward relative to the best of the three
first actions followed by the SAME frozen continuation. It is not percentage
profit, Sharpe, USD loss, or a guarantee of optimal future trading.

| Seed | Head arrangement | Fit bank regret | Monitor bank regret | Monitor pair accuracy | Monitor gap RMSE |
|---|---|---:|---:|---:|---:|
|19|Shared single|0.189174|0.205928|49.78%|0.376426|
|19|Shared mean-of-three|0.185792|0.195351|55.10%|0.374801|
|19|Externally routed|0.185046|0.187110|53.23%|0.375512|
|43|Shared single|0.197531|0.216504|48.49%|0.376306|
|43|Shared mean-of-three|0.182502|0.190921|53.88%|0.374238|
|43|Externally routed|0.183320|0.184829|53.66%|0.373791|

Means across the two seeds:

| Arrangement | Fit bank regret | Monitor bank regret |
|---|---:|---:|
|Shared single|0.193352|0.211216|
|Shared mean-of-three|0.184147|0.193136|
|Externally routed|0.184183|0.185970|

Routing improves monitor regret by3.71% relative to the equal-parameter control
(4.22% and3.19% per seed). Fit regret is essentially tied on average. Ranking
accuracy is lower in both seeds, and gap RMSE is mixed: this is not dominance
across all metrics. With two seeds and correlated banks, no significance or
population-level improvement is claimed.

### Where The Improvement Occurs

Monitor mean regret per native state (these state-weighted values are distinct
from the bank-balanced primary metric):

| Seed | Arrangement | Flat opening | Manage long | Manage short |
|---|---|---:|---:|---:|
|19|Shared mean-of-three|0.280357|0.140963|0.163177|
|19|Routed|0.295544|0.131341|0.130080|
|43|Shared mean-of-three|0.276600|0.140297|0.155626|
|43|Routed|0.293187|0.135928|0.120536|

Routing helps management of existing positions in both seeds but hurts flat
opening. Do not replace the whole model or declare the actor repaired. The
actor never trained in this diagnostic, and no trading account used the new
heads. Fresh-head results must not overwrite older preserved-head experiments.

## Same-Direction Residual Conflicts

At each of512 archived flat origins, attempt native long/short entries at
offsets0/1/3 decision steps and compare surviving positions at steps4/12.
All histories share the exact market observation; actual entry, barriers,
costs, size and age arise from the unmodified simulator. No hand-set positions,
forced survivorship, risk-coefficient changes or substitute origins.

2046 attempted direction/age banks,2 terminal exclusions,866 with at least two
distinct surviving current account states.7074 native suffix simulations,
including48 identical-state repetitions.16 negative controls, maximum error0.
Reference account trace hashes and final cash match the source exactly.

| Temporal role | Direction | Comparable | Incompatible best actions | Fraction |
|---|---|---:|---:|---:|
|Fit|Long|330|52|15.76%|
|Fit|Short|329|58|17.63%|
|Monitor|Long|104|11|10.58%|
|Monitor|Short|103|18|17.48%|
|Total|Both|866|139|16.05%|

Therefore direction routing alone cannot satisfy every constructed history.
This is a finite-panel counterexample to sufficiency, not a claim that16.05%
of live decisions necessarily fail. Later observations are survivor-conditioned;
many histories share market paths. The expanded audit was not used for fitting
or selecting heads.

### Concrete Witness

At origin2023Q3:71, four decision steps later, three native short positions
entered at different offsets share exactly the same market observation. Their
discounted future rewards are:

| Entry delay | HOLD (NOP/SELL) | CLOSE (BUY) | Better action |
|---|---:|---:|---|
|0| -0.883367| -0.345331|Close|
|1| -0.054444| -0.345474|Hold|
|3| -0.056639| -0.345467|Hold|

This is the maximum-common-regret witness, shown as an illustrative extreme,
not an average. It includes the subsequent frozen-policy account trajectory,
not just the original position's isolated P&L. A single market-only short-role
vector cannot produce all three desired choices here.

## Verification And Execution Correction

-15 focused CPU tests passed locally and15 on the worker, including identical
 market/opposed-action positive control, gradient isolation, cached embedding
 parity, native delayed histories and invalid routing refusal.
-Same initial tensors and schedules verified for routed versus mean-of-three.
-Every trained head checkpoint reloaded with exact output parity.
-Full-TRAIN actor logits and every source model tensor unchanged; source files
 retain their hashes. No encoder, target-critic or entropy update.
-Worker and automatic collector completed with exit0; no remaining training.
-First runtime88388fab stopped before fitting on tuple-versus-list equality in
 a JSON-restored entries_day field. Runtime6a6ca141 compares canonical full
 content. Regression still rejects changed cash/counts. Original failure kept;
 nothing from that root reused. Failed start is excluded from106.44s runtime.
-Configured Ruff executable was absent in this worktree; no lint pass claimed.

## Next Design

Keep source actor and current feature selection intact. Role-conditioned
outputs are useful diagnostically but not sufficient for position management.
Next design should evaluate market-only continuation/return or excursion curves
against the actual position's barriers and execution constraints in the native
controller, rather than demanding that a state-blind short/long scalar encode
all possible entries. Require the concrete opposite-decision witness above as
a positive control, plus early-close harms versus hold, before actor transfer.
The sufficiency and learnability of such curves remain hypotheses to test, not
an already implemented solution. Preserve the simple existing actor baseline.

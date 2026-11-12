# Selected71 Critic Calibration

2026-11-12. Owner authorizes critic calibration after299. Preserve the qualified
71-input/28-group actor, all controls, fixed TRAIN scaler and original support.
No actor training, feature pass, live promotion or observation substitution.

| Requirement | Gate / evidence |
|---|---|
| Exact new source | Qualified299 export, model SHA69b5254489d224d4b20ac358956e4dd3b9a67a2f214a9aff98045c5def25a536 |
| Fresh targets | Retain the151 authenticated TRAIN anchor origins; rebuild native continuations with current71 actor, not previous72 returns |
| Critic-only scope | Two seeds19/43;4096 critic-head updates, batch128; frozen encoders, actor and alpha; target synchronization recorded |
| Temporal scope | FIT2020-2022,2023 MONITOR diagnostics only;2024 native reference replay, never critic fit |
| Preserve learning | Actor tensors and full-support logits unchanged, including cold checkpoint reload |
| Report honestly | Both value RMSE and action ranking/regret on FIT/MONITOR; lower FIT loss alone is insufficient |
| Repeatable automation | Immutable runtime; finite calibration -> authenticated report -> verified collection; fresh run identity |

Frozen trading reference: TRAIN16.05822355000215%,VALIDATION18.369945750000117%,
745/207 orders. Calibration does not increase these returns because actor
decisions are fixed. The existing source anchors are a diagnostic panel, not
an exhaustive census of the new policy's decisions. Rebuilding their outcomes
does not expand panel coverage. Keep that limitation explicit.

Tests: calibration-only mode disables all actor arms; invalid mode combinations
fail; native targets enforce origin/support identity and TRAIN-only role splits.
After completion verify critic checkpoints, source hashes, curves and report.
Next inspect executable71-input boundaries before any bounded actor correction.
No raw AdamW retry, actor reset or unchanged floating pass. A later changed-context
return must include all49 currently absent phase2 features. Live COT/feed and
STL qualification remain independent workstreams.

## Completed

Runtime57497834 completed both seeds in196.14s (201.12s including process
startup); authenticated report1.47s. The finite queue and collector exited0.
20 sealed run files plus3 report artifacts were verified.6 focused tests pass,
scoped Ruff passes, and an additional negative integration check rejects the
historical72-input run before writing a report.

Rebuilt151 anchors:113 FIT and38 MONITOR. Their origins match the prior panel,
but444 of453 action return targets differ from the72-input calibration.
Independent checks confirm each continuation terminates inside its own temporal
partition: before/equal2023-01-01 for FIT,2024-01-01 for MONITOR. No old target
array was used as a shortcut for the new policy.

| Seed | FIT RMSE before/after | MONITOR RMSE before/after | FIT pair accuracy before/after | MONITOR pair accuracy before/after |
|---|---|---|---|---|
|19|0.764380 /0.340573|0.668575 /0.733933|52.88% /75.25%|58% /62%|
|43|0.839142 /0.530197|0.681514 /0.708622|52.54% /69.15%|40% /49%|

"Before" means each freshly initialized critic head on the frozen71-input
representation, not the previously trained72-input critic.

MONITOR mean regret (native reward units, not dollars) decreases
0.215009 ->0.151420 and0.341183 ->0.267853. Action-gap RMSE also decreases
0.449601 ->0.407188 and0.598845 ->0.463646, while common-level RMSE increases
0.616126 ->0.695263 and0.587301 ->0.656117. Improved relative action values
therefore coexist with worse absolute-value calibration. This is not uniformly
better generalization or proof of trading improvement.

The curve exposes a stopping-budget question: MONITOR pair accuracy peaks at
2048 updates (65% /50%), then ends at62% /49%; its lowest value RMSE occurs
earlier (256 for19,1024 for43). The declared4096 endpoint remains the reported
checkpoint; no retrospective replacement or claim that more updates is better.
Preserve this evidence for a separately declared stopping-budget contrast after
the executable-boundary diagnosis. The panel has only38 monitor anchors; paired
comparisons are not independent trading replications.

## Preservation And Artifacts

Both saved checkpoints have exactly8 changed state tensors: four online critic
head tensors (containing both twins) and their four synchronized targets.
All other1019 state tensors are byte-equal to the71-input source. The optimizer
records exactly4096 steps. Codec, effective config, recipe and trainability
flags match the source. Full-support actor logits remain bitwise identical after
calibration and after cold reload. No actor update was performed.

The unchanged native reference remains TRAIN16.058224%,VALIDATION18.369946%,
745/207 orders, Sharpe0.781969 /2.690712. This is preservation, not an additional
return improvement. The original299 actor/bundle remains intact.

Run identity:`33052ae3718b91244982191afe5826ca088811b312291ce3f09bca8db05c9394`.
Checkpoint19 SHA:`9aa43f658d2e669d671c711232405bfe828c913db46752c799c5a205e1662b2b`.
Checkpoint43 SHA:`2a4968b42abddc4e001025df594a46998d7758545381c38aed72876e23b29727`.
Private root:`/var/tmp/sac71-critic-calibration-20261112-r1`, with
`critic_19/MODEL.pt` and `critic_43/MODEL.pt` preserved locally and on the worker.

Evidence:[summary](generalization/selected71_calibration_20261112/SUMMARY.json),
[curves](generalization/selected71_calibration_20261112/CURVES.png),
[curve values](generalization/selected71_calibration_20261112/CURVES.json),
[manifest](generalization/selected71_calibration_20261112/MANIFEST.json).

Next: inspect executable71-input TRAIN action boundaries, critic rankings and
shared-head gradient effects before proposing a bounded interior correction.
Keep both calibrated seeds and all prior actors. Do not treat this calibration
as permission for an unconstrained actor update, automatic live promotion or
another floating pass on identical weights.

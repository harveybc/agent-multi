# Native Critic Head Capacity Contrast

Owner authorizes launch after319. Four new fits:318 folds A/B x seeds19/43,
base weight decay0.01 only. Reuse the four sealed318 controls without training
them again. Source actor299,71 inputs, annual labels, splits, MSE, LR, schedules,
batch128,4096 maximum updates,64-update checks and patience20/delta1e-8 unchanged.

## Single Factor

Native architecture: one208->8->3 head per critic twin,3398 total parameters.
Control: three such heads per twin averaged,10194 parameters. Construct the
same initialized three-head ensemble as318, verify its digest against the
sealed control, then retain head0 independently in each twin. Do not simply
construct count1 with the same seed: that changes twin1's RNG position.
Common tensors match exactly; initial functions differ because an average of
three heads is not head0. Record that difference. Optimization geometry also
changes with averaging; results cannot isolate parameter count from it.

Reconstruct each control checkpoint's metrics with the same cached embeddings
before fitting. Reject mismatched source, labels, support, folds, initial
tensors, schedules or control metrics. Select native checkpoints using only
the original FIT stopping block. Audit only after both endpoints are fixed.
Save selected and fixed checkpoints, cold-reload parity, all metrics and curves.

All four stopped audit regrets must improve over their matched controls and
executable-opening regret must not worsen to call this consistently helpful.
Also report fixed-budget endpoints and the preserved actor. No automatic actor
update, phase2 pass, live deployment or representation/feature change.

## Test Design

| Requirement | Evidence |
| --- | --- |
| Common initial weights | Native output equals ensemble head0 for each twin; exact source digest |
| No dormant parameters | Exactly2 heads; parameter count is one third; both receive gradients |
| Source immutable | Copy ownership test, source tensors and full TRAIN actor outputs exact |
| Reuse authentic controls |318 seals, reconstructed endpoints and schedule identities |
| No temporal selection leak |318 blocked folds and tested RegretStopper unchanged |
| Autonomous execution | Four sequential resumable arms, lock, progress, failed/completed receipts |

## Completed Result

Runtime `bddfb7db`, identity
`00d2f39584defa061924fb26869e3dbd3eb47b6c1ff9533e4aec291b14cfd67c`.
Four new fits completed in19.94 seconds, with four318 controls reused, not
retrained. Cached frozen embeddings explain the short duration: these are
head fits, not full-network RL training.15 focused tests passed; CLI smoke
and independent reconstruction of all four stopping decisions passed.

Lower native action regret is better. These are held-block diagnostic scores,
not account profit percentages. Both chronological folds are within TRAIN;
the encoders were previously trained on all TRAIN, so these are not fully
independent end-to-end holdouts.

| Fold | Seed | Native stopped | Ensemble stopped | Native fixed4096 | Ensemble fixed4096 | Actor |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A |19|0.255390|0.297626|0.284994|0.311319|0.168404|
| A |43|0.216413|0.238383|0.277776|0.273249|0.168404|
| B |19|0.213729|0.277413|0.231560|0.252317|0.249439|
| B |43|0.225548|0.267339|0.235676|0.259326|0.249439|

The prespecified stopped comparison passes4/4, including the executable-opening
guard. Equal-weight mean regret falls0.270190 ->0.227770, a15.70% relative
reduction. This is a descriptive paired result, not a significance claim.
Fixed-budget native endpoints improve3/4: A/43 worsens slightly. Native stopped
endpoints beat their own fixed endpoints4/4. Against the retained actor,
native heads win both B cases but lose both A cases.

Selected updates are960,1408,64,1024; patience expires at2240,2688,1344,2304.
The rest of each4096-update trace is only the prespecified fixed-budget
comparison. In particular, B/19 selecting64 is an early optimum, not evidence
that additional fitting helped it. FIT regret falls strongly while the stopping
block does not follow; reducing diagnostic capacity has not eliminated that gap.

## Verification And Limits

- All eight reused control endpoints reproduced their complete metric dictionaries
  exactly from stored checkpoints on the same frozen cache.
- Shared initialization and minibatch schedule digests match318. The native
  initial function differs from the ensemble mean by0.873155/0.918571 max per
  seed; only common weights, not initial functions, are matched.
- Four native selected/fixed checkpoint pairs cold-reload exactly. Output seals
  were verified again after collection; all four stopping decisions were
  reconstructed from curves without using the stopper implementation.
- Source state tensors and all24026 TRAIN actor logits remain exact.71 inputs,
 28 groups and49 cumulative return candidates are unchanged. No2024 inference,
  TEST read, actor update, encoder update, new floating pass or STL inclusion.
- The source retains the historical mixed-COT limitation recorded in311; this
  diagnostic does not qualify it for live use or authorize data substitution.
- The optional repository ruff executable was absent; no lint-clean claim.

## Disposition

Use native one-head-per-twin for subsequent critic diagnostics. Do not promote
these calibrated heads into the actor's training loop yet: improvement over an
oversized diagnostic ensemble does not establish reliable improvement over the
actor on both folds. Next, attribute remaining native errors by action/state
and target common-level versus action-gap error using these sealed endpoints;
this finite CPU review precedes choosing one new loss or coverage contrast.
Keep the original actor299 and paused STL. No additional unchanged floating
pass is justified before the refinement issue is resolved.

Evidence: [paired table](generalization/critic_native_capacity_20261113/COMPARISON.md),
[full endpoints](generalization/critic_native_capacity_20261113/RESULT.json),
[configuration](generalization/critic_native_capacity_20261113/CONFIG.json),
[curves](generalization/critic_native_capacity_20261113/CURVES.csv),
[training plot](generalization/critic_native_capacity_20261113/STOPPING_CURVES.png).
The inherited plot contains two unused panels; its populated panels are A/B,
and the numeric CSV is complete.

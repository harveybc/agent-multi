# 345: Bounded Targeted Rescue Continuation

Status: COMPLETE_INDEPENDENTLY_VERIFIED.73 focused tests and scoped Ruff
pass.344's independently verified corrections
occur in five witnesses covering three of four cases. Test accumulation before
considering any actor change or longer fit.

Use the four334 sources and the identical first16 batches, LR, AdamW state and
guards used in342. Compare against its saved ENDPOINT_16 controls; no repeated
control fitting. At each step first compute the ordinary341 correction. If it
already strictly improves FIT regret in all five views, use it. Otherwise allow
at most FOUR targeted searches per case (the first four eligible attempts),
each with344's fixed8-candidate/radius4/margin/relinearization rules. Select a
strictly cost-and-regret improving targeted candidate by NumPy FIT regret,
cost,row; otherwise use the ordinary correction or its exact rollback.

Optimizer semantics are explicit: this is projected AdamW. Every admitted
update keeps that step's PROPOSED AdamW moments and increments its counter,
whether the direction is ordinary or targeted. A rejected update restores
weights AND moments. No344 candidate checkpoint supplies optimizer state.
No extra gradient step is taken while examining alternatives. There are16
optimizer attempts per case, but treatment has extra projection compute;
this is not an equal-wall-clock comparison.

Dynamic optimal sets are recomputed each step; newly corrected decisions gain
protection. Original anchors stay frozen. No held-label update selection.
Save8/16 endpoints, cold-check both and replay9/10 from8 exactly with the saved
search counter. Also replay1/2 from the original source, including any targeted
searches. Verify optimizer counts, finite guards and trace chaining.
Require344 verification before launch and preserve all parent seals. Stop
after16 attempts in each case; no budget extension or actor promotion.

Tests: inherited finite guard/derivative/transaction tests; target-choice
preference and fallback fixtures; real cold replay with saved search budget;
independent CPU endpoint scoring and per-step trace verification. Report
matched FIT decisions, descriptive internal TRAIN stop/audit, accepted steps,
targeted searches/accepted rescues and source hashes. No2024/TEST reads.
The producer invokes the independent CPU verifier automatically after sealing;
the supervisor resumes whole incomplete cases with bounded restarts.

## Results

Runtime and automatic verifier:7d475072. Identity
`c98043f7715fbf89431b110568ccb5a111ccde9855fac3e014f308536a6d1ba9`.
Four cases completed in254.09seconds with no worker restart. All8 saved
endpoints cold-reproduce and all16 repeated transitions reproduce exactly,
including the initial targeted searches and the saved search budget. The
independent CPU verifier checked64 trace steps,8 checkpoints, optimizer counts
and action/metric parity; manual forward versus Torch64 differs at most7.99e-15.
The generalized verifier also reproduces342's prior verification byte-exactly.

|Case|Matched control FIT regret|Targeted rescue FIT regret|Relative reduction|Accepted steps|Targeted rescues|
|---|---:|---:|---:|---:|---:|
|A /19|0.156435285|0.148179214|5.2776%|13/16|3|
|A /43|0.141511388|0.139623538|1.3341%|16/16|1|
|B /19|0.193833237|0.192573954|0.6497%|15/16|1|
|B /43|0.231144661|0.231144661|0.0000%|16/16|0|

Mean FIT regret drops from0.180731143 to0.177880342: **1.57737% lower**.
There are3 paired wins and1 tie,5 improved endpoint decisions and0 worsened,
relative to the sources. These are diagnostic case-row decisions, not orders
or independent statistical samples. All16 targeted-search budgets were used;
5 searches produced an admitted rescue. Total accepted updates60/64 versus
64/64 for the stored ordinary controls. The additional projection work means
equal optimizer-attempt budgets, not equal compute budgets.

An important counterexample remains: soft FIT cost is HIGHER than the matched
control in all three winning cases, although each admitted update decreases
cost from its own previous state. Better soft-loss minimization alone would
therefore rank these useful decision improvements incorrectly. Both actual
action quality and the surrogate must continue to be reported separately.

Internal TRAIN stop and audit greedy regret are exactly unchanged versus the
matched controls in all four cases. There is no measured held-decision benefit
yet. B43 stop remains worse than the original source, exactly as in its control;
the new method did not cause or repair that prior transfer issue. Frozen
encoders used TRAIN, so these partitions are not end-to-end held-out validation.
Neither2024 nor TEST was read. Actor299,71 inputs,28 groups and49 return
candidates remain unchanged; no trading or live-model promotion occurred.

## Decision And Next Bounded Step

The proposed repair now improves actual FIT decisions across a short
stateful continuation, not merely a one-step counterfactual. Keep its strict
finite gates and explicit optimizer semantics. Do not substitute a block or
resume a344 optimizer-less checkpoint.

Next: test retention through a matched64-attempt endpoint, resuming these saved
16-step states and their optimizer/RNG/search counters. Keep the TOTAL budget
at four targeted searches per case (already spent), so no new target search or
threshold tuning is allowed. Compare against saved342 ENDPOINT_64 controls;
report FIT and descriptive internal transfer separately. This extension is
now completed as [346](346_TARGETED_RESCUE_RETENTION.md), with a separately
pinned, tested continuation rather than a mutation of this completed16-step
run. All5 gains survive; matched64 results are2 wins,1 loss,1 tie, and A43
misses a control improvement on internal stop. See346 for the current next step.

|Requirement|Evidence|
|---|---|
|Matched data, source and control budget|CONFIG, per-case source/control hashes and identical control readback|
|Finite protected decisions and nonzero accepted motion|64-step trace verification in five geometries|
|Projected optimizer semantics and bounded target budget|Saved optimizer step counts and monotone search counters <=4|
|Cold and resumed reproduction|8 endpoints,16 transitions exact|
|Separate decisions, surrogate and transfer|SUMMARY.csv and full RESULT.json|
|No actor promotion|actor_updates0,model_promotedfalse; bundle hash unchanged|

[Table](generalization/targeted_rescue_20261114/SUMMARY.csv),
[raw results](generalization/targeted_rescue_20261114/RESULT.json),
[verification](generalization/targeted_rescue_20261114/VERIFICATION.json),
[identity](generalization/targeted_rescue_20261114/CONFIG.json),
[plot](generalization/targeted_rescue_20261114/FIT_REGRET.png).

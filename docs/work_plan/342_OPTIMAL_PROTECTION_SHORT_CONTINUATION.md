# 342: Matched Short Continuation After Optimal-Set Feasibility

Status: COMPLETE_VERIFIED_SAFETY_LEARNING_TRADEOFF.66 focused tests and scoped Ruff pass.

341's ten repaired proposals lower loss without changing any executed FIT
action. That demonstrates nonzero safe motion, not improved decisions. After
its independent verifier passes, run64 attempts for each of the four334
source checkpoints, with the SAME batch schedule, LR, AdamW state, source
weights and original guards as338. Reuse its authenticated complete-arm64
controls; do not retrain controls. No long fit, actor update or floating pass.

The sole new treatment is341's per-step expanded FIT optimal-set protection
and finite lower-cost/nonincreasing-regret gate. Recompute eligible sets from
the current state each step, so newly corrected predictions become protected
while wrong predictions remain free. Tied native target optima remain sets.

## Tests And Acceptance

- Require341 producer AND independent verifier gates; source/probe/coverage/
  control identities must agree. No continuation if either gate fails.
- Source replay, batch identity and saved baseline/control metrics must match.
- Preserve complete per-step records,16/32/48/64 checkpoints and optimizer count.
  Every accepted step must pass original and expanded finite guards. Rejected
  steps restore weights and optimizer via the tested transaction helper.
- Cold-reload all four endpoints; replay17/18 from16 exactly. All source
  seals and actor hashes remain unchanged. Four case units are resumable at
  completion; a partial failed case restarts from its authenticated source.
- Report full FIT loss, greedy regret, action changes, acceptance, protected
  counts and internal TRAIN stop/audit metrics for source/control/treatment.
  Held metrics are descriptive only and never choose an update or checkpoint.
- No automatic model promotion. Even four FIT gains would not establish
  end-to-end transfer or account profit. This finishes at64 attempts; do not
  silently extend the budget or retune thresholds.

## Results

Runtime d0b7fcdd; verifier4218dff5. Identity
`a4b51f938a66baa7d6868504a36c1d4163e3881fce73b12ee83bc0e2d67c1382`.
The four cases finished in208.44seconds, with243/256 updates admitted.
All16 cold endpoints and8 resumed transactions reproduced exactly. Independent
CPU verification checked all256 trace steps and16 checkpoints; manual NumPy
versus Torch64 forward maximum difference7.99e-15. Source actor unchanged.

FIT regret is mean foregone native target reward, lower is better. These are
critic diagnostic rows, NOT annual account returns or new trading performance.

|Case|Source regret|Saved control64|Protected64|Accepted|Control harmful switches|Protected harmful switches|
|---|---:|---:|---:|---:|---:|---:|
|A /19|0.156435285|0.151383824|0.153583178|57/64|2|0|
|A /43|0.141511388|0.141511388|0.141511388|64/64|0|0|
|B /19|0.193833237|0.193814801|0.192433535|58/64|2|0|
|B /43|0.231144661|0.229738595|0.229738595|64/64|0|0|

Protected endpoints improve3 target decisions, worsen0 and change2 between
equally rewarded actions, relative to their own sources. Controls improve6,
worsen4 and change4 between ties. Counts are case-row occurrences, not unique
market events or independent statistical samples. Protection eliminates the
four observed harmful changes, but also prevents some gains: one paired win,
one loss, two ties. Mean FIT regret is0.78264% lower than source but0.11419%
HIGHER than matched controls. Do not report this as a uniform improvement.

Internal TRAIN stop regret is0.270333834 versus control0.270380670
(-0.01732%); audit regret is identical0.228386774. Stop remains0.36424%
worse than source. These partitions are descriptive, not end-to-end held-out
validation: the frozen encoder was fitted on TRAIN. No2024/TEST reading or
checkpoint selection used them. Loss decreases in all four protected cases,
but lower loss does not guarantee a better action or transfer.

## Decision And Traceability

Keep the correction as an independently tested safety mechanism, not a promoted
training recipe. No actor update or repeated floating pass follows from this.
The immediately executed [343 component diagnostic](343_PROTECTION_COMPONENT_TRADEOFF.md)
tests whether saved blocks can recover the missed gains without new training.

|Requirement|Evidence|
|---|---|
|Same source, schedule, optimizer and control budget|Pinned CONFIG and source/control checkpoint hashes|
|No harmful optimal-set loss in accepted steps|256-step verifier, all five finite geometries|
|Genuine nonzero loss improvement|243 admitted updates,13 rejected transactions|
|Reproducible saved continuation|16 cold endpoints,8 exact resumed transitions|
|Actual decisions measured separately from loss|SUMMARY includes changed/improved/worsened counts|
|No unsupported promotion|actor_updates0, model_promotedfalse|

[Configuration](generalization/optimal_continuation_20261114/CONFIG.json),
[raw results](generalization/optimal_continuation_20261114/RESULT.json),
[verification](generalization/optimal_continuation_20261114/VERIFICATION.json),
[table](generalization/optimal_continuation_20261114/SUMMARY.csv),
[training diagnostic plot](generalization/optimal_continuation_20261114/FIT_REGRET.png).

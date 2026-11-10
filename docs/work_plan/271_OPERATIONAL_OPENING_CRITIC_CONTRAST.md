# 271. Main lane: operational opening critic correction

Continue269 independently of STL270. Source selected69, calibrated SmoothL1
critic as fixed starting control; zero actor updates and no added state inputs.
Rebuild the158 predetermined TRAIN anchors in two native episodes: FIT2020-22
terminating before2023, and diagnostic MONITOR2023. All fit suffixes terminate
before monitor begins; never reuse the old full2020-23 suffix as a fit label.
Retain actual flat states, including forbidden-entry equivalence controls;
record any anchor that is holding in the restarted episode, do not fabricate
its position. Monitor observation windows must start within2023. Existing
encoders/scalers already saw2023: this is internal diagnosis, not a pristine
generalization claim.2024 never chooses correction parameters.

For each anchor force NOP/BUY/SELL and continue the frozen original policy
with three common-random-number tapes; targets are their mean soft returns.
Compare SmoothL1 versus existing pairwise-gap-plus-level loss, each with
critic-head-only and full-critic-representation scope, seeds19/43,4096 updates
batch128, original learning rate/weight decay. Actor and temperature frozen.
This reuses known losses, not a claim they are new: the corrected experiment
uses selected69, flat-only action semantics, matched three-action targets and
nonoverlapping return-support periods. Report all fixed endpoints and curves.

Tests/gates: actual suffix terminal <= role bound; fit bound <= monitor start;
all labels TRAIN; exact equivalent-action targets on prohibited states;
source/actor hashes unchanged; both twins' fit/monitor ranking/regret and tie
error. Check real parameters move and actor logits remain exact. No actor
refinement or model promotion is triggered merely by lower training loss.
Save diagnostic weights and a CPU-generated learning-curve plot.

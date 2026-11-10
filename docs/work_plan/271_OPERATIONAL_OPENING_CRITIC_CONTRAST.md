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

## Execution record

Initial panel construction finished158 anchors, then stopped before the first
optimizer update: the runner called the one-time PyTorch interop-thread setter
again when reseeding an arm. Fixed by configuring process threads once and
resetting only RNG seeds between arms. The failed attempt remains separate.
The corrected run reconstructs the panel under a new identity:
`aa6f79868bfe332b8c775a33264cb301df2af5844c3bcc0e554d97040d01c595`.
Persistent primary-GPU worker started; the side-lane primary training queue
waits for its authenticated terminal receipt. No actor update or promotion.

### Inference dispatch repair

The second attempt stopped after the first4096-update arm with
`OPENING_CRITIC_ACTOR_CHANGED`. A controlled production probe localized this
to changing requires_grad flags, BEFORE any optimizer update: original reload
and calibrated checkpoint reproduce exactly; freezing actor parameters changes
35542 logit entries by at most1.9073486328125e-6, with ZERO changed greedy
decisions and all actor weights byte-identical. Four diagnostic critic updates
leave that difference unchanged. This was not an actor-weight change.

Do not relax the equality gate. `emit_actor_reference` temporarily restores
the source trainability flags during inference and restores current fit flags
in a finally block. Real GPU verification reproduces all24026 valid TRAIN
rows exactly;9 focused tests pass, including exception restoration. New run
identity572c6cfe079d27a38c2f42e6f966c770fa6d32a21ec604b834234244f07b84df
replaces the failed attempt, which remains retained. Its actor remains frozen
through every optimizer step. STL workers continue independently.

The unqualified first arm reduced fit regret0.101733->0.037473 while internal
monitor regret rose0.103982->0.138988. These are diagnostic Q-ranking regrets,
not trading returns and not a selected model. Await all corrected arms before
interpreting their comparative results.

# Findings and Progress Register

This register links measured mechanisms, adopted practices and native trading
results without treating a calibration result as a profit improvement.

| Date | ID | Finding or progress | Evidence | Operational disposition |
|---|---|---|---|---|
|2026-11-09|F-P2-69|Frozen70 ->69 ->69; VAL return13.80849% ->14.59109%, TRAIN11.51453% ->10.79211%; VAL DD3.62493% ->4.04467%|[225](../225_PHASE2_69_AND_WEIGHT_SPACE_RESULTS.md)|Preserve both models; tradeoff, not dominance on all metrics|
|2026-11-09|F-CRITIC-GRADIENT|Degraded actor's critic-driven gradient norm78.27x entropy term; encoder fixed|[227](../227_ACTOR_LEARNING_MECHANISM_RESULTS.md)|Investigate critic action gaps; not evidence that inputs are useless|
|2026-11-09|F-ADAM-RESET|Empty AdamW state alone worsens matched4096-update calibration:RMSE5.00672 ->6.06244|[227](../227_ACTOR_LEARNING_MECHANISM_RESULTS.md)|Do not treat optimizer-memory reset alone as the remedy|
|2026-11-09|F-CRITIC-HEAD-RESET|Same frozen representation, empty optimizer in both arms: fresh online heads reduce final RMSE6.06244 ->0.29032; zero-Q reference0.29542|[229](../229_CRITIC_HEAD_INITIALIZATION_RESULTS.md)|Owner adopts fresh critic heads plus own-policy calibration for new transferred SAC refinements; never reset actor, inference, selection evaluation or resume|
|2026-11-09|F-PRESERVED-HEAD-SCOPE|On preserved epoch0,retained/fresh heads converge to RMSE0.617907/0.617996; fresh/1 improves TRAIN with identical VAL,8/32 actor updates degrade VAL|[231](../231_PRESERVED_POLICY_RESTORATION_RESULTS.md)|Reset benefit is not universal; calibration alone has not solved refinement. Keep source champions and inspect action-gap/decision sensitivity before long joint training|
|2026-11-09|F-ACTION-BOUNDARY|113 changed-signal occurrences,72 unique interventions;12 native accounts reconcile exactly. Fresh/8 VAL loss comes entirely from one added short; fresh/32 loss comes from two added shorts partly offset by skipping a buy|[233](../233_CHANGED_DECISION_ATTRIBUTION_RESULTS.md)|Inspect executed actions and account effects,not just logit drift or signal counts. Preserve69/70 champions|
|2026-11-09|F-ACTION-RANKING|Fresh critic prefers harmful actions in same-state counterfactuals inside TRAIN too,including suppression of a useful early close; three paired stochastic continuations disagree with its ranking|[233](../233_CHANGED_DECISION_ATTRIBUTION_RESULTS.md)|Qualify alternative-action ranking as well as average RMSE before long refinement; no account-state feedback introduced|
|2026-11-09|F-POLICY-CONTINUATION|A May27 additional short harms the greedy account while all three sampled stochastic soft-return differences are positive|[233](../233_CHANGED_DECISION_ATTRIBUTION_RESULTS.md)|Do not equate greedy cash loss with critic error against a stochastic objective; qualify the execution-policy mismatch explicitly|

## F-CRITIC-HEAD-RESET Details

### 2026-11-09: Paired Action Correction

F-PAIRED-ACTION-REPAIR: [235](../235_PAIRED_ACTION_RANKING_RESULTS.md) completed
all four arms. At32 actor updates, paired stochastic correction preserves
VAL13.80849% versus control12.71370%; paired greedy reaches14.61633%,Sharpe2.205,
204 trades,58.33% wins and3.62490%DD. Greedy TRAIN11.57813% is below the matched
control12.24694% but above unchanged-parent11.51453%. No source model replaced.

F-RANKING-REPAIR-LIMIT: improved trading is not a demonstrated global ranking
repair. Greedy monitor accuracy remains50.51% and regret worsens0.16274 ->0.18676;
early-close diagnostic remains4 helpful/2 harmful. Replication and attribution,
plus separation of continuation-policy and entropy effects, are the next gates.

### Original Reset Finding

Priority: critical to qualification of further phase2 refinement. Evidence:
one initialization seed,4096 updates,24025 TRAIN transitions,unchanged actor,
encoders and alpha; alternate action draws are not independent market holdouts.
RMSE reduction95.21% versus retained degraded heads but only about1.73% versus
constant zero. Improvement includes the change in initial value scale; it is
not95% additional learned predictive signal. No trading uplift measured yet.

Per-action RMSE improves NOP5.57336 ->0.26914,BUY18.98708 ->0.62879,
SELL9.98134 ->0.58587. NOP95.83% of the degraded replay; BUY bias remains-0.21167.
The historical origin of degraded heads and recurrence under bootstrapped
joint SAC remain questions for the controlled probe, not closed conclusions.

Implementation and next experiment:[230](../230_CRITIC_REFINEMENT_POLICY_AND_PROBE.md).
Original checkpoints and invalid/superseded evidence are not silently overwritten.

## 2026-11-09: Exact Attribution Of Paired-Greedy Improvement

F-RANKING-UPLIFT-ATTRIBUTED: [236](../236_RANKING_ATTRIBUTION_AND_REPLICATION.md)
replayed both full accounts and reconciled native chronological interventions
with zero error in all four reference/period contrasts. Validation uplift versus
unchanged70 is+80.783650 USD from five nonzero changes, all SELL-to-NOP while
flat: three helpful, two harmful. This is an entry-selection change, not evidence
that early closing was repaired. Versus matched control32, validation improves
190.262400 USD while TRAIN falls66.881150 USD. Two prespecified correction-seed
replications and the entropy-separated stochastic arm now run automatically;
neither dates nor labels are adapted to these attribution results.

## 2026-11-09: Correction Replication Completed Before Reboot

F-RANKING-REPLICATION: [237](../237_RANKING_REPLICATION_RESULTS.md) authenticates
1190 files after reboot, ten completed arms and60 eligible full accounts.
Greedy correction validation improves versus matched controls in both new seeds
(+1.34368/+0.23740pp), but not consistently versus the untouched source actor.
The reward-only stochastic variant has opposing paired effects, so entropy
removal alone is not a supported repair. Greedy diagnostic accuracy improves
modestly, whereas diagnostic regret has mixed changes. Preserve the source
weights and evaluate subsequent refinement against unchanged-parent accounts
as well as matched-update controls. This is not an early-close repair finding.

## 2026-11-09: Actor Anchoring Mitigates Drift

F-ACTOR-ANCHOR: [239](../239_ACTOR_PRESERVATION_RESULTS.md) completed all six arms
with native checkpoint parity in both seeds. Beta1 KL(parent || current) reduced
mean validation probability drift about97%, changing11/26 greedy actions versus
65/113 for native. Both KL and learning-rate/3 improve the final256-update
validation return against their native controls, but fail to outperform the
untouched parent in both seeds. This supports drift mitigation, not a resolved
refinement or generalization mechanism. Preserve source weights and inspect
residual harmful action changes from stored endpoints before further fitting.

## 2026-11-09: Residual Anchored Losses Attributed Exactly

F-ANCHORED-ORDERING: [241](../241_ANCHORED_ACTION_ATTRIBUTION_RESULTS.md) reconciles
all four account differences exactly using624 suffix simulations. Among seven
harmful validation event occurrences, six are favored by the minimum-twin
critic despite negative matched realized discounted advantage. One NOP-to-BUY
change occurs against critic preference. Twin disagreement covers4/7 harmful
events but three are favored by both twins; consensus is not a sufficient fix.
Small average KL hides local changes and not all affected rows are near ties.
In this validation event panel cash and discounted-reward signs agree, so a
different fitness is not indicated by that explanation. No training, new input,
source mutation or validation-fitted margin threshold occurred.

## 2026-11-09: Action-Gap Calibration Has Small Mixed Benefit

F-ACTION-GAP-CALIBRATION: [243](../243_ACTION_ADVANTAGE_CALIBRATION_RESULTS.md)
authenticates all four arms and reproduces both absolute controls exactly.
Pairwise-plus-mean calibration improves fit-anchor ordering and TRAIN return
in both replicas. Fixed256 VALIDATION effects against control are0 and+0.11855pp.
Monitor ordering falls in one replica and rises in the other; early-close
harmfulness counts are unchanged. No consistent improvement over the untouched
parent is established. Preserve this candidate and inspect its changed actions
before claiming that critic ordering or refinement has been repaired.

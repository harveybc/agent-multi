# Findings and Progress Register

This register links measured mechanisms, adopted practices and native trading
results without treating a calibration result as a profit improvement.

| Date | ID | Finding or progress | Evidence | Operational disposition |
|---|---|---|---|---|
|2026-11-13|F-NATIVE-SAME-MARKET-STATE-AMBIGUITY|316:16 native origins,144 suffixes plus9 exact repeats;3 origins have incompatible optimal actions despite identical next market tensor. Preserved actor held regret0.200629 beats calibrated heads0.263956;15/55 held windows overlap FIT|[316](../316_CRITIC_IDENTIFIABILITY.md)|Specify marginal versus position-specific Q estimand before more actor refinement. This does not prove no signal or invalidate market-only SAC; no feedback added, actor unchanged,71/49 pool preserved|
|2026-11-13|F-CRITIC-REPRESENTATION-NEGATIVE-CONTROL|315 completed6 fits plus2 reused controls. Representation training worsens true-label regret/RMSE in both seeds; shuffled representation attains lower regret despite worse RMSE|[315](../315_AUTOMATED_CRITIC_REPRESENTATION.md)|No promotion. Critic value fit is not sufficient action evidence; native identifiability and FIT-only baseline diagnosis follows|
|2026-11-13|F-STL-OWNER-PAUSE|All paired/legacy STL queues stopped and disabled; checkpoints retained, no seasonal inputs admitted|[314](../314_CRITIC_COVERAGE_CONTRAST.md)|Overrides earlier STL continuation entries. Resume only under a new owner request|
|2026-11-13|F-STL-CONTROL-IDENTITY-AND-FULL71|STL31 raw and seasonal were fresh31-input networks, not the preserved71-input phase2 model. Continuous improves its matched raw control. Owner authorizes all71 with reduced common temporal support;1027 donor tensors copy exactly;47 tests pass|[313](../313_FULL71_MATCHED_STL.md)|Run raw/continuous from same preserved weights, repaired activity in both, no pruning; retain original model and cumulative49 return candidates. Seasonal donor transplant requires explicit compatibility tests, not automatic admission|
|2026-11-12|F-FROZEN-Q-COVERAGE-AND-POLICY-DRIFT|312 completed88 new TRAIN anchors/1320 same-state continuations in325.33s. New executable-flat pair accuracy47.92%/47.22%; changed actor alters optimal set14-23/48 times. First effects and parent states exact|[312](../312_FROZEN_CRITIC_COVERAGE_AND_DRIFT.md)|Coverage and stale continuation targets both matter; no width-only fix or causal attribution of whole account losses. Plan own-policy critic refresh on qualified corrected data; retain299|
|2026-11-12|F-COT-IDENTITY-MATERIAL-POLICY-DEPENDENCE|311 frozen weights/scaler/rows: corrected EURO FX counts under legacy clock change1202 TRAIN and527 VAL signals; VAL18.369946% ->1.541746%,207 ->122 trades. Original replay exact. Independent arithmetic detects104 pre-TRAIN warmup nulls contradicting declared min52|[311](../311_COT_IDENTITY_IMPACT.md)|Preserve checkpoint, no transparent live substitution. Qualify actual release/vintage times and repaired data before refit/phase2. Positive diagnostic is not causal certification. Retsu receives source/live completion tasks|
|2026-11-12|F-HEAD-CAPACITY-FROZEN-Q-LIMIT|310: four paired fresh heads,20000 updates each,143.49s on cached CPU. Head16 improves frozen-Q objective and greedy agreement in both seeds, but VAL delta vs8 is-2.786/+2.529pp; all endpoints lose versus299. Cache/native/cold logits and non-head tensors exact|[310](../310_ACTOR_HEAD_WIDTH_CONTRAST.md)|Retain299. Better frozen-critic fit is not account improvement. Examine Q extrapolation from113 FIT anchors and policy drift before more resets/core widening. No automatic promotion or unchanged floating|
|2026-11-12|F-SELECTED71-HEAD-FUNCTIONAL-SPILLOVER|309 CPU on24026 TRAIN windows: frozen encoder exactly equal; six greedy flips match GPU. Hidden-head response dominates target margin change (~97%); final winning margin only5.3e-6. Core effective rank46.86/208 versus head3.96/8 is descriptive, not a capacity theorem|[309](../309_SELECTED71_SPILLOVER_AND_CAPACITY.md)|Preserve299. Prioritize paired head8/16 before core8/16/32; no automatic promotion or new unchanged floating. Retsu owns real-source qualification including COT; STL31 is not a deployability selection|
|2026-11-12|F-FIT-SOURCE-VALUE-FLOOR|303 protects already-wrong choices from worse alternatives:113 FIT checks,79 constrained rows,105 margins;16/16 full nonzero updates,zero native FIT regressions. The302 account losses disappear; both seeds match control16.058224% TRAIN /18.369946%VAL with exact trades/equity|[303](../303_SELECTED71_SOURCE_VALUE_FLOOR.md)|Adopt this guard contract, retain original actor. Loss improves slightly but zero useful decisions change; four changed symbols occur only at blocked entry times. Diagnose useful-boundary feasibility under unchanged constraints before longer training or new floating|
|2026-11-12|F-OPTIMAL-SET-REPAIR-AND-SUBOPTIMAL-REGRESSION|302 repairs16 omitted HOLD guards and2 tied targets:42 protected rows,65 errors,16/16 accepted updates,1023 frozen tensors exact. Both endpoints nevertheless lose87.326775USD TRAIN and24.091780USD VAL. FIT row107 increases optimal SELL probability while switching NOP to worse BUY|[302](../302_SELECTED71_OPTIMAL_SET_REFINEMENT.md)|Keep selected71 control. Correct-action preservation plus lower set loss does not protect already-wrong choices from becoming worse. Next test a source-relative native FIT value floor on all executable FIT states; never fit MONITOR/VAL timestamps|
|2026-11-12|F-HOLD-TIE-GUARD-AND-TARGET-GAP|301: unique-optimum guard omits16 correct FIT HOLD decisions; unique-optimum target builder omits2/65 errors, including a harmful long close. Two-row witness passes old guard despite turning good HOLD into bad CLOSE|[301](../301_SELECTED71_DECISION_DIAGNOSIS.md)|Implement versioned optimal-action-set protection and correction before another bounded update; allow equivalent HOLD symbols, protect against bad actions, verify actual greedy membership. No attribution of all prior losses and no real policy change in diagnosis|
|2026-11-12|F-SELECTED71-CRITIC-ACTOR-TRANSFER|301:82 executable FIT openings,actor optimal26 versuscritics52/35; projected shared first-AdamW lowers optimal mass50/60 versuslocal28/34. Native FIT targets alone still lower63/57 under shared projection|[301](../301_SELECTED71_DECISION_DIAGNOSIS.md)|Keep local derivative evidence separate from finite trading updates; no entropy-only or unconstrained optimizer retry. Actor misses8 helpful FIT closes; neither critic repairs all closing cases|
|2026-11-12|F-SELECTED71-CRITIC-RELATIVE-VALUES|300:4096 head updates/seed,151 fresh anchors. MONITOR ranking58->62% /40->49% and regret improve; action-gap error falls but common-level error rises. Ranking peaks at2048, absolute error earlier|[300](../300_SELECTED71_CRITIC_CALIBRATION.md)|Keep fixed endpoints and actor unchanged (1019 frozen tensors exact); diagnose executable boundaries before correction. Record stopping-budget opportunity without silently choosing the best MONITOR checkpoint|
|2026-11-12|F-REFINED72-CUMULATIVE-TIE|299:72 ->71 ->71,28 groups; elapsed hours removed on return tie, all49 absent features tested in return. TRAIN gains24.003430USD; VAL net unchanged but Sharpe2.692275 ->2.690712|[299](../299_SELECTED72_CUMULATIVE_SELECTION.md)|Two validation entries shift and29 equity points change with identical per-trade net. Equal terminal profit does not imply equal risk path. Preserve controls and exact71-input export; refresh native critics before next refinement|
|2026-11-12|F-SELECTED72-INTERIOR-UPLIFT|298 recomputes25 protected decisions and64 error targets;16/16 full steps accepted. Both schedules gain20.546375USD TRAIN,VAL unchanged18.369946%;747/207 trades|[298](../298_SELECTED72_INTERIOR_REFINEMENT.md)|Preserve source and endpoints; full-account gain is one additional SELL outside sampled errors, not correction of their64 argmax actions. No early-close improvement. Qualify/export fixed endpoints then fresh cumulative phase2 pass|
|2026-11-12|F-SELECTED72-BOUNDARY-TRANSFER|297: critics select native optimum52/82 and34/82 executable FIT probes; actor25/82. Correct native targets reduce harmful local directions to3 but shared first-AdamW projections harm64/59 optimal probability masses|[297](../297_SELECTED72_BOUNDARY_DIAGNOSIS.md)|Local derivative diagnosis, not applied trades. Rebuild72-input FIT error/protection sets and use interior decision-preserving refinement; no unguarded SAC, source replacement or live promotion|
|2026-11-11|F-COT-CONTRACT-ORDER|Retsu reports mixed contracts in cot.eur historical source; pinned producer keeps last duplicate date without contract filter. Independent two-row unit fixture flips net_long-5 ->100 when reordered; original-file impact not yet re-counted|[live handoff](../../handoffs/MUSASHI_TO_RETSU_SELECTED72_LIVE_GAPS_2026_11_11.md)|Block substitution of clean EURO FX into old recipe; recover raw identity and rebuild corrected store if confirmed. All three COT columns remain in72 actor; research values are store-specific, not live readiness|
|2026-11-11|F-SELECTED72-CRITIC-CALIBRATION|296 rebuilds151 own-policy anchors,4096 critic-head steps/seed. FIT pair accuracy76.27%/67.46%,MONITOR62%/49%; MONITOR value RMSE worsens. Actor and trading unchanged|[296](../296_SELECTED72_CRITIC_CALIBRATION.md)|Preserve actor15.612726% TRAIN/18.369946%VAL; diagnose executable TRAIN boundaries before interior update. Calibration success is not a trading improvement|
|2026-11-11|F-INTERIOR-CUMULATIVE-READMISSION|295 pass69 ->69 ->72,28 groups: no beneficial deletion but three historical readmissions;112 evaluations include all51 absent inputs. TRAIN14.756076 ->15.612726%,VAL16.342145 ->18.369946%|[295](../295_INTERIOR_ACTOR_CUMULATIVE_SELECTION.md)|Feature usefulness changes with the actor/context; not a global69/72-input limit. Preserve69 control and qualified physical72 actor; new critics require own-policy calibration. Early-close quality not uniformly improved; no automatic live promotion|
|2026-11-11|F-INTERIOR-PROJECTION-REPAIR|294 fixed FIT numerical cushion changes acceptance0/16 to16/16 without relaxing1e-7 guard; all26 correct actions preserved. Both endpoints gain84.302095USD TRAIN and0.600545USD VAL; four account deltas reconcile exactly|[294](../294_INTERIOR_FIT_PROJECTION.md)|Adopt interior proposal in this corrective path, preserve nonlinear checks and source.53/63 FIT error probabilities improve but none crosses its greedy boundary; early-close quality unchanged. Qualify changed actors then cumulative floating, no default refit or live promotion|
|2026-11-11|F-TARGETED-ERROR-GUARD-BOUNDARY|293 targets63 FIT errors with26 protected anchors; both schedules accept0/8. First full proposals lower corrective loss and meet trust ceilings but violate protected margins by9.54e-7/3.28e-7 with zero protected action flips|[293](../293_TARGETED_FIT_ERROR_CORRECTION.md)|Retain exact frozen source. Diagnose/repair interior nonlinear feasibility without loosening the1e-7 gate; no learning or profit improvement claimed, no unchanged floating repeat|
|2026-11-11|F-FIT-PRESERVATION-FEASIBLE|292 protects26 source-correct FIT decisions; raw guard accepts0/8 twice, projection4/8 and5/8; unchanged full trading actions and VAL16.336139%|[292](../292_FIT_DECISION_PRESERVATION.md)|Feasible nonzero refinement without regression here, not a profit improvement. Retain frozen reference; target incorrect FIT decisions next, no default unchanged floating|
|2026-11-11|F-GUARD-DISPATCH-PRECISION|Actor trainability changes GPU kernel dispatch by9.54e-7 logits without changing actions; mismatched reference falsely violates1e-7 guard|[292](../292_FIT_DECISION_PRESERVATION.md)|Capture source and candidate panels under identical trainability and batch shape; preserve tolerance; exact starting-panel gate before optimizer|
|2026-11-11|F-SELECTED-CRITIC-AND-TRANSFER|291 critic picks native-optimal action45/82 or36/82 executable FIT anchors; native-target independent logits lower only2 optimal probabilities but shared AdamW lowers63/60. Actual SELL-NOP crossings explain missed VAL entries|[291](../291_SELECTED_REFINEMENT_CAUSE_DIAGNOSIS.md)|Both target ranking and shared update matter; first head weights contribute71-88% of posthoc linear shift. Preserve frozen export; next FIT-only preservation feasibility, no claim that freezing a layer is already a fix|
|2026-11-11|F-SELECTED-EXPORT-PARITY|Physical69-channel,28-group actor preserves313 mapped tensors;30234-window actions and all accounts exact; cold export exact|[290](../290_SELECTED_ACTOR_EXPORT_AND_REFINEMENT.md)|Retain frozen export; critics explicitly fresh/uncalibrated until separate FIT calibration|
|2026-11-11|F-REFINE-MISSED-SHORTS|Both bounded refinement seeds improve TRAIN but lower VAL16.336139% to15.725127%/15.262069%; missed profitable SELL entries explain exact losses; monitor critic RMSE worsens despite FIT improvement|[290](../290_SELECTED_ACTOR_EXPORT_AND_REFINEMENT.md)|Do not replace frozen baseline; diagnose FIT critic ranking versus shared-head transfer; no claim that representation lacks signal|
|2026-11-11|F-CUMULATIVE-READMISSION-UPLIFT|289 pass69 ->68 ->69 swaps px.range_12 for historical RAW_BAR_LOG_RETURN;113 evaluations;VAL14.693262 ->16.336139%,DD4.044574 ->3.257322%|[289](../289_COVERAGE_ACTOR_CUMULATIVE_SELECTION.md)|Full historical return matters; preserve prior model because TRAIN DD worsens7.930285 ->9.548436%;313 current tensors exact; qualify changed-input trainable export before refinement|
|2026-11-11|F-COVERAGE-PAIRED-TIE|113 versus437 FIT anchors, same critic and8x128 budget: all four arms TRAIN13.201812%,VAL14.693262%;32 bounded updates and eight exact attributions|[288](../288_TRAIN_COVERAGE_CONTRAST.md)|No measured coverage advantage here; preserve all checkpoints; qualify narrow19/43 before changed-policy floating|
|2026-11-11|F-LOCAL-GAIN-MIXED-DECISIONS|Net TRAIN gain49.31USD includes skipped profitable SELL(-36.46), avoided losing SELL(+56.45) and BUY(+29.33); no VAL gain and no early-close improvement|[288](../288_TRAIN_COVERAGE_CONTRAST.md)|Net uplift is not universal action improvement; preserve decision-level attribution and do not claim solved generalization|
|2026-11-11|F-BOUNDED-CANDIDATE-QUALIFIED|Both286 critic/AdamW checkpoints exported with full native/cold parity;fresh phase2 pass69 ->69 ->69,112 evaluations and all51 historical removals;313 current tensors exact|[287](../287_BOUNDED_CRITIC_ACTOR_SELECTION.md)|Preserve qualified primary19/comparison43;TRAIN12.708707%,VAL14.693262% unchanged by selection;no identical repeat or default refit;next design targets broader TRAIN coverage|
|2026-11-11|F-BOUNDED-LOCAL-REFINEMENT|Critic/AdamW endpoint8 gains31.656215USD TRAIN in both schedule seeds with identical VAL14.693262%;one harmful BUY skipped and downstream effects account exactly|[286](../286_CRITIC_TARGET_VS_UPDATE_GEOMETRY.md)|Retain snapshots for qualification then changed-policy floating;no claim of new validation gain or a generally solved refinement problem|
|2026-11-11|F-TARGET-GEOMETRY-INTERACTION|Critic/projection loses59.69/116.71USD VAL; native-FIT/projection preserves VAL in both seeds but creates a harmful TRAIN early close costing20.20USD|[286](../286_CRITIC_TARGET_VS_UPDATE_GEOMETRY.md)|Both target quality/objective and shared update transfer matter;same drift ceilings do not guarantee same realized step magnitude|
|2026-11-11|F-POLICY-TARGET-REFRESH|86/151 native continuation targets changed after actor refinement despite identical position direction at all anchor timestamps|[286](../286_CRITIC_TARGET_VS_UPDATE_GEOMETRY.md)|Rebuild TRAIN continuation targets for the current actor before new critic fitting;retain old targets only with their original policy identity|
|2026-11-11|F-ABSOLUTE-QUALIFIED-CUMULATIVE|Saved282 endpoint qualifies both seeds; fresh pass69 ->69 ->69,112 evaluations including all51 historical removals;TRAIN12.392145%,VAL14.693262% preserved|[285](../285_ABSOLUTE_ACTOR_CUMULATIVE_SELECTION.md)|Both research bundles saved;313 current weights preserved, exact cold/accounts; no repeated unchanged screen or refit; critic/shared-head refinement remains the next mechanism to address|
|2026-11-11|F-FUNCTIONAL-DIRECTION-NOT-PROFIT|Six bounded-update arms pass all48 steps and12 exact account reconciliations; projection actual/local-target cosine0.932-0.962 but no new endpoint beats282 in TRAIN and VAL|[284](../284_BOUNDED_FUNCTIONAL_ACTOR.md)|Preserve282 absolute19/43; qualify saved actor for fresh cumulative floating; no default optimizer change|
|2026-11-11|F-FUNCTIONAL-TRADING-TRADEOFF|Projection19 adds a harmful TRAIN early close; projection43 loses52.351235USD by skipping a profitable VAL short despite gaining10.217220USD from the known avoided early close|[284](../284_BOUNDED_FUNCTIONAL_ACTOR.md)|Local objective/monitor regret improvements do not replace full native account and activity checks; same trust ceilings are not equal realized displacement|
|2026-11-11|F-ACTOR-CROSS-SAMPLE-INTERFERENCE|Native-return oracle improves21 of24 optimal MONITOR opening probabilities under independent-logit descent,3 negligible changes; shared FIT head gradient lowers23-24; six first-step checkpoints replay exactly|[283](../283_ACTOR_GRADIENT_COUPLING.md)|Confirmed local coupling alongside remaining critic error; no optimizer-sign bug or absent-signal claim; test bounded functional update before long refinement|
|2026-11-11|F-ACTOR-MARGIN-NOT-PROBABILITY|Absolute19 first step lowers24 useful monitor margins against NOP but23 action probabilities; third action moves too|[283](../283_ACTOR_GRADIENT_COUPLING.md)|Report probabilities and executed account effects, not margin signs alone; aggregate objective may improve despite individual suppression|
|2026-11-11|F-BOUNDARY-CALIBRATED-UPLIFT|Absolute calibration both seeds: TRAIN11.565540 ->12.392145%, VAL14.591090 ->14.693262%;12 endpoint account deltas reconcile exactly|[282](../282_OPENING_BOUNDARY_REFINEMENT.md)|Preserve both candidates; TRAIN gain from10 opening changes, VAL gain from one already-known avoided early close; qualify before new floating pass|
|2026-11-11|F-SHARED-HEAD-SUPPRESSION|Calibrated shared-head gradients suppress every beneficial opening margin in the sampled FIT/MONITOR panel; absolute19 includes26/5 comparisons also favored by Q|[282](../282_OPENING_BOUNDARY_REFINEMENT.md)|Separate local softmax competition, cross-sample coupling and AdamW effects before changing loss; no claim that calibration alone repairs refinement|
|2026-11-11|F-ALL-STATE-BOUNDARIES|Eight-arm critic/entropy contrast completed; full account losses trace to harmful NOP-to-SELL opening changes; all16 endpoint deltas reconcile exactly|[281](../281_ALL_STATE_REFINEMENT_CONTRAST.md)|Qualify TRAIN-only executable action boundaries, not aggregate Q error; preserve reference69|
|2026-11-11|F-CRITIC-CONSTANT-CONTROL|Recalibrated critic monitor ranking67.65% does not beat constant-BUY regret; one seed chooses BUY16/16; actor's monitor greedy decisions unchanged|[281](../281_ALL_STATE_REFINEMENT_CONTRAST.md)|Add constant-symbol and actual greedy-boundary controls; no claim of useful learned discrimination from ranking alone|
|2026-11-11|F-ONE-CLOSE-TRADEOFF|Calibrated seed43 VAL14.59109 ->14.69326% from avoiding one harmful early close before forced-flat;TRAIN11.56554 ->11.54582%;other seed worse|[281](../281_ALL_STATE_REFINEMENT_CONTRAST.md)|Retain checkpoints as research tradeoff; no automatic champion/live replacement or global early-close improvement claim|
|2026-11-11|F-CHANGED-ACTOR-CUMULATIVE|Qualified one-update actor69 ->69 ->69;112 fresh evaluations include51 historical removals; no accepted change;TRAIN11.565540%,VAL14.591090%|[280](../280_ONE_UPDATE_CUMULATIVE_SELECTION.md)|Preserve candidate and original; do not repeat identical screen or call frozen donor rejection proof against retraining|
|2026-11-11|F-COMPACT-TRANSPORT-SHAPE|Physical compaction changed input shape but left transport shape stale; five regression failures reproduced and repaired;36 focused tests pass|[280](../280_ONE_UPDATE_CUMULATIVE_SELECTION.md)|Update both dimensions, reject unsupported long-context compaction; no retroactive performance claim|
|2026-11-11|F-ONE-UPDATE-QUALIFIED|Both saved controls reproduce full accounts and cold logits;TRAIN10.792108% ->11.565540%,VAL14.591090% unchanged;4 head tensors changed,337 actor tensors exact|[279](../279_ONE_UPDATE_ACTOR_QUALIFICATION.md)|Preserve research bundles;fresh-cache cumulative floating can use changed actor;no live promotion or new validation gain|
|2026-11-11|F-STL-SOURCE-REPAIR|Four repaired FX inputs recover479 original windows under unchanged STL parameters;December2023 regains222 windows|[278](../278_STL_REPAIR_AND_PREHISTORY_RESULTS.md)|Separate diagnostic candidate;January2020 andJune2022 still have no common STL support;no training release|
|2026-11-11|F-STL-PREHISTORY|Original producers reconstruct34/36 needed columns from1567 real pre-TRAIN rows;baseline error below4e-14,prefix error0|[278](../278_STL_REPAIR_AND_PREHISTORY_RESULTS.md)|Calendar and cross-fit activity scope remain unqualified;no synthetic context or admission|
|2026-11-11|F-USDCHF-ZERO-VOLUME|Native missing-hour archive record is flat,zero-volume and equal to preceding close|[278](../278_STL_REPAIR_AND_PREHISTORY_RESULTS.md)|Reject carry-forward as a real observation;do not remove model inputs or overwrite the source|
|2026-11-10|F-P2-CUMULATIVE|69 ->69 ->69; all51 historical phase2 removals tested in46 return blocks;112 evaluations; no accepted change|[263](../263_CUMULATIVE_PHASE2_RESULTS.md)|Keep current69/28 actor; frozen donor-assisted rejection is not evidence against matched retraining|
|2026-11-10|F-WEIGHT-RESTORATION|All313 current actor tensors exact after restoring archived connections;30,234 windows give zero action differences and exact native accounts with absent channels masked|[262](../262_CUMULATIVE_PHASE2_RETURN.md), [263](../263_CUMULATIVE_PHASE2_RESULTS.md)|Historical readmission no longer silently limited to same-pass removals; preserve source and donor/scaler lineage|
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

## 2026-11-09: Paired Uplift Comes From Two Validation Entries

F-PAIRED-ADVANTAGE-ATTRIBUTION: [245](../245_PAIRED_ADVANTAGE_ATTRIBUTION_RESULTS.md)
reconciles all four account differences exactly with258 native continuations.
Only9 of43 changed signals affect cash, all entry decisions from flat states.
The r43 validation gain is+56.465540 minus44.610795 USD from two suppressed
SELL entries. Both critic calibrations retain the wrong ranking in both cases;
the useful abstention is against the new critic's preference. No Q-gap sign
repair occurs in any of the9 effective events. Magnitude errors improve in5,
not all. This does not demonstrate a resolved critic or actor update mechanism.
Preserve source weights and inspect TRAIN fitting/coverage before another fit.

## 2026-11-09: Full-Critic Gradient Canary Catches Fused Inference Dispatch

F-CRITIC-FIT-DISPATCH: The246 prelaunch canary passed head-only updates but
failed when unfreezing the critic encoder: functional twin execution selected
an inference-only fused transformer operator without an implemented derivative.
The diagnostic now disables MHA fastpath only during that critic forward and
restores it even on failure. Three real-checkpoint gradient canaries pass;
actor outputs stay exact. This failure was caught before experiment launch
and does not retroactively invalidate the completed head-only contrasts.

## 2026-11-09: Critic Representation Adaptation Passes Real-Label Memorization

F-CRITIC-SMALL-FIT: [247](../247_CRITIC_COVERAGE_AND_SMALL_FIT_RESULTS.md)
authenticates four arms and26 files. Real native-label witnesses reproduce
with zero error. Only80/74 of284 fit anchors are permitted flat entry states.
On32 fixed examples per replica, heads-only minimum-twin pair accuracy reaches
86.46/85.42%; adapting the existing critic encoder and heads reaches100% in
both, with gap RMSE0.000021/0.001334 and zero regret. Actor outputs remain exact.
The existing architecture can fit this diagnostic without growing. This is
memorization, not generalization: monitor accuracy stays mixed, and monitor
state distribution differs from the flat-only fitting set. Broaden coverage
and compare trainability with state-specific monitors before actor refinement.
# F-CRITIC-STRATIFIED-MONITOR-REGRESSION

Results249,2026-11-10:1248/416 quota ceilings yielded1075 fit and351 monitor
states across16 quarterly native episodes. All4278 action suffixes completed,
witness error0.0. Identical initial metrics and minibatch schedules per seed.
Full critic representation adaptation improved fit pair ordering to66.51/67.02%
versus54.66/54.92% for heads, but monitor ordering fell to51.04/51.31% versus
54.70/54.44%. The main deterioration is open-long decision ordering. Aggregate
regret improves slightly, so do not collapse mixed metrics to one verdict.
Early monitor gains decay with more updates while fit improves: evidence of
overfitting in this calibration, not evidence that market inputs are useless.
Source actors unchanged; diagnostic checkpoints not promoted. Follow-up should
isolate module drift and state-dependent value conflicts before bounded actor
refinement. Source pretraining already used the internal monitor year.

## F-PARTIAL-CRITIC-AND-FUSION-RESTORATION

Results251,2026-11-10: two last-core+head fits (6534 parameters) completed
against the exact archived controls and panel. Monitor pair ordering52.48/51.57%
does not beat heads-only54.70/54.44%; source actor unchanged. Partial unfreezing
reduces full-unfreezing gap-error deterioration but is not a generalization fix.
The66 isolated parameter restorations show that resetting fusion reduces
open-long monitor gap RMSE17.46/17.71%, yet worsens aggregate ordering and
regret. This is a conditional module effect, not a global repair or a reason
to discard fusion. All diagnostics and model weights preserved. Next proposed
diagnosis: same-market Q-target compatibility across feasible account states,
without adding state feedback or asserting the cause in advance.

## F-SAME-MARKET-STATE-COMPATIBILITY

Results253,2026-11-10: native histories produced844 comparable identical-market
banks;190 (22.51%) require incompatible optimal actions across flat/long/short
states. All844 have nonzero shared-Q action-gap fitting floors, but654 still
permit a common optimal action. Sixteen identical-state controls have zero
error and all seven model checkpoints preserve identical outputs for identical
inputs. This is an exact finite-panel interface limitation, not evidence that
market inputs are useless or that every policy must fail. Actor mean regret
remains far above the best common-action bound (monitor0.197915 versus0.016972).
Proposed next design: externally role-routed value outputs without new model
inputs, with explicit tests for residual same-direction position differences.
No model or interface promoted; source weights and observations unchanged.

## F-ROLE-HEADS-AND-SAME-DIRECTION-RESIDUAL

Results255,2026-11-10: six fresh-head diagnostic arms, frozen source critic
encoders and actor. Role-routed heads lower2023 monitor bank-balanced regret
from0.193136 to0.185970 (3.71%) versus the same three heads averaged without
role selection. Both seeds improve management of long/short positions, but
flat opening worsens and pairwise accuracy is lower; not global dominance.
No policy or feature-selection evaluator substituted.7074 further native
suffixes expose139/866 same-direction banks with incompatible best actions.
Direction routing is therefore insufficient for all constructed positions.
Next proposed design: market-only curves evaluated against actual native
position constraints externally, with same-direction witness and harmful-close
tests before actor transfer. No extra account-state input introduced.15 tests
passed on coordinator and worker;561 collected files verified, source actor
exact. A tuple/list receipt comparison failure before fitting was corrected
with canonical-content equality and regression, retained as execution history.

## F-POSITION-FIELD-EXPRESSIVITY-VERSUS-LEARNING

Results257,2026-11-10: eight primary and four post-primary constant-market
fits complete with source actor/encoders unchanged. Surface81 mean monitor
regret0.132305 versus equal-size mean810.140061 improves5.54%, but its paired
constant-market control reaches0.129143 and wins in both seeds. Do not attribute
this improvement to learned market patterns. Progress9 has modest market benefit,
but neither learned interface resolves the critical CLOSE,HOLD,HOLD witness.
Surface harmful-close fractions remain52.14%/48.63% in monitor.

Per-bank exact-value oracles are insufficient evidence of simultaneous field
sufficiency. Merging identical market windows and direction collapses866 banks
to860 groups. Exact-value least squares can require6.15e11 coefficients and
lose signs in float32 even when every target requests HOLD. A separate
minimum-infinity-norm margin LP finds correct action signs for both geometric
interfaces across all860 groups, with coefficients <=18.70/15.81 and no float32
sign errors; scalar interfaces have138 infeasible groups. This is output-level
finite-panel feasibility, not proof of neural learnability or generalization.
Next contrast should target decision regret/signs and within-market position
differences, retaining constant-market controls. No actor transfer or new
feature-selection evaluator admitted.13 final focused tests pass.

## F-PAIRED-DECISION-LOSS-AND-OFFSET

Results259,2026-11-10:16 fixed-endpoint diagnostic head fits complete,21 final
tests pass,158 files verified. Surface81 SIGN_PAIR lowers mean monitor regret
0.132305->0.128891 versus SmoothL1. Across two seeds, harmful local closes
306->241, beneficial303->367, total609->608: not merely suppressing action.
Both seeds improve local close quality; account regret improves only in seed19.
FIT conflicting-bank regret improves0.123920->0.097372 while other-bank regret
worsens0.118294->0.148838. No global dominance or smaller-gap success claim.

Constant-market SIGN_PAIR is almost as good (regret0.129421), so evidence for
market-pattern use remains weak. Critical witness still fails: all three HOLD,
but the required CLOSE position now has the highest score in both interfaces
and both seeds. Pair ranking is invariant to a common offset; the sign term
has not calibrated this witness's zero crossing. Next test should isolate
fit-only offset calibration, not choose thresholds from monitor results.
Actor, encoders and selected inputs remain untouched; no model promotion.

## F-FIT-ONLY-THRESHOLD-NOT-A-GENERAL-REPAIR

Results261,2026-11-10:48 frozen-head fit-only calibrations,72 evaluations,
84180 decision partitions.28 tests and48 monitor-perturbation controls pass;
all original metrics and model artifacts reproduce unchanged. Primary surface81
SIGN_PAIR monitor regret worsens0.128891->0.130963 (common) or0.129488
(directional), despite improved FIT. Harmful close counts across seeds increase
241->300/295. The critical primary witness still fails. Other interfaces can
pass that witness without improving aggregate regret, so witness success alone
does not justify promotion. Keep the original thresholds; do not chase offsets
on monitor data. The learned ranking/representation problem is not reduced to
this simple scalar calibration. No claim that more data necessarily fixes it.

Recommended next work returns to frozen selected69 phase2 and component weight
diagnostics, preserving exact functional export and historical donor identity.
Critic refinement and experimental controller integration remain separate.
Existing predictor phase4 configurations and synthetic transport acceptance
were read, not executed or admitted: their compatibility with current joint
RL features is unverified. Audit real source coverage before choosing augmentation.

## F-SELECTED69-OPENING-DRIFT-AND-MIXED-CRITIC-ERRORS

Results267,2026-11-10:18 authenticated accounts,779 changed-signal occurrences,
zero account attribution discrepancy. All58 harmful validation occurrences
are added entries; no added early-close command explains this regression.
In TRAIN,20/60 harmful occurrences conflict with all three matched soft-return
draws,21 change against Q, and47 have a different position from calibration.
These overlapping associations are not independent events or causal proof.
TRAIN-only decision remains MIXED_MECHANISMS, not an automatic loss replacement.

Weight/bias/posterior decomposition closes within7e-7. Posterior averaging's
mean absolute boundary term2.58e-7 is far smaller than weight0.02966 and
bias0.01064 terms. Reduced probability uncertainty did not improve trading;
do not use confidence shrinkage or critic RMSE alone as a promotion criterion.
Keep the69-input source. Next isolate flat-opening state/target consistency
and actor movement against Q on TRAIN, with unchanged/helpful controls, before
further refinement. All worker jobs and collection finished; no training launched.

## F-OPENING-CONTEXT-AND-SHARED-GRADIENT

Results269,2026-11-10:158 TRAIN anchors,2844 native suffixes,21 tests passing.
All60 harmful opening occurrences reproduce. Of20 flat-state Q/soft conflicts,
17 also conflict in the actual calibration history; only3 occur solely outside
it. A concrete NOP->SELL witness reverses value across the two account contexts,
but that ambiguity does not explain every ranking error. Unchanged eligible
controls also expose critic ordering errors. Three draws are not exact Q truth.

The full TRAIN initial mean-head gradient favors all60 harmful changes and all56
helpful changes in the flat-anchor panel.21 harmful directions oppose the local
gradient, showing shared-update interference in this first-order diagnostic,
not a reconstruction of the256-step AdamW trajectory. Strata of11540 eligible
flat,4828 forbidden flat and7657 holding rows sum exactly to the full gradient.
Removing the forbidden-flat contribution algebraically still favors all60
harmful changes: filtering those rows is not an established repair.

Preserve selected69. Next qualify critic action ordering for actual opening
contexts and equivalent actions before another actor update. No training,
state inputs, model promotion or repeated unchanged floating pass performed.

## F-OPERATIONAL-CORRECTION-CAN-SUPPRESS-USEFUL-ENTRIES

Results277,2026-11-11:16 saved accounts, exact native attribution, maximum
cash reconciliation discrepancy0USD. At32 updates, control validation losses
come mainly from two new short entries; corrected-critic losses come mainly
from suppressing three useful short entries. Improving critic calibration
does not establish improvement of the induced actor. A shared final metric
can hide opposite failure mechanisms; retain per-decision counterfactuals.

The saved1-update control gives TRAIN11.565540% versus10.792108%, with
VALIDATION unchanged14.591090% and206 trades. Two helpful effective TRAIN
changes and no harmful ones explain its+77.34311USD. This is a preserved
candidate, not a promoted model or proof of generalization gain. Both seeds
give identical books; do not count duplicate books as independent evidence.

The independent source repair rebuilt four selected FX inputs using verified
November observations. It recovered5/25/257/129 missing values in factor4,
factor24,wavelet,Hilbert respectively, preserving frozen parameters and all
unaffected inputs. Missing USDCHF and pre-TRAIN coverage still prevent STL
release; they are not grounds for halting the separate actor investigation.

## F-SAMPLED-GUARD-DOES-NOT-PROTECT-FULL-ACCOUNT

Results305,2026-11-12: the304 fixed FIT witness crosses one desired boundary
without panel regressions, but changes7 TRAIN and2 validation signals outside
that local guarantee. Full-account deltas are-6.311600USD and-19.762225USD;
cold replay is exact and chronological attribution discrepancy is0USD.
One helpful TRAIN entry is outweighed by a harmful TRAIN entry elsewhere.
Keep299. The next corrective guard must inspect changed TRAIN decisions across
the account, not assume the sampled panel covers shared-head effects.

## F-CAUSAL-FILTER-CAN-BREAK-CROSS-FIT-ISOLATION

2026-11-12: the activity producer's original13-observation,stride2 context has
zero direct target-row membership overlaps. Extending its context with480-row
STL reaches504 raw observations and exposes898 rows whose historical feature
producer was fitted using that target row. This is a TRAIN cross-fit lineage
defect introduced by the extra context, not evidence of validation fitting or
future observations entering the endpoint STL formula.

A separate producer candidate increases embargo96 ->503 and reduces this
direct-overlap diagnostic898 ->0, with exact cold replay and unchanged
validation predictions. It changes21809 TRAIN rows and therefore needs a new
source identity, parameter tuning and a matched raw control before any STL
comparison. Never splice it silently into the preserved model or live provider.
This bounded test is not a blanket causal certification of upstream sources.

## F-ACCOUNT-COUNTEREXAMPLES-REPAIR-TRAIN-NOT-VALIDATION

Results307,2026-11-12: a finite full-TRAIN account counterexample constraint
removes the harmful December2023 opening while keeping the useful July2021
change. TRAIN return16.058224% ->16.571680%, drawdown9.358326% ->8.958845%,
745 ->746 trades. The native chronological attribution closes exactly at
+51.345685USD with no harmful changed TRAIN contribution. Both predetermined
margin modes finish in two proposals and give identical accounts.

Validation remains18.172324% versus source18.369946%, so neither candidate
qualifies for the next selection pass. No validation row supplied a constraint,
gradient or step size. Cold logits and accounts reproduce exactly. The source
remains299; candidate weights are retained for diagnosis, not promoted.

The corrective scope deliberately expanded to the declared2020-2023 TRAIN
period.2023 is therefore not an independent monitor for307; do not carry over
that label from earlier sparse-FIT experiments. Correcting a historical TRAIN
counterexample is not proof that shared-head spillover generalizes safely.

STL71 independently completed all columns and recovered479 windows through
the qualified FX repair. Two empty months and the not-yet-integrated activity
scope repair still prevent matched training. The queue must distinguish
completed preprocessing from an admitted dataset and from a running GPU job.

## F-NONEMPTY-MONTH-IS-NOT-SUPPORT-PRESERVATION

2026-11-12,308: after the owner authorized removal of blocking STL features,
a53-column candidate passed the old nonempty-month gate but retained only4
January2020 windows. It was superseded before training. The final availability
criterion rejects every column losing any original TRAIN window and preserves
coherent blocks; validation does not choose the subset.31 of71 inputs remain,
and the frozen confirmation verifies exact original support in both periods:
24026 TRAIN and6208 validation windows. No synthetic filling or window reduction.

This is a separate raw31/seasonal31 experiment, not a predictive feature-selection
claim or a change to the live selected71 model. Forty removals have explicit
availability/scope/coherence reasons. Both actual GPU workers entered training;
continuous->ordinal and raw->binary are sequential, resumable queues. The
original wider seasonal candidates and partial source repairs remain recorded.

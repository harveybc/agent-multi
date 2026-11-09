# Findings and Progress Register

This register links measured mechanisms, adopted practices and native trading
results without treating a calibration result as a profit improvement.

| Date | ID | Finding or progress | Evidence | Operational disposition |
|---|---|---|---|---|
|2026-11-09|F-P2-69|Frozen70 ->69 ->69; VAL return13.80849% ->14.59109%, TRAIN11.51453% ->10.79211%; VAL DD3.62493% ->4.04467%|[225](../225_PHASE2_69_AND_WEIGHT_SPACE_RESULTS.md)|Preserve both models; tradeoff, not dominance on all metrics|
|2026-11-09|F-CRITIC-GRADIENT|Degraded actor's critic-driven gradient norm78.27x entropy term; encoder fixed|[227](../227_ACTOR_LEARNING_MECHANISM_RESULTS.md)|Investigate critic action gaps; not evidence that inputs are useless|
|2026-11-09|F-ADAM-RESET|Empty AdamW state alone worsens matched4096-update calibration:RMSE5.00672 ->6.06244|[227](../227_ACTOR_LEARNING_MECHANISM_RESULTS.md)|Do not treat optimizer-memory reset alone as the remedy|
|2026-11-09|F-CRITIC-HEAD-RESET|Same frozen representation, empty optimizer in both arms: fresh online heads reduce final RMSE6.06244 ->0.29032; zero-Q reference0.29542|[229](../229_CRITIC_HEAD_INITIALIZATION_RESULTS.md)|Owner adopts fresh critic heads plus own-policy calibration for new transferred SAC refinements; never reset actor, inference, selection evaluation or resume|

## F-CRITIC-HEAD-RESET Details

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

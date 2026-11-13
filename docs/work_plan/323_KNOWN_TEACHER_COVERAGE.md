# Known Teacher FIT Coverage Curve

Owner authorizes the next step after322. Weight-space analysis is not exhausted;
the evidence justifies isolating FIT coverage before more head surgery.

## Prespecified Contrast

Use source299,71 inputs,13 observations at stride2, frozen encoders and native
208->8->3 twin heads. Reuse the four authenticated balanced-teacher322 controls
at8192 updates. Do not retrain them. Two folds x two initialization seeds x
two expansions =8 new fits. A:49 ->196 ->784 FIT examples. B:98 ->392 ->1568.

The exact teacher from each original fold is frozen, including its original
FIT-only tertile thresholds. Increasing coverage must not refit those thresholds
or rebalance the task. Original239 cached embeddings and all old stop/audit
examples are retained byte-for-byte; new embeddings use the same frozen model.
Sample additions without labels, round-robin across available calendar quarters
inside the original FIT years, with deterministic within-quarter permutations.
Expanded sets are nested and retain every original FIT example. Exclude every
other original panel row, preserve feature-window year boundaries and TRAIN
membership, and never add a stop/audit row.

Same initial tensors as322, LR3e-5, WD0.01, AdamW, batch128, MSE and clip norm10.
Primary endpoint:8192 updates for every size. Secondary endpoint:32768 for4x
and131072 for16x, preserving nominal presentations per example. The secondary
endpoint also changes optimization time; report it separately, not as a pure
sample-count effect. No early-stopping checkpoint selection in this diagnostic:
fixed endpoints avoid the known patience/LR artifact. Plot FIT and unchanged
stop/audit value RMSE, action-gap RMSE and action regret. Record sample counts,
quarter/action distributions, actual optimizer parameters and cold reload.

Primary evidence of a consistent effect requires lower audit RMSE AND no worse
audit action regret in all four fold/seed pairs versus their corresponding
original controls. Report every cell regardless of that rule. This is known-
function calibration, not actual account performance or an actor promotion.

## Test Design

| Requirement | Test/evidence |
| --- | --- |
| Nested additional examples | Superset, exact sizes, deterministic ordering, balanced quarter sampling |
| No held-row admission | Explicit exclusion and temporal membership checks |
| Teacher unchanged | Frozen state digest; original target/metric replay exact |
| Original controls unchanged | Seals and complete endpoint metric replay, no control training |
| Same initialization | Original322 state digest equality per seed |
| Real additional inputs | Pinned store/support, finite native windows, no synthetic feature substitution |
| Fixed objective | Teacher targets only for this diagnostic; original trading labels untouched |
| Autonomous sequence | Immutable runtime, per-arm checkpoints/resume, exclusive lock, progress and seals |

The source's historical mixed-COT limitation is unchanged. No2024 inference,
TEST access, actor update, encoder update, STL or live deployment. Frozen
encoders have seen full TRAIN, so these are internal representation diagnostics.

## Completed Results

Eight fits completed in632.93 seconds, service exit0, runtime `a9531198`.
Identity: `a0b5051bc365d3b57dd5132ae7d95d7288aa59ed0af08bdbf8a8e59fbe55b949`.
17 focused tests passed. Independent CPU verification authenticated10 seals,
16 endpoint states, effective optimizer steps/LR/decay, original initialization,
teacher hashes, nested supports and original cached embeddings. GPU cold
reload reproduced all16 endpoint metric records exactly. No clipped updates.

The four original controls replayed exactly at both initial and final states.
1470 new real TRAIN windows were cached; the combined cache has1709 rows.
FIT counts are196/784 forA and392/1568 forB. Teacher thresholds remain those
of the original49/98 FIT samples. No held-panel row was added to FIT.

Means below give equal weight to each of the four fold/seed pairs. Regret is
the known teacher's best action value minus the value of the selected action;
it is not dollars, Sharpe or measured trading return. Lower is better.

| FIT coverage | Updates | Audit value RMSE | Audit action regret | Paired passes |
| --- | ---: | ---: | ---: | ---: |
| Original49/98 |8192|1.522872|0.406730|reference|
|4x |8192|1.084347|0.152615|4/4|
|16x |8192|1.140319|0.232579|4/4|
|4x, equal presentations/example |32768|0.650348|0.056149|4/4|
|16x, equal presentations/example |131072|0.463141|0.056052|4/4|

The primary4x contrast reduces mean value RMSE28.80% and regret62.48% without
extra optimizer updates. Both expansions pass the prespecified paired rule at
both endpoints. This identifies sparse FIT coverage as a contributing problem
in the known-function diagnostic; the head architecture is capable of learning
more of that function without any change in feature count or encoder weights.

More examples are not an automatic optimum: at8192 updates,4x beats16x in
both metrics in all four pairs. With equal presentations/example,16x improves
value RMSE versus4x in all four pairs, but improves regret in only two. Mean
regret is essentially tied. The longer runs also change optimization time;
do not attribute their entire improvement to coverage. TheB stop-block curves
retain late degradation even while audit error declines. No checkpoint was
selected by inspecting either curve, and this is not a guarantee against drift.

## Next Controlled Step

Collect additional native counterfactual labels from existing FIT years before
changing the architecture or loss. Start with4x coverage, not an automatic16x
budget escalation. Keep original stop/audit anchors and teacher-derived samples
out of the real-target fit; reuse existing native labels wherever valid. Freeze
the real-target estimand, continuation policy, weighting and sampling rule before
generation. Match source weights, initial heads and optimizer exposure to the
native control; include constant and source-actor decision references. No actor
update or new floating pass until real-label calibration is qualified. The known
teacher is a diagnostic target only, never an observation or live signal.

This native-label contrast is the next action, not a running job. Source299,
71 inputs and49 cumulative phase2 return candidates are preserved; STL remains
paused. We have not exhausted weight-space analysis or certified the entire SAC
training process. No new trading account result is claimed here.

Full16-cell table, machine-readable results, replay evidence and training plot:
[coverage evidence](generalization/teacher_coverage_20261113/RESULTS.md).

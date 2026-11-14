# Native Counterfactual FIT Coverage

Owner authorizes execution after323. Freeze this design before new labels.
Source299 and71 inputs unchanged; no STL, state feedback, synthetic targets,
actor/encoder fitting,2024 evaluation or TEST access.

## Contract

Reuse323's label-blind4x nested supports, not16x: foldA49->196 FIT anchors
in2020; foldB98->392 in2020-2021.294 unique additional real windows. Preserve
the original239 anchors, embeddings and stop/audit masks exactly. Exclude every
other original panel anchor from new FIT. Do not select rows by outcomes.

At each additional row, replay source299's greedy policy from the original
2020-2023 calibration interval to recover its native account state. Fork that
same state three times, force NOP/BUY/SELL respectively, then follow the same
frozen policy until the original annual boundary. Target is discounted native
reward_return with gamma0.99, including native costs and forced terminal close,
as in318/322. This is not an oracle trade, a best future policy or a new reward.
Record actual position and action legality for diagnosis, never as model input.
Check two original-year witnesses against sealed318 states and all three
outcomes. Reject missing rows, changed support, invalid horizons or changed
source state; never fill labels or silently substitute another date.

Reuse four322 real-target base controls at8192 updates; authenticate their
initial/final metric replay before fitting. Four new fits: A/B x seeds19/43,
native208->8->3 twin heads, same initial tensors, AdamW LR3e-5,WD.01,batch128,
MSE,clip norm10. Primary endpoint8192 updates; secondary32768 for equal
nominal presentations/example. No audit-selected checkpoint or early stopping.
Fixed endpoints isolate the original patience artifact. Uniform FIT sampling;
do not introduce class/position weighting during this coverage contrast.

## Acceptance And Automation

| Requirement | Evidence |
| --- | --- |
| Identical old support/cache/labels | Explicit tensor, row and target equality |
| Nested FIT; held rows excluded | Unit rejection tests and runtime membership checks |
| Same native target | Year-boundary tests, original-state/outcome witnesses, cash identity |
| No source mutation | Full TRAIN actor signal replay and source tensor/file checks |
| Controls not retrained | Four control seals and initial/final replay |
| Actual optimizer contract | Endpoint state LR/WD/step checks; cold metric replay |
| Resume without babysitting | Per-row labels, two CPU workers, sequential GPU fits, optimizer/RNG checkpoints |

Report audit value RMSE, action-gap RMSE, action regret, pair accuracy and
action counts, including open-position and eligible-flat subsets. Include
source-actor and FIT-mean references. A consistently helpful endpoint must
reduce aggregate audit regret and not worsen eligible-flat audit regret or
aggregate RMSE in all four fold/seed pairs. Report failures as well as passes.
These remain internal frozen-representation diagnostics, not account returns.
The actor is not promoted from a critic-only metric. Preserve source299 and49
cumulative phase2 return candidates. Retsu's source qualification is separate;
the inherited mixed-COT limitation is not repaired by this experiment.

## Next Gate

After execution, inspect real-label calibration and state/action errors before
deciding on one bounded actor-refinement contrast. Do not automatically repeat
unchanged floating selection or infer that known-teacher gains guarantee native
reward gains. DEAP/genome diagnostics remain after completion of phase2.

## Completed Execution

Runtime `b4c784e7`, identity
`1a6a64dfd71e2ada45f3c3e069521580108ce7db5a804692733d3be39885ebb8`.
The two CPU label workers and four sequential GPU fits completed in146.18s;
service exit0.18 focused tests passed.294 additional anchors produced882
three-action continuations, plus four repeated original-year witness checks
(two per worker, each with all three actions).533 total records were verified.
All original239 records and embeddings remained exact; eight original control
initial/final replays and eight new cold endpoint metric replays were exact.
Independent CPU review verified eight seals, eight optimizer/checkpoint
endpoints, temporal memberships, initialization and the decision rule.

New positions comprise203 eligible-flat,66 restricted-flat,23 short and2 long
states. This is source-policy visitation, not a balanced-position intervention.
Expanded FIT composition is146/34/9/7 forA and271/72/35/14 forB in the order
eligible-flat/restricted-flat/short/long. No weighting was silently introduced.
Zero gradient-clipped updates. Source actor and all71 input columns preserved.

### Results

Equal-weight means over four fold/seed pairs, on unchanged audit rows:

| Arm | Value RMSE | Gap RMSE | Action regret | Eligible-flat regret | Pair accuracy | Joint passes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Original FIT,8192 updates |1.099600|0.746141|0.236451|0.295203|0.516358|reference|
|4x FIT,8192 updates |0.888191|0.590720|0.249055|0.306374|0.467283|1/4|
|4x FIT,32768 updates |1.450898|1.274173|0.253084|0.316853|0.498800|0/4|

At equal updates, value RMSE improves in all four pairs, averaging-19.23%,
but action regret improves only inA/19 and averages+5.33%. Eligible-flat
regret improves in bothA seeds but worsens in bothB seeds. Thus fewer numeric
errors, including lower gap RMSE on average, do not establish better action
ordering. The joint decision rule does not pass consistently. Longer fitting
further lowers FIT regret but worsens aggregate audit value RMSE in all four
pairs versus their original controls; it is not a general recovery.

The source-actor reference uses one-hot scores solely to measure its selected
action and resulting regret. Any RMSE of those one-hot scores retained by the
generic metric helper is not a calibrated actor Q-value error and must not be
compared to critic value RMSE. No new account return was evaluated here.

### Decision And Next Action

Do not promote any critic or update the actor from these results. Keep299,
71 inputs and49 cumulative return candidates.323 demonstrated that the head
can learn a known function better with more coverage;324 shows that this alone
does not fix real action ordering. Neither result identifies all causes of the
historical end-to-end training instability or proves lack of market signal.

Next perform a frozen-checkpoint decision-error analysis: compare control and
expanded heads on the same rows, stratify by actual target action gaps and
native position/legality, attribute changed winners and harmful versus helpful
switches, and separate common-level error from ordering error. Use FIT for
any subsequent loss/threshold design; already inspected audit strata remain
descriptive. Do not select new thresholds or checkpoints from these audit rows.
This analysis is the next task, not another running training job. STL stays
paused and DEAP remains after phase2.

[Full paired table and evidence](generalization/native_fit_coverage_20261113/RESULTS.md).

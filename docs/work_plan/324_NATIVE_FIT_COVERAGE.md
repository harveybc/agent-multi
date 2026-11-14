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

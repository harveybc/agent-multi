# Action-Advantage Calibration Contrast

## Scope And Fixed Design

Owner authorized launch after finding241: determine whether learning action
differences improves the critic ordering used by the anchored actor. This is
a bounded diagnostic on the unchanged70 parent, not another selection pass.
Keep inputs, strategy, account rules, actor architecture and all source weights.
No state feedback, rolling scaling, TEST, or automatic model promotion.

Two replicas (19,43), each with two sequential arms on the primary RTX5090:

- `absolute`: reproduce the existing paired-greedy smooth-L1 return fit.
- `advantage`: smooth-L1 of all three unordered action differences, plus
  smooth-L1 of each twin's action-mean against the target action-mean.

The mean term fixes the otherwise unidentified Q offset. Independent twin
offsets matter when the actor takes the minimum across twins. Its coefficient
is fixed at1, not tuned against validation. The pairwise term has coefficient1
and retains reward units; no arbitrary binary win/loss labels or margin cutoff.
Each term averages over its elements. This changes the loss geometry and
gradient scale as well as its emphasis, not the model or optimizer budget.

Reuse authenticated same-state greedy-return panels for each replica, including
their original fit/monitor assignment. Counterfactuals follow the parent policy
after the forced first action. Only fit anchors train the critics. Monitor
anchors are excluded from fitting, but overlapping trajectories mean they are
not independent market holdouts; disclose this in every result.

Both arms start at the same authenticated fresh calibrated critic checkpoint,
with original actor, cleared optimizer state, identical seed and minibatches.
Critic heads:4096 updates, batch128, original learning rate, frozen encoders,
actor and alpha. Log ranking/regret/RMSE at0,64,256,1024,2048,4096.
Then freeze critics and update actor heads256 times using the same full-TRAIN
replay and frozen-parent KL(beta1) from238. Report native TRAIN and VALIDATION
at32 and256 updates, plus the unchanged parent. No early-stop selection.
Validation is reporting only and cannot enter fitting, schedules or thresholds.
Include full trading statistics and local as well as mean KL. Fixed256 is the
primary endpoint;32 diagnoses the learning path and is not a new winner search.

## Requirements And Tests Before Implementation

| ID | Requirement | Evidence/test |
|---|---|---|
| A1 | Compare actual action gaps, preserve common Q level | Manual three-pair loss, offset and permutation tests; finite gradient |
| A2 | Same data, seeds and update budgets | Authenticated panel, cohort/timestamp checks; archived schedule digests |
| A3 | No monitor or validation fitting | Fit-index membership and TRAIN-only cohort assertions; malformed cohort rejects |
| A4 | Changes reach real critic parameters only | Real CUDA canary; changed critic-head tensors and frozen-state checks |
| A5 | Reproduce the absolute control | Exact calibrated parameter parity and native anchored256 account parity |
| A6 | Preserve source policy and accounting | Epoch-zero logits/accounts match; source hash unchanged; native evaluator |
| A7 | Autonomous sequential execution | Persistent service, per-arm receipt/resume, failure marker, automatic collector |

Normal flow: authenticate -> reconstruct replay once -> paired critic fits ->
anchored actor refinements -> CSV/PNG/JSON report -> seal -> collect and verify.
On a mismatch, fail with explicit evidence; never silently skip an arm or
substitute a cached result from another identity. Completed arms resume only
after receipt verification. Interrupted current arms restart from the same
fixed checkpoint, not a partial optimizer state.

## Interpretation

Require paired ranking/regret and account comparisons, including activity,
drawdown, buy/sell counts and harmful closure diagnostics in subsequent review.
Do not equate lower calibration loss with better trading or infer that shared
actor interference or position ambiguity has been solved. Preserve all arms
and report inconsistent replicas. This experiment introduces no new signals.

# Frozen-mask execution before DEAP

Owner-approved reorder, 2026-11-02. Supersedes the launch order in 187-190.

## Execution

- Stop and disable the DEAP coordinator and its closure-wait timer. Do not
  dispatch the remaining initial-population candidates. Preserve all results.
- Allow the two already running batch128 fits to finish independently. They
  do not replace checkpoints during a mask search.
- Runtime `5b65c8d1`, identity
  `13961f7797bff4c1f960f7df8ecbb0136ddb54453806111a569e6bab07773e5f`.
- Persistent single-writer service, bounded memory, cache-based resumption.
  Inference uses the available secondary GPU; two seeds are evaluated in turn.
  Scientific identity/parity failures stop explicitly, not infinite retries.
- `feature_selector` owns `floating_backward`; the native adapter is
  `tools/run_frozen_mask_selection.py`. This does not fabricate new causal
  stage evidence or weaken the fresh-training `rl_model` evaluator contract.

## Frozen comparison

120 features, 31 groups, 13 observations, stride 2. No dates, observations,
window positions, costs or business restrictions removed. Original full TRAIN
and INNER_A accounts are evaluated for every new subset. Atomic sin/cos pairs
and regime probability triplets yield 111 removable blocks.

Use full-input pilot checkpoints selected by the same TRAIN-best rule for
seeds 7/19 over the same 524288-decision training budget. This was the paired
sensitivity baseline, not a claimed DEAP winner. Every removed standardized
channel is zero across its entire window; weights and tensor dimensions remain
fixed. No training, replay updates or patience checks per mask.

Rank by paired mean INNER_A return/week, gap as tie-breaker; show TRAIN
separately and require >2 orders/week in both periods for both seeds. Tolerance
1e-5, safety budget 16384 unique subsets. Conditional reinclusion is enabled.
A budget termination remains incomplete; no global-optimum claim.

## Verification and timing

31 focused tests pass: native action/cash/close parity, whole-window masks,
unchanged support, cache resume/tampering, structural projection regressions,
and atomic encoding blocks. Independent real preflight authenticated both
checkpoint receipts, replayed archived accounts exactly, and checked new FP32
GPU emissions against CPU inference. Reload on the pinned device is exact.

cuDNN TF32 differed across devices. Disabling it changed one archived signal
for seed7 and none for seed19; importantly, both full account trading results
remain unchanged. Historical metrics are nevertheless not reused as the new
control. CPU/GPU maximum logit differences were 5.91e-5 and 6.46e-5. These
numerical checks do not introduce clipping or change the tie-breaking rule.

At 18:31 UTC: six new paired masks complete, median **9.18 seconds/subset**:
about 4.0 seconds inference and 5.2 seconds native accounts, including close
counterfactual diagnostics. First deletion neighborhood: approximately 18-22
minutes at this rate. Full floating duration depends on removals/reinclusions;
the safety-budget ceiling is about 42 hours at the initial rate, not an ETA.

The original full TRAIN replay took 4.24 seconds under cProfile. This does not
justify attributing hours of fitting to the account loop. Checkpoint monitoring
at different weights is not duplicate champion evaluation, and rewards gathered
while weights change are not the performance of one frozen checkpoint. This
speedup comes primarily from eliminating per-mask fits and exact-result reuse,
not shrinking the scientific workload or claiming training became faster.

## Next

Complete the autonomous mask search and publish paired deltas/trace. Then
confirm a small set of structural finalists with fresh fits against the full
control. Only afterward resume hyperparameter DEAP on the selected inputs.
Mask ranking is not a final structural feature selection; internal validation
is development data, and 2024/TEST remain closed.

# Capacity Contrast: 70x Hypothesis

## Question and Scope

The owner requests an empirical test of a proposed capacity deficit derived
from segment SNR and a model-size analysis. That analysis has been requested;
its derivation and definition of capacity are not yet available to this runner.
For this experiment only, operationalize capacity as unique trainable actor plus
two online critic parameters (plus the learned temperature), not target copies,
optimizer slots, representable bits or a proven number of memorized patterns.
This experiment does not certify that SNR implies a required parameter count.

Reference: the current attention-only model with93 selected features,30 groups,
13 observations at stride2. The new multiscale experiment remains separate and
continues unchanged. Do not combine capacity, new context and regularization.

## Frozen Design

- Widen the existing branch, core, head and attention embedding/feedforward
  dimensions proportionally. Preserve block count, kernels, dilations, receptive
  field, patch length, attention heads, positional encoding and categorical output.
- Select the smallest common even width with an actual parameter ratio >=70
  and <=75. Width selection uses architecture counts only, never performance.
- Actor and both critics grow together; entropy-temperature count is included
  once. Report target copies separately. Require every optimizer parameter to
  correspond to an actual trainable loss parameter, without duplicates.
- Preserve all input files, TRAIN scaler, calendar, costs, trade rules,93 columns,
  row support, fixed13x2 window, replay65536, batch128, optimizer settings, update
  cadence and early stopping. No state feedback or rolling normalization.
- Full TRAIN2020-2023 and VALIDATION2024; no TEST access. Fresh weights/replay.
  Max2000 epochs, patience20, delta1e-8, validation return/week; checkpoint
  selection starts at epoch1, patience counting after20.

## Execution and Decision

1. CPU architecture/count/contract tests, then a real full-batch CUDA canary and
   update/reload tests before pilot training. Never shrink batch or input to fit.
2. Pilot seed7 on the primary GPU once G2 has moved to the larger-RAM worker.
   Compare to the existing
   attention-only seed7 history under the matched monitor, without rerunning it.
   This uses the original control's GPU type. If memory protection moves the
   pilot to the secondary GPU, label the pilot comparison cross-device and
   exploratory, not a clean same-device causal estimate of widening.
3. If activity>2orders/week in both TRAIN/VAL and VAL return/week improves by
   >1e-8, automatically run capacity seeds19/31 and corresponding R0 seeds19/31
   on the same worker as the pilot. Reuse the exact R0 definitions already
   prepared by G2. On a shared worker, the coordinators share identities and
   services, so no duplicate fits. Controls on different GPU types are explicitly
   separate device-matched runs, not claimed byte-identical experiments.
4. All workers use the same host-local GPU lock as G2. At most one training
   worker per GPU. No changes to the running G2 job or its frozen queue.
5. Collect native TRAIN/VAL performance, gaps, Sharpe, drawdown, long/short order
   counts and wins, close reasons, learning curves, parameter counts, memory
   and epoch time. More TRAIN profit alone is not a generalization improvement.

## Test Design Before Implementation

| ID | Requirement | Test / Evidence |
|---|---|---|
| C01 | >=70x actual trainable parameters | Meta count versus instantiated SAC count; smallest admissible even width |
| C02 | Only widths change | Reject unsupported configuration; deep comparison of invariant architecture/settings |
| C03 | Same inputs and behavior contract | Source authentication, file digests, support hash, native preflight reuse |
| C04 | No dormant or unoptimized extra width | Optimizer coverage, nonzero branch/core/head/attention gradients and weight updates |
| C05 | Causality and output codec unchanged | Prefix perturbation, finite categorical logits/Qs with3 outputs |
| C06 | Real training feasible | Full batch128 real-data CUDA canary; record peak memory and loss |
| C07 | Recoverable usable weights | Same-device bundle reload plus existing epoch recovery regression |
| C08 | No concurrent workers or duplicate R0 | Shared flock, existing pinned R0 services and authenticated receipts |

Failures in code, identities or resource acceptance stop that arm with an explicit
error; they do not silently change the experiment. The existing G2 queue is
independent. Synthetic generation is not part of this contrast.

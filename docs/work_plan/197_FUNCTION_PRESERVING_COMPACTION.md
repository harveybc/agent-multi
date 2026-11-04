# Frozen weight-space compaction diagnostic

## Purpose and position in the work plan

Owner authorized on 2026-11-04. After selection195 and fresh fit196, before
another selection/training campaign, test whether the masked108 actor can be
represented with94 inputs while preserving its learned function. Inspiration:
[weight-space programming](https://rikiyat.github.io/blog/weight-space-programming/).
This is a controlled algebraic transformation, not a claim that the article
demonstrated trading generalization or that weight editing replaces training.

The masked actor returned +10.447% in development VALIDATION, whereas the
fresh94 fit returned -1.211%. Changing architecture and training from scratch
changes many weights at once. The gap alone does not identify its cause.

## Predeclared arms and invariants

1. Original108 actor with the14 removed channels set to zero AFTER the fixed
   TRAIN scaler: authoritative masked control, no parameter updates.
2. Packing control:94 observations lifted into the original108 layout with
   zeros and the unchanged actor. Must be bitwise identical; NOT compaction.
3. Compact actor: slice first-convolution input weights for surviving channels;
   preserve every subsequent weight and original fusion slot. An empty branch
   is replaced with its full deterministic zero-input temporal response as a
   frozen buffer. Do not replace it with zero or with its temporal mean.
4. Diagnostic intervention: zero that response in a separate compact copy.
   Measure effect; do not tune it or select an amplitude from VALIDATION.
5. Previously completed fresh94 actor, unchanged checkpoint30.

Same full TRAIN2020-2023 and VALIDATION2024,13 observations at stride2,
original valid rows, fixed scaling, price/calendar/risk contract, native costs
and account initialization. No TEST reading, learning, DEAP, state feedback,
new transforms, dropout or interpolation. VALIDATION remains exposed development.

## Test design before implementation

| Requirement | Test / evidence | Expected outcome |
|---|---|---|
| W01 Effective weights | Modify functional SAC actor parameters then extract | Copy matches surfaces; no stale standalone weights |
| W02 Packing | Ordered subset lifted into masked input | Bitwise identical logits |
| W03 Physical pruning | Partial and fully removed synthetic branches | Fewer parameters;94 real inputs; fusion order unchanged |
| W04 Bias response | Empty causal branch with nonzero temporal response | Full time profile preserved; zero ablation differs |
| W05 Causality | Perturb suffix in retained observations | Earlier encoder positions unchanged |
| W06 Isolation | Hash/compare source before and after all interventions | Original model unchanged |
| W07 Artifact | Save/reload and tamper with saved weights | Exact reload; hash mismatch rejected; inference-only schema |
| W08 Account parity | Full paired native ledgers and equity | Same actions/trades/equity for compact and packing controls |
| W09 Scope | Authenticate source receipts and support | No changed rows/scales/costs/TEST access |

Floating-point sums can change after reducing convolution width. Declare
atol=1e-5 and rtol=1e-5 before running; report max error and bitwise equality
separately. Tolerance is NOT permission to change actions or trading accounts.
If actions differ, compact deployment fails qualification; preserve the exact
packing control and publish the mismatch. Do not relax the gate after results.

## Execution and artifacts

CPU unit/integration tests first, then one sequential inference-only GPU job.
Authenticate both fitted bundles and selection receipt; replay archived controls;
evaluate all arms and capture branch/core/head activation diagnostics. Persist
inference bundle with projected scaler/column order, source fusion layout, weights
and constant buffers, source hashes and reload proof. No critic, optimizer or
replay migration is claimed. Full account evidence includes BUY/SELL, wins,
Sharpe, return, drawdown, orders/week and early-close counterfactuals.

Results determine the next intervention. Compaction preserving the +10.447%
account would preserve an existing selected policy, not establish fresh
out-of-sample performance or prove the entire training gap came from one branch.

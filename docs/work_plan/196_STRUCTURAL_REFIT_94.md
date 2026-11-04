# Fresh structural94 SAC confirmation

Owner authorizes the next structural fit after the second ordered mask round.
Selection195 closed automatically at2026-11-04T06:17:33Z:108 ->93 ->94 columns,
31 ->30 groups,113 unique evaluations, no exact-tie removals, one reinsertion.
Result sha256 `ee172eb0f2f5e692196129cd457889579840dca211f6067d81d495f2dce3e065`.
Masked validation return improved -3.4537431% ->10.4470635%, Sharpe1.44955,
DD4.72764%,298 trades. This is development selection, not independent confirmation.

## Fixed comparison

- Fresh seed7 policy, critics, optimizers and replay. Do not transplant the
 108-input weights or initialize from their masked outputs.
-94 actual channels,30 surviving groups. Remove the empty branch and remap
 fusion inputs. Keep branch width8/two residual TCN blocks, core width16/two
 blocks and head width8; three categorical outputs unchanged.
- Same13 observations at stride2, same native support and calendar, original
 fixed TRAIN constants projected to the94 columns. No new scaler fitting,
 clipping, imputation, agent-state feedback or rolling normalization.
- Full TRAIN2020-2023 per epoch, full VALIDATION2024 evaluated after each epoch;
 TEST2025 closed. Same costs, business rules and account initialization.
- Same batch128, learning rate0.0003, weight decay0.01, gamma0.99, tau0.005,
 replay65536, initial learning warmup4096 and update every4 decisions.
- Up to2000 epochs. Monitor validation net return/week only; patience20,
 min_delta1e-8. Epochs1-19 cannot select a checkpoint; first reference20.
 TRAIN metrics and the three shadow monitors remain recorded.
- Persist best inference bundles and complete epoch-boundary recovery, then
 restore the best checkpoint and export native accounts and close diagnostics.

## Implementation and gates

`run_structural_sac_refit.py prepare-selected` authenticates the completed mask
receipt, its source/runtime lineage and parent fit; derives a structurally reduced
recipe and never copies learning/replay/checkpoint files. The new worker uses the
existing tested epoch loop and fresh model constructor. A pinned runtime and
persistent bounded-restart service run on the primary GPU; a collector exports
receipts and inference bundles automatically. No additional search is launched.

Acceptance: altered/incomplete receipt rejected; partial coding blocks rejected;
column order and scaler projection exact; empty branch removed;48/12 months and
all original decision rows preserved; native synthetic controls pass; fresh model
shape/parameter count measured; epoch resume and inference reload remain exact.
This is a single-seed structural test, not evidence of robustness across seeds.

## Verified execution

Runtime `1431d492`,47 focused tests pass. Source receipts, original48 TRAIN
months and12 VALIDATION months,24025/6207 actionable decisions and support digest
are unchanged. Native synthetic controls pass. Actual input shape `[13,94]`,
30 groups and96442 parameters (previous108-input model99586).
Identity `410d64cc36705f5dbe7fb99c896fb88629dab9585aa0b7e6f7f1e16149717a0e`.
Persistent primary-GPU worker started2026-11-04T07:38:43Z; first gradient update
observed after the4096-transition replay warmup. Independent collector enabled.
No prior weights or recovery state were copied. Runtime settings and code are
pinned; reporting changes do not modify the running source tree.

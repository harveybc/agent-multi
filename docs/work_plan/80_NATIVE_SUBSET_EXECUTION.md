# Native temporal PPO subset execution

Owner authorizes integration and autonomous sequential GPU execution, including
the paired cost diagnostic, without intermediate approvals. Use new run roots;
leave other workers, sealed recipes and the expert-guided lane unchanged.

## Tests before deployment

| Requirement | Evidence |
|---|---|
| Structural subset | surviving column order/constants exact, empty branch removed and restored, fusion width recalculated |
| Paired support | select columns from the full frozen tensor, preserve timestamps/validity even if a removed feature caused a hole |
| Train-only scaling | refit constants on the declared TRAIN interval, never internal validation; future perturbation leaves them unchanged |
| Small temporal core | residual causal TCN, two blocks, two kernel-3 convolutions each, dilations 1 and 2; core RF13; prefix invariant |
| Actual policy | categorical PPO actor/value heads, fresh model/optimizer/rollout, native business simulator, no expert or feedback channels |
| Reload/identity | save/reload policy tensors and predictions equal; request/model/data/support/cost provenance verified |
| Recovery | exclusive queue lock, immutable results; bounded fresh retries, no reuse of partial weights |
| Cost contrast | fresh configured and zero-cost policies with three paired seeds; both evaluated under both cost modes |

Sequence: synthetic CPU integration -> real-data no-fit preflight -> GPU
engineering admission -> paired full-input cost diagnostic -> bounded native
backward floating search using feature_selector, with fresh fits and retained
per-subset receipts -> confirmation/report. The core is fixed within subset
search; there is no simultaneous architecture tuning. Keep the earlier shallow
CNN as an explicitly separate control, not a purported identical architecture.

Preserve all existing upstream evidence and its limitations. Historical proxy
selection/profile evidence is not new RL utility, and internal 2023 includes
upstream selection decisions. Current search is DEVELOPMENT, not independent
confirmation or TEST. Do not fabricate missing stage measurements or certify
unidentified causal effects. An interrupted/budget-limited search is provisional.

Feature_selector owns the generic floating algorithm; agent-multi owns the
native evaluator and execution. No new search algorithm here. Use coherent
sine/cosine blocks, not an implicit irreversible family veto. Report actual
parameters, orders/direction/wins, early closes, costs, profit, Sharpe, drawdown
and TRAIN-validation gap per seed. Undefined scores are not zero.

First bounded search budget: 128 distinct subsets x three seeds; full-input
controls and later confirmations additional. Each fit has 102400 environment
decisions, as the previous actual-policy control. This first pass can end before
reaching 20 features; it must say BUDGET, not pretend selection is complete or
that untested candidates failed. Use its measured cost for the next search
allocation. Preserve byte-authenticated completed evaluations across restarts.

## Implementation verification

Twelve focused CPU tests pass, including actual PPO updates, exact checkpoint
reload, corrected native rollover rewards, causal prefix/oldest-window witnesses,
paired support, train-only constants, and bounded recovery/authentication. The
real-data preflight admits 120 columns in 31 branches, with 21,221 TRAIN and 2,805
INNER_A valid origins. These counts are input support, not completed training.

Controls are TCN/configured, TCN/zero-cost and shallow-CNN/configured, each with
seeds 7, 19 and 43. Every control is replayed under both cost modes; only
TCN/configured results feed the wrapper. The no-cost arm cannot be promoted as a
cost-aware winner. PPO uses no dropout during likelihood calculation, as in the
existing policy; inherited weight decay remains fixed and recorded. No encoder
or scaling choice is silently tuned inside the feature search.

The search is an additional native wrapper phase, not a substitute for the
existing causal/profile/encoding stages. Completion of this bounded pass does
not issue a global release or certify independent generalization. Undefined
Sharpe or insufficient activity cannot win selection.

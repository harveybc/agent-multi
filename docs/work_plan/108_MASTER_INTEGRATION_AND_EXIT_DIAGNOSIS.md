# Master integration and optimal-exit diagnosis

Date: 2026-10-28. Scope: source integration and CPU-only inspection of sealed
synthetic results. No financial fitting, TEST access or campaign redeployment.

## Current result

The full positioned grid closed at 2026-10-28 00:49:21 UTC and its auditor at
00:50:45 UTC (19:50 on October 27 in America/Bogota): 874800 cells, zero
reported failures; 463792 ideal-output early closes, 233341 long and 230451
short. All identified opportunities were captured under that reference.
The early-close share was 35.2515%, a description rather than a target.
These are oracle-driven engine/contract checks, not learned-policy returns.

The separate synthetic study closed nine DEAP searches and six PPO fits.
All six failed deterministic learned-control qualification. In confirmation,
each produced zero discretionary early closes, mean net -17.937039 USD per
seeded position and mean exit regret 1.355925 times initial SL cash risk.
This is neither annual validation performance nor a percentage account loss.
No learned champion or financial selection is admitted.

## New read-only findings

Tool: `tools/audit_optimal_exit_outcomes.py`. Evidence:
[REPORT.json](../audits/evidence/optimal_exit_policy_20261028/REPORT.json).
The audit authenticates every consumed case, array, result and checkpoint
against the study's retained CLOSE manifest. It never changes those artifacts.

### A1: sampled actions and deterministic evaluation are different policies

The real learner is categorical PPO, not SAC. Training samples commands;
the published qualification chooses the highest-logit legal command.
On 4274 TRAIN and 2566 confirmation decision rows per checkpoint, all six
argmax policies choose HOLD. CLOSE probabilities nevertheless range from
about 0.148 to 0.265 across models/rows. Therefore CLOSE is reachable under
sampling; saying the models literally never assign it probability is false.
Actor and critic Tanh saturation (absolute output > 0.99) is zero in this probe.
This localizes the deterministic failure; it does not establish its learning
root cause or justify silently changing the deployed evaluation policy.

For a memoryless policy with close probabilities p_i, the exact probability
of exiting at i is p_i * product(1-p_j, j<i). The residual survival mass goes
to the actual automatic terminal outcome. The report uses this sum, not a
favorable random rollout, to measure the sampled policy's expected outcome.
Confirmation sampled expectations have 98.53%-99.79% early closure and
mean net +1.11 to +1.37 USD per synthetic position. Those are expectations
over the same artificial corpus, not newly accepted trading results.

### A2: a constant-hazard control nearly reproduces the learned behavior

For each model, average its CLOSE probability over TRAIN decision rows
separately by direction; freeze those two constants and apply them to every
confirmation row. No confirmation outcome is used to set the constants.

| Reward / seed | Network expected regret | Constant expected regret |
|---|---:|---:|
| dense / 7 | 0.24959687 | 0.24963590 |
| dense / 19 | 0.25025126 | 0.25024092 |
| dense / 43 | 0.25908175 | 0.25921650 |
| terminal / 7 | 0.25692367 | 0.25692666 |
| terminal / 19 | 0.25040312 | 0.25036423 |
| terminal / 43 | 0.26443872 | 0.26485114 |

Regret is normalized by initial risk, lower is better. The largest absolute
difference is below 0.000413. No significance claim is made. This is evidence
of very weak useful state dependence on this corpus; positive sampled P&L
alone would not prove learned optimal timing. The six failed qualification
verdicts remain unchanged.

### A3: the last DEAP is not whole-strategy optimization

Its only genes are long/short CLOSE-logit biases in [-2, 2], applied to
oracle-derived ideal/noisy/delayed tapes. It does not optimize SL/TP, volume,
entries or the learned network. Nine searches (population20, generation cap100,
patience10, relative tolerance1e-8) stopped at generations13-17 by patience.
All nine have identical confirmation aggregate metrics: normalized regret
0.01284964, unannualized episode Sharpe1.621611 and mean return fraction
0.0005502223. The zero-bias control has regret0.02348338. Several finalists hit
the search bounds. No unique best coefficients or fully optimized strategy
are established, and no confidence interval was computed for this contrast.

The `sharpe_return` fitness adds a return fraction of order 0.0005 to Sharpe
of order 1.6 without equalizing scales. It is not a meaningful balanced test
of both factors. A future comparison must freeze appropriate scaling or a
multi-objective rule before new confirmation; do not reinterpret these runs.

### A4: reporting and architecture gaps

The old grid auditor's `scope` string still mentions finite-horizon reference,
although the deployed positioned mode uses the full remaining position menu.
This stale label must not be used to infer the target definition. Preserve the
closed file and attach corrected scope in a new report/version.

The study's PROGRESS heartbeat and plan state still described training/running
after closure. STATUS/CLOSE were authoritative. This update corrects the
current plan without rewriting historical evidence. Old dated updates remain
historical, not launch orders.

The strategy README now defines layers, three architecture views, semantic
actions and the separate entry/exit optimum. Existing legacy helpers still
include buy-favoring ties and non-finite filtering; the README explicitly does
not certify those as generic behavior. No evidence connects those legacy
helpers to the failure of this separate categorical exit-control runner.

## Next actions, in order

1. Instrument actual PPO rollouts: sampled/executed commands, close/hold
   advantages, policy/value losses and per-branch/head gradient norms. Verify
   the command and likelihood contract under an optimizer update.
2. Establish observation sufficiency and encoder capacity on the same synthetic
   corpus, without adding order-state feedback, clipping or rolling scaling.
   Separate a supervised diagnostic (not the pure-RL result) from RL credit
   assignment. Compare against the fixed-hazard baseline above.
3. Freeze an explicit execution/evaluation policy and test its temporal
   sensitivity with positive/negative controls. No post-hoc threshold change
   on confirmation, or reward/width/decoder changes bundled into one arm.
4. Only after learned controls pass, optimize the declared full strategy
   parameters under the synthetic contract, obtain cross-audit and resume
   model-specific floating feature selection.

No successor training is running after this inspection. There is no measured
ETA for completing the learning repair. These are actionable diagnostics,
not a conclusion that financial inputs lack predictive information.

## Integration and verification

Integrated committed research branches using ordinary merges in separate
worktrees. No force push, historical rewrite, runtime checkout alteration or
commit of unrelated untracked files. In predictor, the single documentation
conflict preserves both the source-coverage addendum and later operating notes.
Feature selector's existing `main` is preserved and also published as `master`.
Research code on master is not a strategy release; frozen producer pins stay
unchanged.

Focused verification before publication: heuristic-strategy58 passed;
predictor42 passed; feature_selector38 passed/4 skipped (legacy pinned
integration checks); agent-multi91 passed including native quote/corpus
integration against the merged engine; gym-fx118 passed; financial-data60
passed across availability/feed and curation tests; feature-eng82 passed/1
skipped. The latest phase-one feature-eng branch was already on master; its
separate TRAIN-only metrics/calendar changes have also been merged and tested.
These are targeted checks, not a
full-suite certification of every historical research branch.

### Published integration receipts

All entries below were pushed normally to remote `master`, not merely left
on integration branches. The agent-multi row names the audit-code commit;
this documentation receipt is a subsequent descendant. Integration preserves
existing master work and committed source histories, including superseded
research reports with their existing qualifications. It does not promote
every archived experiment or unfinished local edit into an accepted result.

| Repository | Remote master integration commit |
|---|---|
| heuristic-strategy | b6c20debd30b960c33a7b210453ced1f890daf74 |
| agent-multi | 42478d194e58085a9856b04793b8b20f3a715b86 |
| predictor | 372c87920fbd0e0ee8aaa3e380551e29d86297bb |
| feature_selector | bec66cca0ab3367a856ec1d70c8783c9919d8704 |
| gym-fx | 92da3cd621a67e98a76344e0f807eb3ffdeed7ee |
| financial-data | 974b29e3f (curation plus availability/feed line) |
| feature-eng | 61aacf14a8a318f28d159a2bbc09acfbdc90c742 |

The financial-data documentation conflicts preserve current master wording;
the inventory merge retains the explicit retrospective-archive delivery path
with UNKNOWN availability and no date-range claims, exercised by its tests.
The agent-multi pre-push sensitivity gate scanned125 outgoing commits CLEAN;
financial-data's publication scans also reported no leaks. Unrelated dirty
working trees and the existing interrupted predictor merge were not altered.

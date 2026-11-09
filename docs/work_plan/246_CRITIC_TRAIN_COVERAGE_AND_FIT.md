# Critic TRAIN Coverage And Small-Set Fit

## Scope

Owner authorized steps1/2 after245. Audit all authenticated TRAIN anchors in
r19/r43; then test fitting32 distinct permitted flat-account states per replica
(8 per year2020-2023). No actor update, account strategy change, additional
observation, feature selection, validation-based fitting or TEST access.

The small-set experiment is deliberately a memorization test, not evidence of
generalization or improved trading. All input columns and the original window
remain unchanged. It does not replace training on the complete market dataset.

## Stage1: Data And Target Audit

Authenticate both panel shards and their original greedy counterfactuals.
Classify each anchor as flat/both entries permitted, flat/restricted, open-long
or open-short using actual native first-action effects, not action masks.
Report counts by role/year/state, refusal reasons, tied action pairs, target
quantiles, target gaps and original actor-path cash reconciliation.
Check greedy soft-return equals reward-return (zero entropy in these labels),
positive durations, finite labels, immutable parent states and TRAIN timestamps.
For open positions verify NOP and same-direction action give identical native
continuations; report rather than discard equivalent decisions. Inspect exact
duplicate input windows with conflicting labels without introducing state input.

## Stage2: Fixed Small-Set Diagnostic

Select8 eligible fit anchors per year using a fixed seeded permutation, never
choosing by model performance or validation. Both BUY and SELL must actually
open, and at least one action-return difference must be nonzero. Refuse if
coverage is insufficient; never silently change the requested sample count.

Each replica starts from the same fresh calibrated critic checkpoint as242.
Use the existing pairwise-plus-mean loss, all32 states per update,4096 updates,
native learning rate/weight decay, cleared optimizer state and fixed seed.
Record loss, pairwise ranking, action-gap RMSE, common-level RMSE, regret,
gradient norm and changed tensors at0/64/256/1024/2048/4096. Measure each twin
and their minimum separately. Save final weights and plots automatically.

First run `heads`: only critic heads train. If it does not meet the fixed
diagnostic criterion, automatically run `critic_representation` from the SAME
initial checkpoint, seed, batch and update budget. Only online critic encoder
and head parameters train; actor, alpha and target networks stay frozen.
This is a change of trainability, not a larger architecture.

Criterion: all informative pair orderings correct (within numerical tie rule),
gap RMSE <=0.05 reward units, maximum regret <=1e-8 reward units, for BOTH twins
and their minimum. These fixed engineering tolerances are not trading gates.
Evaluate them at the fixed4096 endpoint, without choosing the best checkpoint.
Missing the criterion does not prove insufficient capacity: optimization budget,
learning rate, regularization and representation remain possible explanations.

No full market account reevaluation is needed: the actor is unchanged. Compare
all valid TRAIN logits exactly before/after and verify all frozen parameters,
with changed-parameter counts proving that intended tensors received updates.
Other TRAIN anchors are diagnostic only; never use them to select an endpoint.
No claim of an independent holdout from overlapping continuation labels.

## Tests And Execution

| ID | Requirement | Evidence |
|---|---|---|
| C1 | Native eligibility, finite labels, holds equivalent | Unit cases for allowed/rejected entries, unequal hold labels and nonfinite values |
| C2 | Deterministic32, fit-only, all4 years | Selection tests, missing-stratum refusal, no monitor selection |
| C3 | Real gradients only in intended critic scope | CPU trainability test and real CUDA canary for both scopes |
| C4 | Diagnostic metrics and conditional branch | Manual gap metric and fit-criterion tests; explicit skip record when passed |
| C5 | Immutable actor/source | Frozen-state checks and full TRAIN logit identity after each arm |
| C6 | Autonomous execution | Sequential persistent worker, sealed arm resume and automatic collection |

Audit precedes training. The primary RTX5090 runs at most four arms sequentially.
Each completed arm is authenticated before resume. Interrupted current arms
restart from their declared initial checkpoint; no partial optimizer recovery.
Archive failures and never promote these diagnostic critic checkpoints.

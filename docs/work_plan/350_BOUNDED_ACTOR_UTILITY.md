# 350: One Bounded Actor-Utility Trial

Protocol fixed before training. This executes349's finite exit plan, not a
new critic search. Two actors, seeds19/43, each starts from immutable299.
Critic source is the decision-selected349 family at16, fold B for both seeds:
B has392 expanded FIT rows versus A's196. This coverage rule ignores results.
Importantly B's349 checkpoints equal345: this trial tests the accumulated
critic correction, NOT the isolated effect of the two A-only local repairs.

## Fixed Contract

- Same71 inputs,28 groups, full historical support and native strategy.
  TRAIN2020-2023, VALIDATION2024, no TEST. No state feedback, new normalization,
  masks, feature removal, STL, architecture or risk-parameter changes.
- Transplant eight native twin-head tensors, online only, by exact names and
  shapes. No encoder, actor, target-network or temperature changes at import.
  Authenticate all parents, support/recipe, input rows and frozen embeddings.
  Verify standalone native-head versus full-model Q and original actor/account
  parity before the first update. Frozen target heads are not used by this
  actor-only experiment; it is not a full joint-SAC resume certificate.
- Actor objective is existing categorical SAC: E_pi[alpha log pi - min(Q1,Q2)].
  Use native stored alpha, no retuning or entropy sweep; stop gradients through
  Q and alpha. Gradients only for the four actor-head tensors, with all original
  trainability flags retained to keep attention dispatch unchanged. No other
  parameter receives a gradient or update. Fresh AdamW lr3e-5,wd0.01,batch128,
  gradient clip10. Minibatches cover only B expanded FIT, never stop/audit/VAL.
- Maximum8 accepted updates and64 attempted updates per seed. Existing
  bounded_functional_actor trust limits unchanged: step margin RMS0.002,
  source-to-current mean KL1e-4,max KL1e-3,relative parameter drift0.01;
  maximum24 backtracking scales. Accept nonzero motion and non-increasing
  minibatch objective. Rejected steps roll back weights AND AdamW memory.
  Eight accepted updates is the primary endpoint. If64 attempts cannot reach
  it, record INCOMPLETE_BUDGET, never substitute a best-looking earlier point.
- Save full model/optimizer/RNG and ledger after each attempt. On restart
  resume exact attempt and accepted counters. Replay first transaction from
  cold saved state. Evaluate native accounts at0/1/8 accepted steps;1 is only
  diagnostic, not eligible for selection. Evaluation does not reset patience
  or extend the budget. No early stopping or epoch search in this microtrial.
- Full checkpoint reload must reproduce inference and native accounts.
  Record TRAIN/VALIDATION returns, Sharpe, drawdown, activity, buy/sell, wins,
  early closes, costs and cash consistency; retain ledgers and training plots.

## Decision And Stop

Per seed, endpoint8 qualifies only if each full account has>=2 trades/week,
nonnegative return and return no lower than299, with a strict improvement in
at least one period. Cash checks and frozen-state/replay checks must pass.
Report both seeds separately; never average an ineligible seed into a pass.
If both qualify, deterministic rank is VALIDATION return, then TRAIN return,
then lower seed. This is development selection, not live promotion.

If a candidate qualifies, verify and export an inference bundle; otherwise
retain299 and CLOSE this critic/actor refinement line. Neither outcome launches
another critic variant. A later phase2 pass requires a qualified changed model;
its return pool includes all49 absent phase2 columns. STL remains paused.

|Requirement|Acceptance test|
|---|---|
|Correct twin mapping|Exact eight-tensor names/shapes and native forwards; malformed donor rejected atomically|
|Frozen components|Every non-actor-head parameter/buffer bit-identical after updates|
|TRAIN-only exposure|Support hash, timestamps, disjoint FIT/stop/audit; schedule membership|
|Bounded progress|8 accepted/64 attempts/24 scales; incomplete endpoint cannot qualify|
|Rollback|Rejected proposal restores weights and optimizer state|
|No silent baseline change|Original logits and complete299 accounts reproduced before updates|
|Correct decision rule|Tests for inactivity, worse TRAIN, worse VAL, ties and nonfinite values|
|Reproducibility|Cold first-transaction replay, full-state resume and full-account checkpoint reload|
|Finite exit|Terminal KEEP299 or QUALIFIED_EXPORT; no automatic diagnostic extension|

## Execution

Prelaunch:25 focused tests pass in3.33seconds; scoped Ruff checks pass.
Tests include exact native twin transplant, atomic rejection, only-actor
updates, full-state first-transaction replay and rollback with nonempty AdamW
memory. Live processes and source299 remain untouched. Runtime evidence follows.

The first launch stopped BEFORE any actor update: direct q_surfaces used a
different execution path from the cached critic producer, with
max Q discrepancy1.572e-5. The baseline account had already matched299.
The adapter now uses the producer's existing probe_q: functional twin vmap
with controlled attention dispatch, restored after each call, rather than
sequential materialized parameter views. This contrast does not isolate those
two implementation differences from one another. The original1e-5 Q gate and exact FIT
action comparison remain unchanged; no tolerance relaxation or weight fitting.
Failed r1 is retained, and the corrected launch receives a new identity/root.

## Completed Results

COMPLETE_INDEPENDENTLY_VERIFIED. Runtime `054862d5`, identity
`84ff192219fe96965b8dd1834fd6333b732acbdbf9afd63d1c92bf4365627e14`.
Corrected producer completed in23.45seconds, exit0, zero restarts. Each seed
accepted8 of8 attempts; this is a short head-only utility trial, NOT eight
full-data training epochs or a new end-to-end SAC training campaign.
All original71 inputs and full account periods were retained.
Updates used the sealed392 FIT anchors (197 from2020 and195 from2021), as did
the imported B critics. No2022/2023 or VALIDATION labels entered these updates.
The source model was pretrained over2020-2023; full TRAIN account evaluation
still covers all four years. This was not a four-year full-replay refit.

|Model / endpoint|TRAIN return|VALIDATION return|TRAIN / VAL trades|VAL Sharpe|Verdict|
|---|---:|---:|---:|---:|---|
|Original299|16.058224%|18.369946%|745 /207|2.690712|Retained|
|Seed19 /1|16.058224%|18.369946%|745 /207|2.690712|Diagnostic only|
|Seed19 /8|16.058224%|18.369946%|745 /207|2.690712|No strict improvement|
|Seed43 /1|15.759431%|18.369946%|745 /207|2.690712|Diagnostic only|
|Seed43 /8|15.186874%|18.172324%|746 /207|2.659815|Regression in both periods|

Seed19 changed weights/probabilities but reproduced the original economic
accounts exactly. Seed43 endpoint8 loses0.871350 percentage points TRAIN and
0.197622 points VALIDATION. Both remain active above2 orders/week. The original
has229 BUY/516 SELL trades in TRAIN and22/185 in VALIDATION, with58/6 early
closes. Seed43 endpoint8 has230/516 and22/185, with59/6 early closes. Detailed
win rates, drawdowns, fees and close reasons are in the result artifact.

Actual native critic versus independent cached NumPy Q discrepancy is
2.949e-6 for19 and3.003e-6 for43, below the UNCHANGED1e-5 gate; all FIT argmaxes
match. The encoder-cache discrepancy is9.54e-7, below1e-6. Imported critic
heads leave initial actor logits bit-identical; both original full accounts
match299 before any training. Native saved alpha is0.005312912631779909,
not the constructor's historical initialization value1. No scale/alpha tuning.

Both first transactions reproduce weights, AdamW state, RNG and receipts
exactly after cold restore. All four actor checkpoints reload with exact
producer logits and native accounts. A SEPARATE CPU verifier replays ten full
accounts (source TRAIN/VAL plus four checkpoints x two periods), checks actual
frozen tensors, saved primary endpoint, AdamW step counts and finite trust
budgets, and independently obtains the same terminal decision in35.14seconds.
32 focused tests and scoped Ruff checks pass. No TEST or live replacement.

## Terminal Decision

**KEEP299_CLOSE_REFINEMENT.** No candidate qualifies and no replacement bundle
is exported. Keep299,71 columns,28 groups and49 absent phase2 return candidates.
These results close the agreed critic/actor micro-refinement sequence. They
do not establish that the inputs lack predictive information, nor that no
other training method can work. They establish no qualifying gain for THIS
fixed eight-update transfer under its existing constraints.

No more critic variants, automatic budget extensions or unchanged floating
passes are queued. The owner subsequently corrected the next-task assignment:
[351](351_PHASE2_EXIT_AND_ARCHITECTURE_SEARCH.md) defines phase2 closure followed
by architecture/training optimization, not recent-data/live work. Satoshi owns
live trading. STL remains paused; DEAP follows the explicit phase2 exit gate.
Source checkpoint and live providers remain unchanged.

[Configuration](generalization/actor_utility_20261114/CONFIG.json),
[complete results](generalization/actor_utility_20261114/RESULT.json),
[independent verification](generalization/actor_utility_20261114/VERIFICATION.json),
[training/validation plot](generalization/actor_utility_20261114/TRAINING_VALIDATION.png).

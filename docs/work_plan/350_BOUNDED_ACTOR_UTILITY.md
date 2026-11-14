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

# Output audit transfer and NOP decision

Date: 2026-10-25. For the owner and the expert-guided lane maintainer.
Scope: code/document review of the existing decision, reward and arbitration
paths. This is NOT full-system certification or acceptance of the replacement
learner, which is still a design in plan 67.

## Evidence inspected

- This lane: `env_plugins/order_contract.py`, `env_plugins/recipe_native.py`,
  native engine producer `89512e8`, and the existing 16-case business witness.
- Expert-guided lane checkout `87104b652d4e6a5a545b006be37985c666c3bc32`:
  `tools/expert_guided_e3m.py` (`labels_for`, `Composed`, arbiter selection),
  `docs/short_horizon/clean_round/OUTPUT_CONTRACT_AUDIT_AND_DESIGN_2026_10_25.md`.
  This review does not assert that all inspected paths are currently deployed.

Executed in this review: existing `tools.order_contract_study.witness` on CPU,
16/16 cases passed, maximum absolute reward-versus-ledger difference
8.810729923425242e-13 USD. Cases cover long/short, close, daily quota, SL, TP,
ambiguous both-hit, gap, forced flat and legacy invalid-HOLD behavior. They use
gamma=1 for ledger equality and do not test the proposed categorical learner.

## Adopt and correct

1. Adopt categorical command semantics, bounded conditional risk parameters,
   immutable business masks and separate tuning/reporting periods. Tuning an
   arbiter on INNER is development optimization, not an independent estimate
   on that same INNER block. The selected-minus-median Sharpe statistic in the
   peer report is selection uplift, not a measured unbiased estimate of future
   optimism or proof of identical bias for every arm.

2. Reject the claim that no standalone NOP classifier means nobody learns
   abstention. `labels_for` assigns both opening classifiers negative labels
   on HOLD examples. Their joint arbitration still needs checking, but the
   negative supervision exists. Our expert-free lane instead learns a NOP
   category through trajectory advantage, with no section labels.

3. Do not transfer `p*TP-(1-p)*SL-cost` as expected profit. In `Composed`, p is
   an expert opening-class score, not P(TP before SL under these order levels).
   Even calibrated imitation probabilities do not fix that semantic mismatch.
   Early exits, forced exits, gaps and financing add outcomes beyond two fixed
   payoffs. A proposed net-value head must target actual executable net return
   under a declared continuation policy, or model the relevant outcomes.

4. Independently weighted opening classifiers are not automatically comparable
   as calibrated probabilities. Calibration must name its target event and
   fit only on its designated period. Our categorical policy probabilities are
   action-selection probabilities, not calibrated success probabilities; Brier
   or ECE against future profitability would not mean policy calibration.

5. In `run`, `max(ok or trials, ...)` and `max(ok or variants, ...)` select an
   invalid candidate when every candidate fails eligibility. Preserve such
   candidates as diagnostics, but return NO_ELIGIBLE_POLICY rather than a
   deployable or activity-qualified winner. This is a control-flow finding,
   not evidence that every historical run entered the fallback.

6. Environment clipping creates an intended/executed action mismatch. However,
   it is not correct to infer that environmental clipping directly zeroes the
   supervised Huber gradient: that loss does not backpropagate through the
   simulator. For RL, use the bounded distribution's proper log density and
   record intended versus executed parameters; do not assume a deterministic
   sigmoid regression head alone implements a stochastic RL distribution.

## Remaining verification

Plan 67 OA01-OA10 plus its NOP cases govern the replacement. Particular open
checks are conditional likelihood/entropy, masks at training and inference,
learned sizing, cost timing with discounting, and partially observed closers.
Current equity reward books financing on close; agreement in total P&L does
not alone establish equivalence of differently timed discounted rewards.
Do not silently change accounting/reward timing while changing action heads.

No expert-label training, real-data fit, TEST read or deployment was performed
for this review. Keep the input recipe and completed campaign artifacts intact.

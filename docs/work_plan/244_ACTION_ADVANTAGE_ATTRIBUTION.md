# Paired Action-Advantage Attribution

## Authorized Scope

Diagnose the fixed256 endpoints of243: advantage versus absolute in r19/r43,
TRAIN and VALIDATION, with zero optimizer steps. Keep both original source
policies and all endpoints unchanged. No new model, feature pass or threshold.

For every changed categorical decision, including economically neutral ones:

1. Reload both checkpoints and reproduce archived native signals exactly.
2. Verify each online critic equals its own calibrated checkpoint.
3. Reuse isolated and chronological native account continuations to attribute
   the entire account difference, including subsequent opportunities and sizing.
4. Also force both actions in the SAME isolated control-account state and then
   follow the unchanged parent70 greedy tape. This matches the continuation
   policy used for critic labels; account-state distribution can still differ
   from the original label panel and is explicitly recorded.
5. Compare both critics' action gaps with that matched-policy realized gap,
   alongside actor margins, local KL and the control-policy continuation gap.

There are six suffix simulations per changed event: isolated control suffix
old/new, chronological control suffix old/new, calibrated-parent suffix old/new.
Chronological cash contributions must sum to the complete account delta within
1e-6 USD. The analysis is descriptive; errors at diagnostic states do not alone
prove a coding bug, and shared-actor movement is not attributed to a unique
gradient source without a separate intervention.

## Requirements And Verification

| ID | Requirement | Test/evidence |
|---|---|---|
| P1 | No fitting or source replacement | Input receipts, immutable source hashes, zero optimizer steps |
| P2 | Matched support and actual actions | Existing trace parity and native replay checks |
| P3 | Do not conflate continuation policies | Synthetic dual-tape regression, explicit calibrated gap fields |
| P4 | Compare both critics on identical actions | Manual paired-gap, error and sign tests, malformed input rejects |
| P5 | Reconcile all account effects | Native telescoping regression plus all four real comparisons |
| P6 | Autonomous evidence | Persistent bounded worker, automatic authenticated collection |

CPU executes native continuations; the primary GPU emits frozen logits/Qs.
Report exact event counts rather than treating repeated market dates as
independent samples. Preserve harmful and neutral events. Validation results
are not used to fit filters. Reuse existing attribution/receipt helpers; the
new optional diagnostic continuation must leave existing cash attribution
unchanged when omitted.

# Changed Decision Attribution

Owner approved2026-11-09. Identify which changed categorical signals in231
cause losses,without changing weights,features,policy,risk or the simulator.

## Test Design

| ID | Required evidence |
|---|---|
| A01 | Source,probe and model hashes authenticated; native reference/candidate tapes reproduce archived cash and order counts |
| A02 | Clone mutable account and wrapper state; intervention cannot change its parent or read future features |
| A03 | At one timestamp change only the signal,then resume original policy; both alternatives start at identical position,balance and calendar state |
| A04 | Original continuation from a reference snapshot reproduces reference terminal balance |
| A05 | Chronological prefix contributions telescope exactly to candidate-minus-reference net profit; isolated effects are NOT summed as account PnL |
| A06 | Record requested command and actual first-step entries/closes/refusals; a changed signal may have no executed effect |
| A07 | Verify calibrated critic tensors unchanged in every actor checkpoint; record logits,margin and twin/min-Q gaps for each changed signal |
| A08 | Three common-random-number stochastic continuations of the reference actor from each reference snapshot compare realized soft returns for the two first actions |
| A09 | Return convention excludes entropy of forced first action; later surprisal uses reference-policy probability and native elapsed-time discount |
| A10 | Full TRAIN/VALIDATION only; counterfactual outcomes are retrospective diagnostics,never observations or new fitted labels; no TEST |

Scope:both retained/fresh arms,actor1/8/32,all changed decision timestamps in
TRAIN and VALIDATION. Reuse identical interventions across checkpoints/arms,
not across incompatible identities. The chronological attribution is order
dependent,not Shapley. Q gaps describe the fixed stochastic reference policy;
greedy cash deltas alone are not a critic correctness test. Three stochastic
draws give a diagnostic spread,not an exact conditional expectation or proof
that the market-only critic must distinguish hidden account states.

Independent native fixtures cover both directions,close versus hold,identical
actions,mutation isolation and discounted entropy accounting. Frozen feature
selection and live/paper deployment remain untouched. Outputs include per-event
timestamp/state,action and command changes,cash effects,Q gaps,logit margins,
ranking diagnostics and exact endpoint reconciliation.

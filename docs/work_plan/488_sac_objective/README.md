# 488. Can the frozen SAC critic recommend a better first action?

Follow-up to the [master checkpoint](../487_MASTER_PLAN_CHECKPOINT.md),
[critic calibration](../369_FROZEN_ACTOR_CRITIC_REPAIR.md) and
[role conflict diagnostic](../378_ROLE_CONFLICT_GATE.md).
Disposition: preserve Selected299 as a control. No new training, deployment,
2024 decision evaluation, or claim of repeatable training.

## Design

Reuse the 369 stratified panel of historical decision states, three forced
first actions per state. Compare terminal account net USD after continuing
with the same frozen greedy actor. Regret is the best of those three nets
minus the chosen net; it is not additive trading P&L. Ties within 1e-8 USD
do not count as inferior choices. Fit: 48 states in 2020-22; monitor: 16
in 2023. The monitor was previously used by 369, not a fresh holdout.
No inferred action gets to see the future. Retrospective rankings use the
engine outcomes and are explanatory ceilings, not deployable policies.

| First action ranked by | Mean USD regret, fit | Mean USD regret, monitor | Inferior choices, monitor |
| --- | ---: | ---: | ---: |
| Frozen actor | 15.80 | 7.29 | 4/16 |
| Lower of the two trained Q estimates | 14.68 | 13.42 | 6/16 |
| Mean soft target from four paired stochastic continuations | 3.24 | 4.38 | 2/16 |
| Realized native reward with subsequent greedy actions | 0.00 | 0.00 | 0/16 |

Dropping each stochastic tape in turn gave 3.18 to 6.79 USD monitor regret.
These are sensitivity bounds across omissions, not confidence intervals.
With all four tapes removing the entropy term from sampled continuations did
not change monitor cash regret. Even the correct stochastic ranking need
not maximize the net under greedy continuation. Conversely, greedy realized
native reward made no economically inferior choice on this panel; this is
not evidence that reward and dollars always agree.

For each state compare the SAC actor objective gradient in **logit** space
with the gradient of expected terminal net for a randomized first action
followed by a fixed greedy continuation. The cosine was negative for 9/12
informative monitor states (median -0.435); against the four-tape soft target
including first-action entropy, 10/16 (median -0.146). This is not a
network weight update or a derivative through a discrete argmax. The code
cross-checks the analytic SAC derivative with autograd, the registered source
and support identities, and two regret metrics with the earlier run.

## Decision and disconfirmation

Priority hypothesis: unreliable action advantages out of the fit period,
compounded by training under stochastic continuation and acting greedily.
Not a finding that overfitting, scarce independent episodes, or hidden
position information are ruled out. This small panel does not isolate which
of them makes Q inaccurate. Merely lowering LR cannot reverse a wrong local
direction for a fixed gradient.

Before another full actor run, preselect a larger panel of sessions without
using outcomes, group inference by session, cover flat/long/short and record
ties. Keep the actor, inputs and realized paths fixed. Cross two factors:
market-only versus position-aware critic (diagnostic only), and native soft
versus realized greedy-continuation target. Compare paired cash regret to
the actor and report by temporal block, role and independent fit seed. If
the advantage disappears on new sessions, reject this as a general repair;
if a critic transfers, test bounded actor updates against a frozen-actor
control before resuming full training. No implication for seasonal forecasting.

## Reproduction

Run [the CPU audit](../../../tools/role488_objective_audit.py) in the pinned
369 runtime, with arguments: parent runtime bundle, Selected299 export
bundle, archived 369 panel directory, output directory. Source checkpoint
SHA-256: `69b5254489d224d4b20ac358956e4dd3b9a67a2f214a9aff98045c5def25a536`.
Full outputs: [choices](choices.csv), [logit gradients](gradients.csv),
[tape sensitivity](leave_one_tape_out.csv).

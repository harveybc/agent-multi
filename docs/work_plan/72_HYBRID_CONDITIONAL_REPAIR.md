# Conditional-policy repair

Owner authorized repair and continuation on 2026-10-25. Stopped campaign and
its source snapshot are immutable. This is a new prospective qualification.

## Evidence and test design

The original learner fails the observable three-action fixture even at 102400
decisions. Changing initialization, suppressing value loss, reducing GAE,
splitting the value network, and modestly widening the core did not establish
the required conditional behavior in their recorded bounded controls. A direct
supervised control can fit the same signal; the ordinary SB3 PPO reference fits
it at 32768 decisions. These do not establish a unique cause in financial data.

Eight native paired scenarios confirm correct long/short net P&L and show that
closing beats holding before an adverse barrier while holding beats premature
closing before a favorable barrier, after costs and sizing. Fixture corrections
handled pandas read-only arrays and the configured TP penetration distance.

Qualification must include conditional NOP/long/short, long HOLD/close and short
HOLD/close, seeded repeats, current versus lagged signal, serialization, masked
actions and no future/feedback feature access. Financial activity alone cannot
confer acceptance. Input bytes, feature selection, scalers, calendar and costs
must remain unchanged.

## Candidate representation

Keep the admitted feature groups and their causal conv/pointwise branches.
Use a small causal fusion convolution and flatten its whole temporal sequence
at the policy/value-head boundary. This removes the two-channel last-step
bottleneck and two further nonlinear core convolutions. Use SB3's independent
policy/value MLP heads, categorical commands and conditional Beta risk.
This is an explicit architecture treatment, not the unchanged E3-M representation
and not proof of its superiority on financial data. No extra input normalization,
clipping, account vector, expert training or forced trading is introduced.

## Release order

1. CPU structural tests and conditional-learning controls, including native
   close/hold counterfactual accounting.
2. Real-shape GPU qualification with recorded architecture and tensor contract.
3. New identified finite financial batch, first qualified control then reward
   contrasts, same periods and selected inputs, automatic recovery/publication.
4. Report directional/close collapse explicitly. No strategy is promoted by
   synthetic success, profit alone or passing the activity minimum.

## Executed evidence

- 23 focused CPU tests pass, including the new sequence causality, routing-mask
  independence, separate policy/value heads, exact reload, eight native paired
  exit/hold economics cases, queue failure stop and conditional-learning control.
- Legacy flat conditional learning still fails at 102400 decisions. Seed 7 of
  the new small control passes at 65536; seed 19 fails the NOP case at that
  budget. This is recorded, not suppressed. Qualification therefore does not
  rely on a single successful tiny fixture.
- Full admitted tensor shape (120 columns, 13 slots), one policy on mixed
  flat/long/short routes: CPU seed 7 passes all nine contexts at 131072 decisions
  and reloads exactly. Expected-action probabilities are 0.9666 to 0.9991.
  Synthetic rewards are correctness labels for engineering only; no such
  labels, rewards or trained weights enter the financial batch.

## Automatic successor

Queue v2 preserves the stopped v1 runtime. Four fresh real-shape GPU controls:
seeds 7/19/43 on the current signal and seed 7 with the signal six slots in the
past, all at 131072 synthetic decisions. A constant or current-only policy
cannot pass the full suite. Each requires all nine correct argmax decisions,
probability above 0.6 and exact reload. These use rollout/batch 64 for
engineering; financial training retains rollout 512/batch 256. They certify
basic learnability, not convergence on financial data or an optimal strategy.
No failed qualifier is silently retried with different parameters.

If all pass, the existing fixed-risk control and seven reward/risk treatments
run automatically, three seeds each, 102400 financial decisions per fit. The
inputs, scale, periods, calendar, sizing and costs are unchanged. The effective
model records the representation override and parameter count. The learner
remains PPO with true categorical/conditional-Beta actions, not SAC.

Every financial report includes long/short counts, voluntary close counts and
missing-behavior reasons. Ranking requires activity plus observed NOP, both
directions, HOLD and voluntary close in both TRAIN and INNER. This is necessary
coverage, not proof that timing is good. Nothing forces an order or balances
directions artificially. Financial results retain their original measured
metrics even when behavioral eligibility fails. TEST remains closed.

Traceability: order semantics -> native economics witness; temporal observation
-> test_qualified_orders; head independence -> test_hybrid_policy_heads;
conditional learning -> qualification receipts; automatic successor/failure ->
test_hybrid_queue; market performance -> final native reports, not synthetic
qualification. Status before deployment: CPU checks passed, GPU suite pending.

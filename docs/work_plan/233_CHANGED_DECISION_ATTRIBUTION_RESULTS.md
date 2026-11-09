# Changed Decision Attribution Results

Completed 2026-11-09, 11:38:43 UTC. Runtime `035f68ac`: RTX5090 frozen inference,
then CPU native counterfactuals. Duration: 115.15 seconds. All 23 native tests
passed before execution; the final suite, including the report test, passed
24/24. All 103 terminal-listed files authenticated after collection. No training,
new features, observation feedback, policy thresholds or model promotion.

Identity:`c99582ba68f6daf8d9a30f802116154ad7a0cadaace5f4e3febecb12cf4d1416`.
Source is preserved70 epoch0,not the selected69 champion. Both preserved.
Full artifacts:operator artifact root `sac70-action-attribution-20261109`.
Published compact evidence:
[summary](generalization/changed_decisions_20261109/SUMMARY.json),
[all113 decision occurrences](generalization/changed_decisions_20261109/DECISIONS.csv),
[export receipt](generalization/changed_decisions_20261109/EXPORT.json).

## Exact Account Attribution

The 12 candidate accounts reproduce their archived timestamps, positions,
rewards, durations, commands, terminal cash, trade counts and native counters.
Every chronological attribution sums to candidate minus reference net profit
with zero reconciliation error. The 72 unique signal interventions are reused
across 113 occurrences; these are not 113 independent market observations.
Original continuation from each reference snapshot also restores exact cash.
There are 802 suffix continuations: 72 x 8 isolated/stochastic continuations
plus 113 x 2 chronological continuations, without retraining.

| Critic initialization / actor updates | TRAIN return % | VALIDATION return % | Changed signals TRAIN / VAL | Cash-effective changes TRAIN / VAL |
|---|---:|---:|---:|---:|
| Reference |11.51453|13.80849|0 /0|0 /0|
| Retained /1 |11.51453|13.80849|1 /0|0 /0|
| Retained /8 |11.51453|12.67332|3 /3|0 /2|
| Retained /32 |11.21926|12.67332|24 /8|6 /2|
| Fresh /1 |11.94572|13.80849|2 /0|1 /0|
| Fresh /8 |11.51186|13.23948|10 /7|1 /1|
| Fresh /32 |12.24694|12.71370|39 /16|9 /3|

Cash-effective means nonzero terminal net change under the chronological
intervention,not that every other signal had no requested-command change.
Calendar refusals and equivalent HOLD signals explain many zero effects.

## Where Validation Degrades

Fresh/8:the entire-56.90123USD difference comes from one additional SELL on
2024-08-13,13:00UTC.It opens and hits SL within the same execution bar.
Six other changed signals contribute zero terminal cash difference.

Fresh/32 has exactly three nonzero chronological contributions:

| Decision UTC | Old -> new signal | Executed change | Final account contribution USD |
|---|---|---|---:|
|2024-02-05 14:00|BUY -> NOP|Skip initial long entry|+4.649860|
|2024-05-27 17:00|NOP -> SELL|Additional short,later SL|-57.495980|
|2024-08-13 13:00|NOP -> SELL|Additional short,same-bar SL|-56.632630|
|Total||| -109.478750 |

Total:-109.478750USD,or-1.0947875 percentage points on10000USD initial capital.
These are full-account effects,including later balance-dependent sizing,not
the isolated trade's PnL.Two shorts are harmful;the skipped buy helps slightly.
The other13 signals contribute zero cash change. No changed early-close action
is needed to explain this validation deterioration.

Retained/8 and retained/32 instead add harmful shorts on August8 and August13;
their chronological effects sum to-113.516635USD.

## Critic Ranking And Decision Boundaries

Calibrated online twin critic tensors remain exactly unchanged in all six
actor checkpoints.Alpha is0.005312912631779909. On August13,the reference logits
NOP0.723107636 and SELL0.717362999 differ by only0.005744637.After eight fresh-head
actor updates they become NOP0.722577751 and SELL0.722815454:the argmax flips.
A small continuous logit move thus creates an additional full native order.
This is discrete-boundary sensitivity,not evidence of corrupted weight transfer.

At that same reference account state,the fresh min-Q prefers SELL over NOP
by+0.030978.Three paired stochastic reference-policy continuations give soft
return differences-0.551238,-0.554360,-0.547039.The original forced action's
entropy is excluded;future entropy and native-duration discount are included.
This is a conditional ranking counterexample,not just a greedy cash loss.

There are also TRAIN counterexamples after fresh/32:

| Decision UTC | Change | Greedy account contribution USD | Q new-minus-old | Three soft-return differences |
|---|---|---:|---:|---|
|2022-08-29 10:00|Skip BUY|-32.406750|+0.021703|-0.310419,-0.310477,-0.311484|
|2023-01-06 02:00|SELL/CLOSE long -> NOP/HOLD|-39.068220|+0.023172|-0.501748,-0.195065,-0.365904|

The second row is important for early closing:the update suppresses a useful
close of an already-open long.This is not an entry-direction-only problem.
Low average realized-return RMSE therefore does not establish useful action
rankings,even inside TRAIN.Resetting critics alone has not repaired this.

The distinction between evaluated policies also matters.On May27,the new SELL
loses57.50USD under the greedy reference suffix,but the three stochastic soft
differences are+0.000370,+0.004431,+0.006033.So a loss under greedy execution is
not automatically an error against SAC's stochastic continuation objective.
Likewise,some actions flip against their local Q ordering;shared head updates
can change a row without locally optimizing that row. This last explanation
needs a gradient intervention before attributing a specific optimizer cause.

## Scope And Next Correction

Closed:the complete cash deterioration in this bounded probe has been located
to exact categorical decisions and account paths;cash bookkeeping and cloned
interventions reconcile. No evidence here supports changing the feature pool,
discarding the representation or blaming all deterioration on validation shift.

Not established:the historical cause of every critic error,an exact conditional
Q expectation,or a reliable repair.Three common-random continuations are
diagnostic action draws on the same historical market,not independent market
replications.The market-only critic also averages unobserved account states;
we did not add account state to its observations.The chronological allocation
is order-dependent,not Shapley,and isolated effects must not be summed as PnL.

Next controlled repair should target TRAIN action ranking,not another long
unqualified refinement:compare fixed-actor alternative actions at identical
reachable accounts,measure ranking as well as RMSE,and contrast stochastic
versus greedy continuation explicitly.Then bound actor changes and measure
their effect on open/close/HOLD,NOP and eligibility before a long refinement.
Any corrective fit uses TRAIN only;these dated validation losses are diagnostic,
not a blacklist or thresholds to hardcode into the strategy.

The selected69 champion remains10.79211%TRAIN /14.59109%VALIDATION;the preserved70
reference remains11.51453% /13.80849%.Neither was replaced.No successor training
job was launched by this attribution run;the diagnostic service exited0.

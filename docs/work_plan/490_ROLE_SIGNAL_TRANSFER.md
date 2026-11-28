# 490. What the USD shortfall means, and whether action advantage transfers

Follow-up to [489](489_FROZEN_CRITIC_FACTORIAL.md). No change to actor,
deployed strategy, or input contract. No evaluation after 2023. These weeks
are later for the diagnostic readouts, not unseen by the original SAC.

## Account interpretation

Each of the 39 monitor weeks started with USD 10,000. For each of 795
sampled decisions, branch the same account three ways, changing only the
first signal. Continue with the same frozen greedy actor to weekly settlement.
Subtract the realized terminal net under the original choice from the best
of the three terminal nets. Average decisions within week, then weeks equally.
The result, USD 20.57, is not a debit, per-trade P&L, or an additive return.
NOP for the first decision only gives USD 20.42; minimum Q gives USD 23.81.
The actor's 131 executed trades had median margin USD 462.31. Replaying its
unmodified decisions yielded mean weekly net -0.62 USD, with 17 positive
weeks. The weeks were independently initialized, not compounded.

## Role-specific experiment

Reuse the authenticated panel and native critic-head inputs; verify the Q
surface exactly before fitting. With no position, compare NOP/BUY/SELL.
With a long or short position, collapse the two equivalent HOLD signals
after checking their cash identity, and compare HOLD versus CLOSE. Fit an
independent 128-tree regressor per role, minimum 8 examples per leaf, seeds
7/19/41. Flat sees market embeddings; occupied roles also see the 489
position context. Targets are three or two centered terminal net USDs.
Four chronological fits per role: 2020-21 then 2022, and the last 1, 2 or
3 calendar years through 2022 then 2023. No model selection on 2023.

Negative control: permute the complete target vectors within fit year
and role, preserving action dependencies and yearly marginals. Score these
predictions against unshuffled economic outcomes. Positive control: equally
spaced quantiles of the highest-variance observed feature (fit-only) define
a deterministic synthetic class. It tests wiring, not economic predictability.
Best-action accuracy excludes all-action ties. Economic comparisons retain
them; compute paired weekly means, average seeds and bootstrap weeks, not
individual decisions (5,000 resamples; descriptive intervals).

| Role, 2020-22 fit | Best action, fit | Best action, 2023 | Cost vs actor: fit / 2023 (USD) | Planted, 2023 |
| --- | ---: | ---: | ---: | ---: |
| Flat | 94.1% | 27.1% | -24.91 / +2.52 | 99.9% |
| Long | 93.1% | 71.0% | -9.20 / +1.48 | 100.0% |
| Short | 95.9% | 47.3% | -12.35 / +1.24 | 99.8% |

All three fit successfully and none beat the actor economically on 2023
with that training span. Against the shuffled-label controls, the paired
interval includes zero for each role. Long positions provide only 46
monitor decisions from 9 weeks. Short closes trained with 2022 alone
improved on the actor by 4.00 USD in 2023; two years improved nothing
(+0.08 USD), and three worsened regret. Exploratory, no multiplicity
adjustment. Flat trained with 2020-21 beat the actor on 2022 but not with
the expanded fit on 2023. Neither historical scarcity nor market drift
is established as the sole cause. Next contrast: recent versus older
blocks matched for number of informative states, on multiple later blocks.

Scripts: [panel and fits](../../tools/role490_signal.py),
[summary](../../tools/role490_report.py). Reuse the pinned 489 runtime:
run `prepare` once, then `fit --role` for 0, 1 and -1 on the same panel,
then summarize the three output files. Gamma ran flat, Omega long/short
on CPU. [Account replay](490_account.json) and [full results](490_role_summary.csv).

# Four/five future-event results

Completed 2026-10-28 under plan 120. Population 20, generations 30, seeds
7/19/43, horizons 3/6/12/24 h, 24 new searches, no reused candidate scores.
K=4 search runtime 120.896 s; K=5 runtime 123.407 s; total 244.302 s, excluding
audits and analysis. Both post-run audits passed. No active jobs remain.

[Machine-readable results, original receipts, paired and fixed-gene probes](../audits/evidence/zigzag_next4_next5_20261028/RESULTS.json)

## Additional exits on identical origins and genes

These scores use the K=3 champion genotype for each horizon, fixed across
K=2/3/4/5. All four probes use their common complete origins. They isolate
additional allowed exits from the effect of reoptimization. The original
K=2/3 champion fitness and every saved record array reproduced exactly with
the extended evaluator. The full measured score is still retrospective
net-profit / (risk * hours), not account return or learned-model performance.

| Horizon | Common origins | K=2 | K=3 | K=4 | K=5 | Gain 4 vs 3 | Gain 5 vs 4 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 3 h | 18,320 | 0.274700903 | 0.275965623 | 0.276334568 | 0.276520909 | +0.133693% | +0.067433% |
| 6 h | 16,666 | 0.249967997 | 0.250604967 | 0.250780274 | 0.250868176 | +0.069953% | +0.035051% |
| 12 h | 10,698 | 0.189714539 | 0.190001177 | 0.190033628 | 0.190045443 | +0.017079% | +0.006218% |
| 24 h | 4,999 | 0.165838166 | 0.165995834 | 0.166000169 | 0.166003258 | +0.002611% | +0.001861% |

Positive but decreasing marginal gains occurred at every scale in this sample.
With fixed genes and complete support, the best oracle score cannot decrease
as options are added; the measurable question is how often and how much it
increases. This property passed on every shared origin, not only in the mean.

| Horizon | Origins improved by fourth | Origins improved by fifth |
|---|---:|---:|
| 3 h | 1.6485% | 0.8734% |
| 6 h | 0.9600% | 0.5160% |
| 12 h | 0.5048% | 0.1963% |
| 24 h | 0.2000% | 0.1400% |

The complete exit-ordinal distribution is in the evidence. Percentages here
are relative to each horizon's shared-origin set, not all TRAIN origins.

## DEAP champions and coverage

Objective uses the fixed 19,879-origin denominator. It can fall with K because
origins lacking K events in the observed session chunk contribute no score.
Do not interpret this fall as evidence that additional exits reduce the best
per-origin opportunity. Costs, risk floor and all other rules remain unchanged.

| Horizon | K=4 fitness | K=5 fitness | K=4 coverage | K=5 coverage | K=4 genes | K=5 genes |
|---|---:|---:|---:|---:|---|---|
| 3 h | 0.264771386 | 0.254834904 | 96.01% | 92.16% | 3 / 5.0 / 1 | 3 / 5.0 / 1 |
| 6 h | 0.225521801 | 0.210320892 | 90.51% | 83.84% | 4 / 5.0 / 2 | 4 / 5.0 / 2 |
| 12 h | 0.126518326 | 0.102274066 | 67.50% | 53.82% | 8 / 21.1 / 5 | 8 / 21.1 / 5 |
| 24 h | 0.070384871 | 0.038875377 | 44.41% | 23.36% | 12 / 17.1 / 11 | 13 / 114.5 / 10 |

Genes are depth / deviation in pips / backstep. Deviation values are not
uniquely identified optima. Best-of-three-seeds results are not global optima.

## Search adequacy witness

At 24 h and K=5, the registered fixed-genotype control (12 / 17.1 / 11)
scored 0.041745071965 on the same full origin denominator, better than DEAP's
0.038875376650. The two have different complete supports; their comparison
is under the explicitly fixed-denominator objective. The control's coverage
is 4,999 / 19,879 = 25.1471%, versus the search winner's 23.3563%.
This is a directly measured witness that the 20x30 search did not find the
best already-known candidate for this setting. It is retained as a separate
probe, not retroactively called a DEAP-discovered winner.

## Verification and interpretation

26 focused tests passed. Both audits validated 720 generation records total,
18,432 candidate paths and 4,096 selected scalar paths; selected reconstruction
maximum absolute error was zero. Inputs, software, operators, budgets and
business-entry support matched the previous comparison. No validation/test
data or neural training was used.

Keep K=4/5 as measured oracle alternatives, not discarded because gains are
small. Future refinement should include authenticated earlier champions as
explicit starting candidates/controls, and distinguish coverage from objective
gain. In particular, the observed-session boundary strongly restricts K=5 at
24 h; this limitation must remain explicit in any subsequent profile decision.

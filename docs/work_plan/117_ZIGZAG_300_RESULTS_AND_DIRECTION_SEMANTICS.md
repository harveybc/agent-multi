# ZigZag 300-generation paired results and decision semantics

Date: 2026-10-28. [Protocol](116_ZIGZAG_300_GENERATION_CONTRAST.md).
Completed 12 searches: population 30, 300 generations, seeds 7/19/43.
Search runtime 999.29 s; baseline 222.91 s.
Baseline evaluations were authenticated and reused, not old completed units.
All 480 initial generation records match the baseline exactly. Baseline files
are unchanged; 2,048 sampled paths pass independent scalar reconstruction.
53 focused tests passed. No GPU, RL training, VALIDATION or TEST read.

[Machine-readable comparison, audit and outputs](../audits/evidence/zigzag_opportunity_300_20261028/RESULTS.json).

| h | Best F, 40 gen | Best F, 300 gen | Relative gain % | New Depth / Deviation pips / Backstep |
| ---: | ---: | ---: | ---: | --- |
| 3 | 0.272396447 | 0.272396447 | 0.000 | 3 / 1.6 / 1 |
| 6 | 0.245217139 | 0.245217139 | 0.000 | 4 / 1.6 / 2 |
| 12 | 0.180878858 | 0.180878858 | 0.000 | 9 / 1.6 / 3 |
| 24 | 0.142693868 | 0.142693868 | 0.000 | 12 / 1.6 / 11 |

Best-per-horizon fitness is unchanged. Seed 7 at 12h improves by 0.6604%;
seed 43 at 24h improves by 3.8159%; neither beats its prior horizon champion.
The retained deviation changes under ties without measured pivot/fitness change;
do not interpret that as a newly identified meaningful threshold.

## Latest owner clarification: two future extrema

The evaluated mechanism already finds the next strictly later peak and valley
separately for every origin and candidate, by time (not nearest price). It
computes buy-to-peak and sell-to-valley scores and chooses max(0,buy,sell).
Two target arrays and two exit-index arrays are retained in NPZ. Each side's
full intervening OHLC path is used for drawdown and elapsed time: extrema prices
alone cannot determine the requested profit/(drawdown*duration) metric.

It is not an unconditional next-extremum evaluator. If the next pivot is
beyond min(4*h, forced-flat, continuous-data endpoint), it uses that limit's
opening quote. This behavior was declared in 114 before the baseline, but
must not be misrepresented as the owner's literal always-next-pivot rule.
No change to that rule was made during this controlled 300-generation contrast.

| h | Origin buy % | Origin sell % | Origin NOP % | Timed exits among chosen opportunities % |
| ---: | ---: | ---: | ---: | ---: |
| 3 | 50.209 | 49.469 | 0.322 | 7.767 |
| 6 | 50.299 | 49.454 | 0.246 | 5.885 |
| 12 | 49.399 | 50.163 | 0.438 | 10.989 |
| 24 | 49.338 | 49.857 | 0.805 | 18.302 |

Denominator for buy/sell/NOP: all 19,879 origins, with hypothetical overlapping
opportunities. Denominator for timed exits: positive chosen opportunities.
Positive buy/sell exact-score ties: zero in all four finalist maps.

| h | Non-overlapping hindsight intervals | Buy % | Sell % |
| ---: | ---: | ---: | ---: |
| 3 | 4084 | 50.318 | 49.682 |
| 6 | 3121 | 50.368 | 49.632 |
| 12 | 1669 | 50.270 | 49.730 |
| 24 | 1088 | 49.908 | 50.092 |

The second table is the separate fixed-unit maximum-net-profit interval
schedule; it was not the fitness fed to DEAP. No learned strategy performance
is claimed. Discuss next-extremum versus time-limited semantics before altering
the next run or deriving risk profiles. Floating selection remains later.


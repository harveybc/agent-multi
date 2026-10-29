# Corrected ZigZag: population 20, generations 30

Owner-requested comparison, 2026-10-28. Complete: 24 searches, four horizons,
two event counts, seeds 7/19/43, population 20 and all 30 generations each.
No capped-search scores or caches were adopted. TRAIN is 2020-2023; validation
and test were not read. Total search runtime: 226.799 seconds (110.348 +
116.451), excluding subsequent scalar audit. CPU only, no financial model fit.

[Full results, seed scores and audit receipts](../audits/evidence/zigzag_next_events_p20_g30_20261028/RESULTS.json)

## Objective and champions

Per origin, evaluate both directions at each of the next K ZigZag events.
Score = net unit profit / (max(path drawdown, round-trip cost, 1 pip) * hours).
Choose the greatest positive score, otherwise NOP. Costs are the same assumed
profile as before, not newly optimized or broker verified. No 4*h cap and no
timed-price substitution. Both arms use the fixed 19,879-origin denominator;
origins without all K future events in the observed session chunk contribute
no opportunity and are counted separately as censored, not as executed losses.

Best seed per horizon (not a claim of global optimality). The best parameters
happen to agree between K=2 and K=3. Deviation units are pips; depth/backstep
are H1 bars. Nominal horizons are median-pivot-spacing bands, not holding caps.

| Horizon | Fitness K=2 | Fitness K=3 | Depth | Deviation | Backstep |
|---|---:|---:|---:|---:|---:|
| 3 h | 0.272202080 | 0.271176912 | 3 | 5.0 | 1 |
| 6 h | 0.243157645 | 0.237675675 | 4 | 5.0 | 2 |
| 12 h | 0.171272058 | 0.151035613 | 8 | 21.1 | 5 |
| 24 h | 0.128755926 | 0.098117667 | 12 | 17.1 | 11 |

Deviation is a tie-selected genotype, not evidence of a uniquely useful value.
The scores are retrospective opportunity ratios per hour, not Sharpe or return.

## Paired interpretation

| Horizon | Common origins | K=2 mean | K=3 mean | Relative gain | Coverage K=2 / K=3 |
|---|---:|---:|---:|---:|---:|
| 3 h | 19,620 | 0.273540870 | 0.274756668 | +0.4445% | 99.69% / 98.70% |
| 6 h | 19,108 | 0.246621949 | 0.247265791 | +0.2611% | 99.16% / 96.12% |
| 12 h | 16,233 | 0.184662928 | 0.184958846 | +0.1602% | 94.19% / 81.66% |
| 24 h | 12,539 | 0.155440853 | 0.155553162 | +0.0723% | 87.15% / 63.08% |

On identical complete origins K=3 provides a small improvement. Its global
fitness falls because the third event is unavailable more often before the
observed chunk ends. This is not evidence that more information makes an
optimal per-origin choice worse. Nor is the hindsight gain learned performance.

## Direction accounting

Percentages below use all 19,879 origins, including censored origins in the
denominator; buy+sell need not sum to 100. These are hypothetical opportunities,
not a stream of simultaneously feasible account orders.

| Horizon | K=2 buy / sell | K=3 buy / sell |
|---|---:|---:|
| 3 h | 50.18% / 49.47% | 49.75% / 48.95% |
| 6 h | 49.96% / 49.20% | 48.46% / 47.66% |
| 12 h | 47.01% / 47.18% | 40.09% / 41.57% |
| 24 h | 43.75% / 43.40% | 30.06% / 33.02% |

## Verification and next step

23 focused tests passed before launch. Both post-run audits passed: 720 total
generation records checked; 4,096 selected scalar paths with maximum absolute
reconstruction error zero; 10,240 candidate paths independently enumerated.
Receipt/data/code identities and interval-schedule nonoverlap passed.
The optimization is complete, with no active process for these two arms.
Before adopting K=3 for risk-profile design, account for its boundary coverage
explicitly; do not silently alter the observed-session support or reinterpret
these opportunity ratios as strategy qualification. Floating selection remains
separate from this oracle-design experiment.

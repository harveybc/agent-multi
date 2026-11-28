# 491. Does recent experience outperform older experience at equal coverage?

Follow-up to [490](490_ROLE_SIGNAL_TRANSFER.md). No deployment, SAC update or
2024 decision evaluation. Four disjoint held semesters, 2022 H1 through 2023
H2. Before each: recent = preceding 12 months; old = previous 12 months.
The actor that generated the panel was itself trained on these years.

Use the exact SHA-verified 490 panel, same features, role routing, centered
net USD targets, three seeds and forest settings. Select fit states with
action cash spread >1e-8 USD in hindsight; never use that filter to choose a
test action. Retain ties on test. Within corresponding quarters rank training
weeks by number of informative states, randomize ties, pair available weeks,
and take the smaller count without replacement. Thus each fit pair has
identical counts, week loads, and quarterly coverage, but not necessarily the
same market regime or distribution of profitable actions. For the nested
smaller-size contrast keep ceil(n/2) per matched week, not half the weeks.
Skip fits below 16 states (two minimum leaves of eight). 102 forests run;
the original neural weights are frozen. Openings on Gamma CPU, closes on
Omega CPU. Independent worker output by role; no GPU training.

**Sign convention:** positive = recent choice led to a higher terminal net
than old (or actor, as named), from the same decision state followed by the
same greedy actor. This is not aggregate portfolio profit. Average first
within held week, then across seeds and weeks. Intervals resample paired held
weeks 10,000 times; exploratory, not corrected for multiple comparisons or
serial dependence across weeks. Seeds are not independent market histories.

| Recent minus old, USD/decision, full budget | 2022 H1 | 2022 H2 | 2023 H1 | 2023 H2 |
| --- | ---: | ---: | ---: | ---: |
| Flat | -0.94 | +0.74 | +2.84 | +1.30 |
| Close long | insufficient | insufficient | insufficient | +0.52 |
| Close short | +0.28 | -3.82 | +6.15 | -3.33 |

All nine estimable full-budget paired intervals include zero. Recent openings
beat old openings in both 2023 halves but underperformed the actor by 2.04
and 5.68 USD. Recent short closes beat the actor by 12.81 USD in 2023 H1
and fell behind by 6.10 USD in H2. At the smaller size the recent-minus-old
short-close deltas for these halves are +7.04 and -0.19 USD. No reliable
one-year window policy follows. Long-close fits for the first three folds
had just nine matched states across two weeks per arm; the only eligible
long-close fold tested ten states in three weeks. More fit experience
within the recent window helped in three of eight full-vs-smaller comparisons
available for flat/short; that sensitivity on this small diagnostic does
not establish the value of more independent training history for SAC.

The rolling windows change with the held semester. A useful next diagnostic
would keep both fitted models fixed when measuring the first and second halves
of the same later year; that separates changed training composition from changed
evaluation conditions. Keep the actor as a control.

[All counts, actor deltas, intervals and skipped cells](491_summary.csv).
[Script](../../tools/role491_recency.py): run once per role with output
directory, `--panel` pointing to 490's panel and `--role` 0, 1, or -1;
gather all six CSVs, then rerun with the output directory alone to summarize.

# 410: Does the direct opening score rank opportunities?

Role 409 fixed ridge estimator, frozen representation, net managed opening
advantage relative to waiting. Fit on 2022, score 2023 within TRAIN. The
encoder was selected retrospectively; the held-forward claim concerns only
the signed-advantage fit. No 2024 outcomes used and no new trading threshold.
Each row below is the median of three seed-specific diagnostics in basis
points of initial account capital; hypotheses on the same market overlap.

|Side|Hours|D10 predicted|D10 realized|Realized D10 minus D1|
|---|---:|---:|---:|---:|
|BUY|3|1.18|-4.44|2.29|
|BUY|6|1.91|-3.26|3.61|
|BUY|12|2.90|-4.00|1.38|
|BUY|24|3.08|-4.18|-0.14|
|SELL|3|4.02|-5.94|-0.78|
|SELL|6|5.69|-5.95|-1.40|
|SELL|12|5.55|-4.56|-0.85|
|SELL|24|6.04|-5.27|-1.18|

Every D10 mean bid is positive; every D10 realized mean is negative. Ten
of 24 D10-minus-D1 contrasts are positive, including all nine BUY 3/6/12h;
only one of twelve SELL contrasts is positive. None of 24 individual
descriptive 95% weekly cluster bootstrap intervals for the spread lies
entirely above zero (2,000 resamples, fixed decile assignments). These are
not independent replications or simultaneous tests. Deciles are descriptive
ranks of the scored year, not prospective cutoffs. The native 409 direct
selected-offer counts and mean payoffs replayed in all three seeds.

Decision: no promotion and no threshold search. Before more model training,
decompose the same TRAIN-only offers into execution costs, managed versus
barrier-only exit, and the value of the displaced continuation, paired at
the identical opening. This diagnostic cannot itself blame the closer, costs,
or the encoder.

Reproducer: `tools/role410_deciles.py` using the 409 CPU environment.
Full per-seed predicted/realized means, counts and positive fractions:
`/var/tmp/role410-train-deciles/DECILES.csv`. Correlations and intervals:
`/var/tmp/role410-train-deciles/RANK.json`.

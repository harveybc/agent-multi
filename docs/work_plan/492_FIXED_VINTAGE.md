# 492. Same fitted models in both halves of 2023

Follow-up to [491](491_MATCHED_ROLE_RECENCY.md). Fit ExtraTrees on the frozen
critic representations, separately for 2021 and 2022, with the full matched
budget and the three existing seeds. Save each estimator once, verify its
SHA-256 before loading it separately for each half of 2023. No SAC updates
or access to 2024 outcomes. This is a diagnostic of first-action cash
advantages conditional on subsequent actions of the same frozen actor,
not a full deployment backtest or a fresh holdout for the historical actor.

Per year and seed: flat 315 informative states from 34 weeks, short 155 from
21 weeks. Long has only 9 from 2 weeks in the matched design; skip (two
minimum leaves require 16). The 12 fitted diagnostic checkpoints remain in
`/var/tmp/critic492`. Exact H1 reproduction check: the three paired USD
differences for every seed and week agree with 491 to 1e-8 USD.

| Choice | H1: 2022 minus 2021 | H2: 2022 minus 2021 | H2-H1 (descriptive 95% interval) | 2022 minus actor: H1 / H2 |
| --- | ---: | ---: | ---: | ---: |
| Open or wait | +2.84 | -2.22 | -5.05 [-11.92, +1.77] | -2.04 / -6.95 |
| Hold or close short | +6.15 | -1.64 | -7.80 [-17.19, +0.72] | +12.81 / -4.36 |

Positive means higher terminal net in USD per sampled decision. Average
within week, then over seeds and weeks (19/20 H1/H2 weeks for flat; 15/16
for short). The difference-of-halves intervals independently resample the
observed weeks of each half, not seeds or individual decisions. Exploratory:
no adjustment for selection, multiple comparisons or serial dependence.
The mean sign reversal survives removal of any single week within either
half. All three seeds switch for flat; two of three switch for short.
Thus changing the training window is not required for the observed reversal.
These measurements do not establish market regime drift as its cause.
No model promoted; preserve actor as baseline.

Source: [runner](../../tools/role492_fixed_vintage.py).
Artifacts in [492_fixed](492_fixed/): per-week/seed results, aggregation and
three manifests with fit identities and checkpoint hashes. To reproduce, run
the runner with the 490 panel and each role, then aggregate with the 491
weekly folder passed as `--previous`.

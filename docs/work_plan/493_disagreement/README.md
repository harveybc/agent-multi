# 493. Which disagreements reversed the advantage?

Read-only attribution of [492](../492_FIXED_VINTAGE.md). Original actor
unchanged; 2021/2022 fits unchanged. The pinned 490 panel and the SHA-256 of
all 12 diagnostic checkpoints were checked before inference. No 2024 action,
fitting, calibration or policy selection. Long closes had no qualified 492
pair. All per-week/seed cash differences vs 2021 and vs the actor reproduce
492 within 1e-8 USD. States come from the frozen actor's visited distribution.

For every state compare the two first actions, then follow the same greedy
actor to weekly settlement. For each ordered pair (2021 choice, 2022 choice),
its contribution is the 2022-minus-2021 terminal cash difference times the
pair indicator, averaged over **all** sampled decisions, with equal weights
per week and per seed. Agreement contributes zero by construction. The
conditional value divides that contribution by the pair weight. Totals are
USD per sampled decision, not combined P&L from a deployable account.

| Short position: 2021 to 2022 | H1 contribution | H2 contribution | H1 conditional | H2 conditional |
| --- | ---: | ---: | ---: | ---: |
| HOLD to CLOSE | +6.71 | -3.52 | +22.34 | -9.79 |
| CLOSE to HOLD | -0.56 | +1.87 | -4.20 | +10.22 |
| Total, exact values rounded | +6.15 | -1.64 | | |

The first switch contributed about -10.23 USD of change across halves,
partly offset by +2.43 from the opposite switch. Its weighted frequency
increased from about 30% to 36% of all short-position states. Its ranking
reversed: on average, initiating the close now instead of waiting helped in
H1 and hurt in H2. This does not identify the optimal later exit, nor
attribute every dollar to the individual trade rather than subsequent
opportunities. In H2 these close-instead-of-hold choices contributed -3.52
USD of the recent model's -4.36 USD vs the original actor.

Flat: NOP to SELL (-0.24 to -3.34) and BUY to SELL (+2.14 to +0.33)
explain most of the +2.84 to -2.22 reversal. In H2 proposing SELL instead
of NOP was worse by 12.88 USD conditional on that disagreement. This is
not evidence for disabling all sell openings or all early closes.

[Transitions](transitions.csv) includes agreement rows, counts, weekly
coverage, conditional and additive cash measures. Counts aggregate
state-seed evaluations, not independent trades. [Summary](summary.csv)
also counts distinct states with at least one disagreement.
[Per-seed contributions](seeds.csv); [replay script](../../../tools/role493_action_attribution.py).

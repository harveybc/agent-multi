# 386: changing the opener target restores abstention but not account performance

See [preregistered comparison](386_EXPECTED_ADVANTAGE.md). Same source closer,
market embeddings, 385 data and engine. Fit 2020-21; monitor 2022 H1.
2023 is an internal conditional comparison, not an independent validation
of the source representation. No change to live, feature selection or costs.

| 2023 opener | Epochs / restored | NOP on eligible flat account ticks | Trades (buy/sell) | Wins | Net USD | Return | Peak DD |
|---|---:|---:|---:|---:|---:|---:|---:|
| Frozen source | - | 3581 | 204 (34/170) | 123 | 308.64 | +3.09% | 4.02% |
| 385 classifier | 144/64 | 11 | 867 (333/534) | 443 | -3184.86 | -31.85% | 32.97% |
| Conditional mean 19, primary | 83/3 | 826 | 721 (344/377) | 367 | -2364.83 | -23.65% | 24.43% |
| Conditional mean 7 | 2000/1995 | 1271 | 555 (60/495) | 279 | -2675.51 | -26.76% | 27.55% |
| Conditional mean 43 | 145/65 | 1438 | 574 (249/325) | 293 | -2022.15 | -20.22% | 22.82% |

Both historical accounts matched; all three CPU exports reproduced every
held-out GPU decision. Net reconciled to the trade ledgers. On the common
4716 flat opportunities, new entries against source NOP averaged -5.13,
-8.14 and -4.85 bp for seeds 19/7/43. These are paired branch differences,
not additive account dollars: once entered, the account is no longer flat.
All three heads also lost to the constant FIT mean on squared error and
to always waiting on mean local advantage. Training longer was not enough
for seed 7. This rejects these particular replacement openers, not BUY/SELL
specialization in general.

Exploratory follow-up on the 2022 H2 calibration block only: veto an
incumbent entry when its own side's predicted advantage is nonpositive.
Never add or reverse an entry. The source returned +5.71% on 139 trades;
19 returned +6.86% on 110, 7 +2.62% on 62, 43 +4.28% on 98. No
consistent advantage across seeds; do not promote a veto based on seed 19.

Next gate: measure whether any causal confidence score ranks the net value
of extra entries on chronological internal blocks before funding another
account-level training sweep. Separate this from the already tested zero
threshold. No external language model or test-period use.

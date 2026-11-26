# 481: Volatility removed from the direction decision only

Decision: do not replace the control. On archived years 2020-23 the seasonal
arm improved net return in two of four folds, and improved both net return
and realized drawdown only in 2020. This is a diagnostic, not a new holdout.

At each origin compare the best gross BUY with the best gross SELL (three
barrier scales per side). Refit that gross ridge after dropping log(ATR/close)
and its square. Once a direction wins, **both arms use exactly the same six
original net scores** for within-side ranking, abstention and sizing. Hence
the same direction on the same free origin produces the same order. Volatility
is still present in the barriers, forecast normalization and frozen net scores.
No predictor retraining, new early exit, fee change or parameter search.
Each fit uses the preceding semester with all labels settled before evaluation.

| Seasonal arm | 2020 | 2021 | 2022 | 2023 |
|---|---:|---:|---:|---:|
| Control net return (%) | 0.56 | 21.24 | -1.03 | 0.99 |
| Ablation net return (%) | 8.40 | -8.99 | -2.50 | 4.24 |
| Control realized DD (%) | 30.64 | 11.31 | 32.07 | 17.58 |
| Ablation realized DD (%) | 25.06 | 17.17 | 29.15 | 25.09 |

In 2022 at 24 h the seasonal arm recognized 971 rather than 221 of the 1,699
profitable superior BUYs, but recognized only 818 rather than 1,566 of the
1,785 profitable superior SELLs. Gross sign accuracy at that scale moved from
1,818/3,580 to 1,839/3,580. These are overlapping paired probes, not trades.
At account level, 1,910 origins picked gross BUY across scales, 196 had a
positive within-side net score and only 7 could be executed with a free
position (3 wins). Control counts were 198, 164 and 2 (1 win). In particular
the 1,714 rejections by net score cannot be attributed to fees alone: the
frozen score also retains the original gross model.

Reproduced 875 archived control bookings with entry/exit, side, horizon,
size, PnL and balance parity; checked 130,638 paired action outcomes against
the reference fill engine. The 14,360 out-of-fit origins are shared by both
arms. [Accounts](accounts.csv), [gross rankings](paired.csv), [BUY decision
counts](buy_funnel.csv), [trade records](books.csv.gz).

```bash
OPENBLAS_NUM_THREADS=1 PYTHONPATH=. python -m tools.role481_direction_volatility "$RUN477" "$HISTORY397" "$MARKET" "$BUNDLE" docs/work_plan/479_gross_cost "$OUT"
```

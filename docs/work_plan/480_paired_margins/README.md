# Paired BUY vs SELL on the 2022 fold

Use archived 477 forecasts, 479 clean gross ridge, 1,887 calibration origins
(second half of 2021) and 3,580 evaluation origins (2022). For each origin
and barrier scale, simulate both sides with the original execution rules.
This is a retrospective opportunity comparison, not an executable account:
origins overlap and a profitable opportunity may occur while a position is
already open. Gross means before friction on those same fills; net includes
all simulator costs. A side preference here is a pairwise ranking, not an order.

| Barrier scale | Actual gross BUY > SELL | Actual net BUY > max(SELL,0) | Seasonal ridge ranks BUY higher | Profitable superior buys ranked BUY |
|---|---:|---:|---:|---:|
| 6 h | 1,699 | 1,608 | 119 | 40 |
| 24 h | 1,772 | 1,699 | 450 | 221 |
| 48 h | 1,645 | 1,584 | 298 | 110 |

At 24 h the seasonal model ranked BUY above SELL at 984/1,887 fit origins
but only 450/3,580 evaluation origins. The realized gross BUY-superiority
frequency rose from 898/1,887 to 1,772/3,580 rather than collapsing.
Realized mean gross BUY minus SELL improved from -5.88 to -2.59 USD;
the context-only model instead moved from -5.88 to -102.22 USD. Its two
volatility terms together contribute -95.92 USD to that -96.34 USD change.
Amounts refer to a fixed USD 10,000 probe account, not compounded returns.
The decomposition is exact for this linear model; it is not evidence that
volatility itself causes prices to move down. The correlated linear and square
terms should be interpreted jointly.

The complete results for both input sets, in-sample and out-of-sample, are
in [paired.csv](paired.csv); [shift.csv](shift.csv) decomposes the mean
forecast shift by input. The driver verifies fills against the outcomes
engine, label maturity, archived prediction identity, and reconstruction of
the total mean score shift. Reproduce using the same first four arguments
as 479 and a new output directory:

```bash
OPENBLAS_NUM_THREADS=1 PYTHONPATH=. python -m tools.role480_paired_margins "$RUN477" "$HISTORY397" "$MARKET" "$BUNDLE" "$OUT"
```

Next control: fit with both directional volatility terms removed together;
retain ATR for barrier sizing and retain actual execution costs. Compare with
the unmodified selector on all four archived folds before changing any policy.
Judge returns, drawdown and directional accuracy, not proximity to a 50/50 split.

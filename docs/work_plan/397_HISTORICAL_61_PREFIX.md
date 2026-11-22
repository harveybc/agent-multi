# Eight-year historical TRAIN arm (61 inputs)

This is a historical-data comparison before the synthetic-augmentation arm.
Do not use the dual-account collector or stratified replay. One epoch is one
account from 2016 through 2023. Validation uses only the original 2024 market.
The representation, frozen TRAIN estimators and scaler are unchanged; the early
years are retrospective training examples, not a simulated model vintage.

1. Build with `tools.history397`: independent H1 checked against M30 counts,
   five observed BID bar repairs using FXCM weekly files (2016 week 35 and
   2019 week 22). Aggregate source minutes and require all four values to match
   the native H1. No forward fill. The repaired Sunday reopening has 35 actual
   minutes, not 60 invented quotes. Source times and retrieval convention:
   [FXCM candle archive](https://github.com/fxcm/MarketData#candle-data).
2. Preserve the verified early calendar. For the four uncertain year-end
   intervals use the explicit research exclusions recorded in RESULT.json
   (187 historical quotes excluded); these are not asserted venue closures.
   This revision does not change any original TRAIN or validation session.
3. Verify with `tools.account397`: 24,654 historical prefix bars, 24,604 usable
   decision windows, 512 pre-2016 warmup bars for the producers and order risk
   scales. Two undefined price-efficiency observations remain masked. Last
   original-input checkpoint restored: the 61-input projection, not the 71-input
   incumbent. Original four-year control reproduces 286 trades and -3.969722%.
   Equity carries across the join at 8,928.868855 without a reset.
4. `tools.train_history397` defaults to CPU inference preflight, with standalone
   validation parity and input hashes. No learning occurs in that mode. The
   separate operator launch uses the same arguments plus `--execute` after GPU
   admission. Four-epoch matched-design pilot, one native replay buffer, no
   synthetic rows and no promotion. Epoch-zero activity can be ineligible and
   is not evidence for deployment. Compare against the matched 61-input control,
   not a different head, width, or checkpoint.

Epoch zero, not a training result: 499 trades, -14.255850%, Sharpe -0.603 over
2016-23; 52 trades, -1.054650%, Sharpe -0.297 on real 2024.

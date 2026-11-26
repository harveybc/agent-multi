# 483: Sequential account replay with and without the net-score gate

In each arm, both accounts start each fold at USD 10,000 and size every
order at q=1 (10% of then-current balance for margin at leverage 50).
The gross selector chooses the side, and the net model selects the highest
scoring horizon **within** the side, even if all scores are negative. A
gross-side tie abstains in both. The only switch is whether the winning
net score must be strictly positive. Each account independently skips all
origins through its exit bar. Both use the original barriers, weekly
calendar and realized execution costs. Drawdown is based on closed trades.

## Seasonal selector without explicit volatility in the side decision

| Year | Return on/off (%) | Realized DD on/off (%) | Orders on/off | Gated orders displaced |
|---|---:|---:|---:|---:|
| 2020 | +1.75 / +4.29 | 25.52 / 33.07 | 96 / 109 | 35 |
| 2021 | -11.87 / -27.15 | 17.73 / 30.67 | 91 / 118 | 36 |
| 2022 | -6.33 / +12.20 | 29.15 / 32.39 | 121 / 124 | 40 |
| 2023 | -2.33 / +3.15 | 27.30 / 28.02 | 126 / 149 | 18 |

Disabling the gate improves return in three years but increases realized
drawdown in all four. Compounding the four annual returns gives -17.96%
with the gate and -12.07% without it. This is an archived-fold comparison,
not independent confirmation. In 2022, 81 of 121 original trades survive,
33 nonpositive-score trades execute, and the new schedule also admits 10
previously unbooked positive-score trades. BUY goes from 3 wins/7 orders to
14/35. This cannot be inferred by summing rejected probes from 482.
For the original seasonal direction selector the 2022 ledgers are identical;
for the context-only ablated direction selector turning off the gate reduces
return in all four years. Do not disable the filter globally.

All 1,691 gated entries and exits match 481 at the same side and horizon;
fixed-size balances match independent compounding of the archived trades
after removing their original size factor. Absolute returns from 481 are
not the fixed-size control. [All 32 annual accounts](accounts.csv) include
directional win counts, shared entries and nonpositive-score trades;
[booked trades](books.csv.gz) record the actual sequence in each arm.

```bash
OPENBLAS_NUM_THREADS=1 PYTHONPATH=. python -m tools.role483_fixed_gate "$RUN477" "$HISTORY397" "$MARKET" "$BUNDLE" docs/work_plan/479_gross_cost docs/work_plan/481_volatility_direction "$OUT"
```

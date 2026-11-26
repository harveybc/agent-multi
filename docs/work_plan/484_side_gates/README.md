# 484: Which side of the gate changed the account?

Four independent sequential accounts per fold and selector. All use the same
gross-side ranking, within-side horizon choice, q=1 (10% of current balance
as margin), and actual execution costs of 483. An open slot and non-tied
direction are always required. Only the positive net-score requirement changes.
The two reference modes reproduce all 3,633 archived 483 bookings and annual
returns/drawdowns. No fitting on evaluation labels or threshold tuning.

## Seasonal selector, no explicit volatility in side choice

Annual net return (%); bypassing one side still filters the other.

| Year | Both filtered | Bypass BUY | Bypass SELL | Bypass both |
|---|---:|---:|---:|---:|
| 2020 | +1.75 | +0.16 | +5.95 | +4.29 |
| 2021 | -11.87 | -27.15 | -11.87 | -27.15 |
| 2022 | -6.33 | +12.20 | -6.33 | +12.20 |
| 2023 | -2.33 | +3.15 | -2.33 | +3.15 |
| Compounded | -17.96 | -15.56 | -14.57 | -12.07 |

From 2021 through 2023, bypassing SELL alone changes no fills. BUY alone
reproduces the ungated ledger in each of those years. In 2020 SELL alone
improves return by 4.20 percentage points but increases realized drawdown
from 25.52% to 33.07%. In 2021 BUY alone increases realized drawdown from
17.73% to 30.67%; in 2022 from 29.15% to 32.39%.

The added nonpositive BUY trades **lost in aggregate in all four years**.
In 2022 their 33 trades contributed -4.48 log-return points, the 10 newly
accessible positive-score trades +0.17, and excluding the 40 displaced
control trades +22.36. In 2021 the 47 new nonpositive buys cost 26.98 log
points, exceeding the benefit of displacing 36 control trades (+6.35).
These are exact additive differences of 100 times log account growth, not
simple return percentage points. Matching trades cancel at fixed exposure
despite different account balances. No claim that displaced losses could
be identified before entry.

Do not deploy a side bypass on these archived results. A useful next control
is to keep the original directional filter and measure whether simple time
out of the market can reproduce the displacement benefit, using a matched
cooldown established without future exits at decision time.

[Annual accounts and side counts](accounts.csv), [factorial contrasts](effects.csv),
[displacement attribution](displacement.csv), [trade ledgers](books.csv.gz).

```bash
OPENBLAS_NUM_THREADS=1 PYTHONPATH=. python -m tools.role483_fixed_gate "$RUN477" "$HISTORY397" "$MARKET" "$BUNDLE" docs/work_plan/479_gross_cost docs/work_plan/481_volatility_direction "$OUT" --side-reference docs/work_plan/483_fixed_gate
python -m tools.role484_displacement "$OUT"
```

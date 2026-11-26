# 485: A causal cash wait after a rejected BUY

Three independent paths: keep both filters and stay available; bypass only
the BUY filter; or keep both filters and wait in cash on a rejected BUY
while flat. All inherit the same scoring, eligible origins, barrier fills,
calendar, costs and q=1 sizing (margin of 10% of the current balance).
Each annual account starts with USD 10,000. Cash during a wait earns zero
and incurs no spread, commission or financing.

Before each evaluation year, take the median elapsed wall-clock time from
entry to exit of **all** calibration BUY probes at each of the three barrier
scales, not merely selected fills. Calibration exits must all precede the
year boundary. Freeze the resulting three durations; do not optimize them.
On a rejected BUY use its selected scale and cap the deadline at the known
weekly flattening time. Ignore signals through the deadline, inclusive,
without extending the wait. Rejected SELLs do not trigger a wait. The cash
handler does not receive a hypothetical exit or future PnL.

## Seasonal selector without explicit volatility in side choice

| Year | Stay available (%) | Open BUY (%) | Cash wait (%) | Realized DD: available / BUY / cash (%) | Pauses | Executed cash-arm orders |
|---|---:|---:|---:|---|---:|---:|
| 2020 | +1.75 | +0.16 | +3.61 | 25.52 / 25.52 / 25.52 | 4 | 93 |
| 2021 | -11.87 | -27.15 | -21.42 | 17.73 / 30.67 / 25.49 | 60 | 78 |
| 2022 | -6.33 | +12.20 | +18.59 | 29.15 / 32.39 / 21.75 | 53 | 93 |
| 2023 | -2.33 | +3.15 | +16.14 | 27.30 / 28.02 / 21.25 | 17 | 120 |
| Compound of annual returns | -17.96 | -15.56 | +12.14 | | | |

In 2022 the waits were 10/22/28 hours at 6/24/48 h scales before calendar
capping. The full 2022 cash ledger has 2 BUYs and 91 SELLs. Idling can
improve the account without trading the rejected BUY. It does not establish
that these triggers identify impending losses: the exposure and sequence
of entries change. In 2021 the cash rule lost 9.54 percentage points versus
staying available. For the other three selector variants, compounded cash
returns were below their stay-available counterparts (context control
42.15 vs 48.40%, context without vol -23.99 vs -6.43%, seasonal control
5.15 vs 10.62%). This diagnostic uses previously inspected folds.
Do not generalize the rule across selectors. The next discriminating control
is a cash pause unrelated to the rejected BUY, matched for exposure and
calendar, to distinguish timing information from merely trading less.

48 annual accounts, 14,360 distinct evaluation origins. Both reference modes
reproduced all 3,559 archived bookings. Cash intervals never overlap actual
trades; a test changes the rejected BUY's simulated exit and PnL without
changing the cash decision. [Annual accounts](accounts.csv),
[calibration counts and deadlines](calibration.csv), [actual wait intervals](pauses.csv.gz),
[trade ledgers](books.csv.gz).

```bash
OPENBLAS_NUM_THREADS=1 PYTHONPATH=. python -m tools.role485_cash_wait "$RUN477" "$HISTORY397" "$MARKET" "$BUNDLE" docs/work_plan/479_gross_cost docs/work_plan/484_side_gates "$OUT"
```

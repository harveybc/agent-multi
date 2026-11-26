# 482: Did the net-score gate reject good opportunities?

Replay the 481 controls and ablations on the same 14,360 eligible origins
across 2020-23. Select the gross side first, then the highest **predicted**
net score within that side even when every score is negative. Score > 0 is
approved; a side tie is an abstention (none occurred). For each proposal,
evaluate that one selected horizon without using realized outcomes to choose it.
Measure fully costed USD per fixed USD 10,000 account at q=1 (the maximum
allowed by the existing volume contract). These are overlapping probes, not
account profit. Opportunities on exit bars are busy; entry bars are free
before execution. All 1,691 approved/free bookings across 16 books match
the 481 archive on timestamp, side, horizon, score, and net PnL after sizing.

## 2022 seasonal arm, no volatility in the side decision, BUY

| Net score | Slot on original path | Proposals | Wins | Mean net USD at q=1 |
|---|---|---:|---:|---:|
| Rejected | Free | 133 | 83 | +60.23 |
| Rejected | Busy | 1,581 | 748 | -12.77 |
| Approved | Free (executed) | 7 | 3 | -135.75 |
| Approved | Busy | 189 | 81 | -39.43 |

Across all origins, rejected BUYs returned -7.10 on average (831/1,714
wins), approved BUYs -42.87 (84/196). By selected barrier scale:

| Hours | Rejected n | Mean net USD | Approved n | Mean net USD |
|---:|---:|---:|---:|---:|
| 6 | 462 | -23.72 | 10 | -49.35 |
| 24 | 1,032 | +7.86 | 109 | -19.45 |
| 48 | 220 | -42.37 | 77 | -75.17 |

With the slot free, rejected BUYs averaged -127.45 (2020, n=11),
-68.64 (2021, n=169), +60.23 (2022, n=133), and -62.51 (2023, n=34).
Do not disable the filter from the 2022 result. Occupancy comes from the
original trajectory and changes under a different admission rule. The
seven executed BUYs are insufficient to infer a stable effect.

Full breakdowns for **both** sides, input sets, variants and all four years:
[by side](sides.csv), [by side and horizon](horizons.csv), [by availability](availability.csv),
[by both horizon and availability](detail.csv), [denominators](counts.csv).
Counts are proposals; win means strictly positive realized net. Next causal
portfolio comparison, if pursued: replay a gated vs ungated selector with
identical fixed positive sizing in both arms so rejected scores can be traded
without changing the sizing rule between arms.

```bash
OPENBLAS_NUM_THREADS=1 PYTHONPATH=. python -m tools.role482_net_gate "$RUN477" "$HISTORY397" "$MARKET" "$BUNDLE" docs/work_plan/479_gross_cost docs/work_plan/481_volatility_direction "$OUT"
```

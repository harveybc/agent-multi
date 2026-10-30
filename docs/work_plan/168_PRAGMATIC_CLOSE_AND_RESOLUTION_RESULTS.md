# Completed close attribution and temporal-resolution contrast

2026-10-30. Both persistent services completed automatically, exit0, zero
restarts. Attribution finished08:44:31 Bogota; resolution contrast08:59:58.
All4,306 and223 artifact hashes respectively were verified after completion.

## Frozen close attribution

360 baseline accounts reproduced exactly;3,940 one-tick CLOSE-to-HOLD
interventions completed;258 runtime tests passed. Positive delta favors the
original CLOSE, not the intervention. Across all report qualities,1,664
closes improved final profit and2,276 worsened it. These overlapping
interventions cannot be summed into a realizable new policy return.

With ideal reports alone,305 closes produced157 profit improvements and148
losses relative to suppressing that close for one tick. Considering return
and drawdown jointly:124 CLOSE-dominant cases,106 HOLD-dominant and75
tradeoffs. This is different from the local passive-position metric in163.

[Full attribution table](../audits/evidence/pragmatic_close_full_20261030/RESULTS.md),
[numerical report](../audits/evidence/pragmatic_close_full_20261030/REPORT.json),
[completion hashes](../audits/evidence/pragmatic_close_full_20261030/COMPLETE.json).
Per-case native ledgers and traces remain in the immutable run archive.

## One-variable resolution contrast

1,296 accounts, six frozen policies, new paired worlds,261 runtime tests.
No coefficients selected or adjusted on these results. Adding1h/2h close
alternatives was helpful in some settings, not a universal correction.

For original policy59a75d73 versus its1h/2h variant4deff71a at noise1:

| Metric | Original | With1h/2h |
|---|---:|---:|
| Mean account return % | 5.24029 | 5.39285 |
| Worst account DD % | 1.35675 | 1.27394 |
| Mean daily annualized Sharpe | 19.39947 | 19.08984 |
| Orders per market week | 16.97222 | 15.50000 |
| Buy / sell | 300 / 311 | 265 / 293 |
| Winning trades % | 72.50409 | 73.11828 |
| Early closes | 503 | 421 |
| Locally harmful early closes | 289 | 220 |

Under ideal reports, return falls for both variants:6.38424->6.25705% and
7.11678->6.73553%, with unchanged worst DD. Under noise2,59a75d73's variant
falls2.20268->2.05940% and worst DD rises5.02423->5.55764%. Causal-prefix
diagnostic returns remain negative; this reporter is not a trained RL model.
No strategy promotion follows from this mixed comparison.

[All30 confirmation rows](../audits/evidence/pragmatic_resolution_full_20261030/RESULTS.md),
[numerical report](../audits/evidence/pragmatic_resolution_full_20261030/REPORT.json),
[completion hashes](../audits/evidence/pragmatic_resolution_full_20261030/COMPLETE.json).

These are synthetic strategy/output engineering results, not financial
VALIDATION or TEST. The next diagnostic is[169](169_PRAGMATIC_PLAN_EXECUTION_GAP_PLAN.md):
independently replay the committed CLOSE/HOLD plans and separate valuation
error from changes in hourly policy continuation. No floating-selection or
learned-model training phase has been released.

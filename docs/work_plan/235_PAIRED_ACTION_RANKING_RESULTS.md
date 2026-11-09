# Paired Action Ranking Results

Completed 2026-11-09 at 14:31:53 UTC (09:31:53 Bogota), in 26m12s.
Both the automatic queue and report service exited0. Design:
[234](234_PAIRED_ACTION_RANKING_CONTRAST.md). Runtime `f6a78476`.

Identity: `427ab5cd02858245e021ae98d2ab3a88f694bf6a34379230cfa1e61fb6de4b57`.
Verified collections:193 files in shard0,192 in shard1,157 in the fitting
collection; all four report outputs also verified against their receipt.
The uncorrected control reproduced the previous contrast's native accounts.

## Completed Scope

379 TRAIN states:284 correction-fit anchors and95 disjoint diagnostic anchors.
All four TRAIN years included. Three forced actions per state, one greedy and
four common-random stochastic continuations:5685 native suffix simulations.
Three correction arms each received4096 critic-head updates; the control did
not. Each then received the same1,8,32 actor updates. Actor, representation and
alpha remained fixed during critic correction. No observation feedback added.

All24 annual trading accounts satisfy the activity gate. No checkpoint was
promoted or deployed. Preserved69 and70 source hashes remain unchanged.

## Trading At The Fixed32-Update Endpoint

Percentages below are full-period net account returns, not Sharpe or annualized
returns. TRAIN spans four years; VALIDATION spans one. All fees and business
rules remain as in the matched control.

| Correction | TRAIN return % | VAL return % | VAL Sharpe | VAL DD % | VAL trades | VAL win % |
|---|---:|---:|---:|---:|---:|---:|
| None, matched control |12.24694|12.71370|1.90953|3.62494|208|57.21|
| Chosen action, stochastic |11.82040|12.86938|1.91290|3.64822|209|57.42|
| Three actions, stochastic |12.27152|13.80849|2.09706|3.62493|206|57.77|
| Three actions, greedy |11.57813|14.61633|2.20502|3.62490|204|58.33|

Greedy correction improves VAL by1.902624 percentage points versus the matched
32-update control, but reduces TRAIN by0.668812 points. Against the unchanged
70-input actor, TRAIN improves0.063604 and VAL0.807837 points. At8 actor updates,
the same corrected arm already gives TRAIN11.61787% /VAL14.58354%.

The separate69-input champion remains TRAIN10.79211% /VAL14.59109%:
the new70-input candidate exceeds its VAL return by only0.025236 points. This
cross-input comparison is contextual, not the controlled effect of the repair.

Greedy/32 validation details:

* Net profit1461.63USD on10000USD initial capital;204 trades,3.78/week.
* BUY23 trades (11.27%),15 wins (65.22%),net277.08USD.
* SELL181 trades (88.73%),104 wins (57.46%),net1184.56USD.
* Closures:106 TP,66 SL,26 weekly forced-flat,6 early closes.
* The existing fixed-position diagnostic still marks4 early closes helpful
  and2 harmful. Their count and classification have not improved versus the
  original actor; a general early-close repair is not established.

## Ranking Qualification Remains Mixed

The greedy correction improves its fitting-anchor ranking accuracy from52.58%
to59.73%, and mean greedy regret from0.19310 to0.14437. On the95 diagnostic
anchors, however, ranking accuracy stays50.51%, RMSE worsens0.66917 ->0.70710,
and mean regret worsens0.16274 ->0.18676. These anchors are disjoint from the
additional correction fit, not an independent market holdout.

The paired-stochastic correction also improves fitting metrics while its
stochastic monitor ranking accuracy falls47.98% ->45.96% and regret increases
0.12785 ->0.15348. Its trading result nevertheless avoids the validation loss
of the matched uncorrected actor. Average RMSE, ranking and account performance
must remain separate evidence; the critic-ranking problem is not closed.

Mean stochastic-return standard deviation across the four draws is0.62124;
mean future entropy contribution is0.30268. These are discounted-return units,
not dollars, price noise estimates or profit percentages. Four draws do not
certify the conditional expected value of every action.

## Interpretation And Next Controls

Useful measured progress: evaluating all alternatives at a shared native state
changes the bounded refinement outcome. Paired stochastic calibration preserves
the original validation account; greedy continuation improves it in this run.
This is one frozen parent and one correction/update seed, not a replicated
repair. The greedy arm changes continuation policy AND entropy convention,
as preregistered; neither factor's independent effect has been isolated.

Next: attribute the improved account to exact changed orders, replicate with
new correction/sampling seeds, and isolate reward-only stochastic continuation
from greedy continuation before claiming a general mechanism. Keep correction
fits inside TRAIN and retain all endpoint comparisons. Do not blacklist the
previously observed validation dates or replace the preserved champions on the
basis of this one contrast.

## Evidence

[Trading table](generalization/action_ranking_20261109/TRADING.csv),
[calibration table](generalization/action_ranking_20261109/CALIBRATION.csv),
[terminal receipt](generalization/action_ranking_20261109/QUEUE_DONE.json),
[panel metadata](generalization/action_ranking_20261109/PANEL.json).
Full private artifact root:`sac70-action-ranking-20261109`.
The `reports` directory contains `ACTION_REGRET.png`, `TRADING_RETURNS.png`
and `REPORT.json`; model weights are saved under each fitting arm.

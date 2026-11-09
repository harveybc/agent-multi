# Ranking Replication Results

## Completion And Reboot Check

The complete236 queue finished2026-11-09T16:50:04.950236Z (11:50 Bogota),
in2318.21 seconds (38m38s), before the coordinator reboot at11:54:27 Bogota.
Both worker processes are absent because the bounded queue completed, not
because a replica requires relaunch. Post-reboot verification authenticated
all1190 files in the terminal receipt. Source69 and source70 checkpoint hashes
are unchanged. No model was promoted.

Runtime:c3fb4a69. Terminal identity:
`36b1444e47a4fdf4c9fddd67bffa643786d6f3e0cd8e5aafd351f497687b0e4d`.
Private artifacts:`/var/tmp/sac70-ranking-followup-20261109`.
Each replica includes196 authenticated fit files and automated trading and
calibration CSV/PNG reports. Combined output:PAIRED_TRADING.csv and SUMMARY.json.
Ten arms completed;30 actor checkpoints yielded60 full-period accounts. All
passed the greater-than-two-orders/week activity requirement.

## Fixed32-Update Endpoint

Returns are native full-period account return percentages. Validation DD and
win percentages are not fractions. No favorable checkpoint was substituted.

| Replica | Correction | TRAIN return % | VAL return % | VAL Sharpe | VAL DD % | VAL trades | VAL win % | VAL early closes |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| r19 | Control |13.13166|12.71370|1.90953|3.62494|208|57.21154|6|
| r19 | Chosen stochastic |11.02140|12.26222|1.86012|4.04458|207|57.00483|7|
| r19 | Paired stochastic |12.23434|14.05738|2.08947|3.62493|207|57.97101|6|
| r19 | Paired greedy |12.88458|14.05738|2.08947|3.62493|207|57.97101|6|
| r19 | Paired stochastic reward-only |11.97855|14.66318|2.21142|3.62499|204|58.33333|6|
| r43 | Control |12.56591|12.71370|1.90953|3.62494|208|57.21154|6|
| r43 | Chosen stochastic |9.85523|11.17402|1.66277|4.04274|213|56.80751|8|
| r43 | Paired stochastic |11.44257|11.62603|1.73828|4.04467|209|56.93780|8|
| r43 | Paired greedy |11.36449|12.95110|1.95508|4.04461|208|57.69231|8|
| r43 | Paired stochastic reward-only |11.44257|11.06806|1.65789|4.04465|210|56.66667|8|

Paired greedy validation direction details:

| Replica | BUY trades | BUY win % | SELL trades | SELL win % | Orders/week |
|---|---:|---:|---:|---:|---:|
| r19 |23|65.21739|184|57.06522|3.83333|
| r43 |25|60.00000|183|57.37705|3.85185|

## What Replicated

Paired greedy improves validation versus its matched actor-update control in
both new replicas: +1.34368 and+0.23740 percentage points, mean+0.79054pp.
The original235 seed had+1.90262pp. However, TRAIN falls versus each matched
control (-0.24708 and-1.20141pp), and r43 validation DD rises0.41967pp.

The unchanged70 parent returns13.80849% in validation. Greedy r19 beats this
parent; r43 does not. Thus the new evidence supports mitigation of the bounded
update degradation, not a reliable improvement over preserving the actor.
These are correction-seed replications of the same parent and dates, not
independent-market or independently-trained-parent replication.

Removing entropy from stochastic targets does not resolve the problem:
reward-only validation is14.66318% in r19 but11.06806% in r43. Its paired
mean uplift is just+0.15192pp, with opposite signs. The highest individual
return is not a sufficient basis to adopt that variant. Chosen-action-only
correction loses against control in both replicas.

## Ranking And Early Closing

Greedy diagnostic-anchor accuracy improves in both replicas:
r19:52.5510% ->54.0816%; r43:52.2388% ->55.2239%.
Diagnostic regret improves for r19 (0.17814 ->0.16729), but worsens for r43
(0.16924 ->0.17209). These diagnostic anchors are TRAIN-only held-out fitting
rows, not an independent chronological validation set. No global ranking
repair claim follows from this contrast.

Early-close counts are6 and8. Counts alone do not establish better early-close
quality; the original236 account attribution located its uplift in changed
openings. Do not claim an early-close fix from these replicas.

## Next Scope

Retain all endpoints and source champions. Before further unrestricted
refinement or another feature-removal pass, separate actor-update drift from
critic calibration: use a fixed-parent, matched update-budget comparison that
checks preservation of source policy and native account activity. Keep the
observations, window, strategy and data support fixed. A durable benefit must
be judged against both the unchanged parent and its matched update control.
This report does not launch a new experiment or replace the live policy.

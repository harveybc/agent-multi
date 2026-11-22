# 424-425: where SELL overestimation begins

All 18 saved 423 checkpoints, same managed targets, 2 historical scoring
semesters, 3 seeds and 3 architectures (overlapping cells, not independent
replications). No new encoder training, promotion or 2024 evaluation.
The fit/monitor/calibration/evaluation slices reuse the corresponding 423
time boundaries. Purge H1 outcomes at the July boundary before estimating
any correction. The offline top tenth per week/horizon is a matched-volume
diagnostic, not the live trading rule. Baseline account replays match 423.

Median bias (predicted minus actual bp) in the top slice is +0.79 in fit,
+2.90 in the checkpoint-selection monitor, +2.71 in H1, and +4.02 in H2.
The corresponding all-offer biases are +0.17/+0.18/+0.30/+0.13 bp.
The decomposition is exact per offer: for model probability p, estimated
conditional gain G and loss magnitude L, actual net y and I=1[y>0],
error = (p-I)(G+L) + I(G-y) + (1-I)(-L-y).
Medians of contributors need not sum. In H2 the top-slice probability term
is positive in all 18 cases. A fixed H1 tail-bias offset yielded 0/18
positive active H2 accounts.

The follow-up fits ridge-regularized logistic corrections to the frozen
win logit per horizon on H1, either alone or with log conditional gain and
loss covariates. Gains, losses, checkpoint, fee model and closer unchanged.
For comparing probabilities, the original top set stays fixed; the native
account is separately reranked using the corrected bid and the H1 decile
and positive-EV gates.

|Method|Fixed-top median predicted win rate (observed 33.1%)|Median return|Median trades|Positive and >2 trades/week|
|---|---:|---:|---:|---:|
|Original|42.0%|-8.01%|192|0/18|
|Logit only|35.8%|-3.89%|73|0/18|
|Logit + conditional amounts|32.7%|-1.30%|58.5|0/18|

Closeness of group medians is not per-tick calibration. No corrected
method produced positive matched-volume mean advantage in any H2 cell.
Do not use these readouts for feature removal or infer that the original
data lack exploitable information. Further threshold sweeps are not justified
by this finding.

Reproducers: `tools/role424_sell_bias.py`, `tools/role425_sell_probability.py`
in the 413 CPU environment. Artifacts: `/var/tmp/role424-sell-bias/`,
`/var/tmp/role425-sell-probability/`.

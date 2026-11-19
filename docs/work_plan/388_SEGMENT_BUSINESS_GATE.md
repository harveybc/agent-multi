# 388: behavioral fidelity before augmentation training

One GPU per fixed seed from 387. No new optimizer steps or use of 2024.
Build a TRAIN segment catalog with true pivot-to-pivot displacement and duration,
residual noise/SNR, curvature, thirds of the move, adverse retracement and
its timing, within-segment volatility change, and session. Controls are not
actor inputs. Natural-frequency and stress panels must not be pooled.

Natural panel: simulate a hypothetical flat-price-book entry at each bar open
inside a covered segment, both sides and all four reference horizons. No
windows straddling market gaps or quarters. Use the existing direction/horizon
median distances as fixed price barriers, **not** an emulation of net native
fills. Count an ambiguous dual touch separately; gaps beyond a barrier exit at
open. Compare immediate liquidation at each subsequent surviving bar open vs
holding until protection or horizon. Summarize per entry (not per tick).

Frozen screening: by horizon, side, origin trend and volatility half (median
on real 2020-21 segments), compare synthetic 2020-22 vs real 2023. Tolerance
per statistic: 90th percentile absolute differences among real 2020-22 quarterly
means with at least 20 eligible entries per cell. Need at least four reference
quarters and 20 observations per period. Report both the real-only drift
control and unsupported cells. Do not substitute 2023 synthetic controls for
2023 real data. A breach is evidence to investigate, not a training gain.

Stress controls: mirrored long/short paired reversal and recovery with an
identical prefix, forced barrier hit, ambiguous touch and gap; verify signs
and decision timing without giving the model future information.

## Execution

Four verified CUDA workers (one per seed); 175,492 overlapping entry/side/
horizon combinations per synthetic path. Price-only, not strategy trade counts.
4,114 whole segment templates after removing eight quarter-crossing segments.
The catalog's SNR is the standard deviation of the fitted log-price trend
divided by detrended log-price residual SD (floor 1e-8); not a causal input.
Early-close gain is in pips at surviving subsequent opens against the
no-agent-exit continuation. It is not an oracle close or an after-cost reward.

316/320 mean-statistic cells per seed supported; no supported cell breached.
Four sparse conditional time-to-TP cells per seed had no verdict. Exploratory
empirical-CDF checks at five real reference quantiles: 0/624 synthetic
outside the envelope; 224/624 outside for the deliberate wrong-trend control.
CDF and real comparisons across seeds share historical data, not independent
replications. These are screening envelopes, not significance tests.

4,096 seeded CPU scalar vs CUDA path checks passed after fixing float32
rounding on the two scalar barrier payoffs and rerunning all four jobs.
Stress bank: 1,441 paired price continuations across 757 unique origins
(760 long, 681 short; 114/316/435/576 pairs at H3/6/12/24).
One-bar-old underwater position, common real prefix through current open,
synthetic future translated to this open, one branch hits SL first and the
other TP first. Diagnostic labels only: differing futures are not
distinguishable at the fork. Do not feed labels to the actor or use the
stress sample's selected frequencies as natural market frequencies.

Local receipts and outputs: business388/{GATES.csv,CDF.csv,VERIFICATION.json,
STRESS_PAIRS.parquet,real_catalog.parquet,*_receipt.json}. No use of 2024
and no new training weights. Next: matched augmentation training contrast.

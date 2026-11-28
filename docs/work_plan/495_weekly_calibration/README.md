# 495. Causal weekly bias correction for short exits

Question: can the frozen 494 scores use newly settled counterfactual outcomes
to correct their USD level without losing useful ranking? No actor or forest
updates, no changes to training windows, no decisions after 2023.

Use only 2021-HOLD / 2022-CLOSE disagreements on short positions. Restore
the 494 H1 fits without refitting. For a decision week starting Sunday at t,
use observed sessions starting in [t-28 days,t-7 days], whose weekly accounts
have settled by t. Within each eligible prior week average the realized
close-minus-hold terminal net less the original forecast; then average those
weekly residuals equally. Add that scalar to every original score in the
decision week. Veto closing when corrected score is negative. With no recent
eligible session the shift is zero. The history-only comparison uses the
average realized advantage itself, with zero meaning no veto. Model inputs
and outcomes for other decisions stay unchanged. This experiment uses
counterfactuals available by historical replay after settlement, not just the
cash of the action actually taken; deployment would need both simulated paths.

Frozen policy reproduces the 494 test counts, AUC, USD bias and actor delta.
Replacing all current and later week labels with NaN leaves every correction
unchanged. The same shift preserves ranking within each seed/week; shifts in
different weeks can change pooled ranking. H2 is already studied and not a
new confirmatory period. 131 short states over 16 observed weeks; 137
eligible seed-state recommendations from 63 distinct states (87 harmful,
50 beneficial). Economic gains average over all sampled short states,
equally by week, after averaging seeds. Bias, MAE and AUC concern eligible
cases only. Intervals are descriptive paired circular block bootstraps
(two consecutive observed weeks, 10,000 draws).

| Input | Frozen to corrected bias (USD) | Frozen to corrected MAE | H2 pooled AUC, frozen to corrected | Gain vs frozen | Gain vs hold on disagreements | Gain vs original actor |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Margin | +31.25 to +7.16 | 38.15 to 39.67 | .651 to .314 | +1.63 | -1.89 | -2.72 |
| Margin and position | +38.81 to +5.67 | 43.30 to 38.64 | .640 to .319 | +1.53 | -1.99 | -2.83 |

The margin gate vetoed 70 (35 harmful, 35 useful). The position gate vetoed
67 (33 harmful, 34 useful). History alone gained 1.48 vs frozen; adding the
score gained just 0.15 or 0.05 over that control. The paired interval vs
frozen includes zero for both gates. Shrinking the average bias did not
establish better control. Do not promote, change the actor, or tune the
lookback on this semester.

[Summary and intervals](summary.csv), [auditable weekly offsets](offsets.csv),
[weekly paired economics](weekly.csv), [runner](../../../tools/role495_weekly_recalibration.py).

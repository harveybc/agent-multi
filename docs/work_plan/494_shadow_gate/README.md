# 494. Could the harmful 493 disagreements be recognized before acting?

Read-only diagnostic on the two [493](../493_disagreement/README.md)
transitions: NOP -> SELL while flat; HOLD -> CLOSE while short. Compare
identical model files (verified against 492 manifests) on the pinned 490
panel. No change to the original actor, training windows or strategy.
No new GPU training or 2024 decisions.

For each seed and role, calibrate on eligible decisions in January-June
2023 only. Target: net USD advantage of the 2022 recommendation over the 2021
alternative under the same subsequent greedy actor. Inputs are **available
at the decision**: 2022 action-value margin (SELL minus NOP or CLOSE minus
HOLD), optionally signed distances to SL and TP in entry-sigma units and
bars held for shorts. Weighted standardization and ridge regression (penalty
0.1, sample weights sum to one, proportional to inverse number of sampled
role decisions that week). No time, labels or future outcomes as features.
In July-December only, veto if predicted USD advantage < 0 and use the
2021 action instead; otherwise keep 2022. All other disagreements and
agreements retain 2022. Never/always veto refer only to this transition.
Parameter fits, scale and intercept by seed are in [fits.json](fits.json).

| H2 gate | Predicted / observed advantage on eligible cases, USD | Vetos / eligible (state-seed) | Gain vs never / always veto, USD per role decision |
| --- | ---: | ---: | ---: |
| Short, margin only | +21.47 / -9.79 | 0 / 137 | 0.00 / -3.52 |
| Short, margin and position | +29.03 / -9.79 | 0 / 137 | 0.00 / -3.52 |
| Flat, margin only | -1.81 / -12.88 | 114 / 187 | +2.20 / -1.14 |

Short: 137 recommendations over 63 distinct eligible states (three seeds
on overlapping data); 87 were harmful. Mean per-seed tie-excluded AUC for
benefit was 0.651 without context and 0.640 with it. Some ranking information
was present in this sample, but it did not yield a useful dollar threshold.
Flat: 187 recommendations over 90 eligible states. Of the vetoed cases 61
avoided harm, 32 forwent a benefit and 21 tied. AUC 0.525. The gain vs never
veto had a descriptive paired weekly 95% interval [+0.53,+3.93]; the
difference vs always veto was [-2.43,+0.03]. The gated flat policy remained
4.75 USD/decision below the original actor; the short policy 4.36 below.
Do not promote either gate or infer that all sell entries should be vetoed.

Reconciliation: eligible counts and weekly-weighted economic contributions
match 493. Conditional numbers average over eligible cases with the original
week weights; gate gains average across all role decisions. Do not sum
these as portfolio P&L. [Full summary](summary.csv), [per seed](seed_check.csv),
[runner](../../../tools/role494_disagreement_gate.py). H2 outcomes were
previously inspected to choose the question: a temporal transfer diagnostic,
not an untouched confirmatory holdout. Do not tune the cutoff to this half.

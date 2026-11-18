# 375: functional actor checkpoint interventions

Offline CPU diagnosis on two active374 arms; no campaign, deployment or checkpoint changed.
Use epoch60 for equal gradient budget (354354 updates), and the previously selected
early checkpoint (epoch4 on six months/seed19; epoch17 on48 months/seed7).
The latter is validation-informed, so all hybrid results are exploratory.
Check snapshot SHA256 against HISTORY; reconstruct the identical scaled windows
and replay both intact endpoints before interpreting a swap. On48 months at epoch60
CPU vs recorded GPU signal counts differed by +1 NOP/-1 BUY in TRAIN, zero in VAL;
account returns, side counts and P&L, close reasons and costs agreed exactly.
The six-month intact endpoints agreed in returns, trade counts, signal counts and close reasons.

24 actor configurations (two directions x two arms x intact/five individual swaps),
each with full TRAIN and validation native accounts. The reciprocal swaps test
context dependence: hybrids do not imply separable or additive module utility.
Critics are not used in greedy inference, and were not modified or ruled out
as the origin of the learning pressure. No target or test rows used for fitting.

| Epoch60 policy intervention | 6mo TRAIN | 6mo VAL | VAL trades | 48mo TRAIN | 48mo VAL | VAL trades |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Intact | 264.78 | -46.79 | 1403 | 551.47 | -26.39 | 536 |
| Early attention | 33.42 | -33.79 | 1014 | 74.28 | -3.91 | 277 |
| Early branches | -13.49 | -53.91 | 1285 | -59.33 | -22.63 | 797 |
| Early core | 81.17 | -31.61 | 1211 | 70.46 | -16.34 | 464 |
| Early fusion | -6.02 | -41.80 | 1447 | -85.81 | -45.55 | 1046 |
| Early head | 35.54 | -54.79 | 1468 | -78.31 | -23.92 | 707 |
| Average trainable actor weights at40/45/50/55/60 | 251.10 | -50.92 | 1357 | 218.39 | -20.73 | 557 |
| Early attention and core together (post-hoc) | 9.36 | -22.97 | 721 | 29.83 | -5.59 | 334 |

Units: cumulative account return in percent, not annualised or comparable
across TRAIN durations. Evidence localizes part of the late policy degradation
to temporal representation changes in these two runs. It does not identify an
incorrect SAC update, prove the representation unnecessary, or establish
that freezing it during optimization would yield these hybrid returns.
Fewer trades contribute a plausible alternative to improved discrimination.
The early intact VAL benchmarks are -6.35% and +0.50%, respectively: even
the joint rollback loses to both early references. Do not promote a hybrid.

**Concrete tooling fix:** classify the full actor/critic encoder.post_fusion
patch adapter under attention in module_key; prior logs pooled it under other.
This affects diagnostic aggregation only. Two actor/critic regression tests.

**Next falsification:** paired continuation from a TRAIN-predeclared epoch on
each seed, same replay/RNG/Adam and update budget, control versus freeze or
lower LR only on actor attention+core. Keep critics trainable. Compare native
net return, trades, exposure and net per trade with an activity-matched control
on a chronological internal block. Do not choose new hyperparameters on2024.
Leave374 untouched so the paired history-size comparison remains interpretable.

Reproduction: from the repository with its pinned environment and GPU hidden,
run docs/work_plan/generalization/weight_intervention375.py ROOT CONFIG MONTHS EARLY 60.
ROOT contains copied seed_N/months_M_seed_N/{SPEC,HISTORY,QUALIFICATION}.json
and the hashed weight snapshots. For the two additional interventions use
--candidate average or --candidate paired_rollback. Private full receipts are
results_M.json, average_M.json and paired_rollback_M.json under the analysis root.

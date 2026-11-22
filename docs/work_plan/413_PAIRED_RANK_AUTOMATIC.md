# 413: paired opening rank, abstention, and additional real folds

Exploratory TRAIN only. Keep the 208-dimensional encoder, independent role heads,
business clock, costs, weekly opportunity and learned closes fixed within each
seed and evaluation year. No phase-2 feature removal, 2024, Dragon or broker.

1. For each of the 28 pairs of side/horizon alternatives fit ridge to the sign
   of their realized payoff difference on identical origins (ties zero). Rank by
   mean predicted pairwise preference, enforce antisymmetry. This is an ordinal
   preference, not expected money. Never authorize opening from its sign alone.
2. Use 2022 to fit ranks for 2023. Separately calibrate the picked expert's net
   advantage vs NOP on purged H1 of the score year using rank scores and winner
   identity. Evaluate H2 only. Compare same-tick routing at the original opening
   frequency and with the new gate, plus NOP and the native direct 409 baseline.
3. Automatically produce 2020 and 2021 chronological managed outcomes with
   preceding four-year role-head fits, 3 seeds, 4 roles, pinned 408 method.
   Never reuse a head that saw that scoring year. Retry bounded transient process
   failures; resume completed heads. Stop on integrity failure, do not relax it.
4. Repeat on 2023 with 2020-22 rank training, and a separate chronological
   2022 check with 2020-21 training. The same past-year 409 baseline controls
   each evaluation year. Positive gate: return >0, above baseline, >2 trades
   per native H2 week and positive mean selected payoff for every seed in
   both years. Positive means replication candidate only; otherwise reject.
   Additional examples change the historical mix, not just sample count.

State: `/var/tmp/role413-pair-rank-v1`. All outputs are CPU research artifacts.
One command: `bash tools/run_role413.sh`; also used by the user service.

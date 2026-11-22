# 399: historical SAC continuation and independent generation checks

The cache qualification in 397 changes execution cost, not the policy. Extend
that exact 61-input run and measure whether late training surpasses early
validation transients. Run two histories and two stochastic training seeds
from the same weight checkpoint, not four independently initialized models.
Both arms use one chronological account and real validation 2024. No TEST,
synthetic replay, live trading, DEAP, or automatic promotion.

| TRAIN | Seed | Worker role | Start |
|---|---:|---|---|
| 2016-23 | 7 | RTX 5090 | Restore qualified epoch 1 including replay, optimizer, RNG and patience |
| 2016-23 | 19 | RTX 4090 | Common frozen start |
| 2020-23 | 7 | RTX 5070 Ti | Common frozen start |
| 2020-23 | 19 | RTX 4070 | Common frozen start |

Admission uses the hashes in the historical contract, per-device cached vs
uncached paired updates, and the expected initial real-account trade count
and return. No cache on validation. Hash frozen parameters at every epoch.
For the 5090 extension, stop on any discrepancy against the recorded original
epochs 2-4 (metrics, decisions, steps, update counts and losses). Retain
the original run and its receipts separately.

Use normal early stopping, **not** the fixed-budget pilot override that
suppresses patience. Source design: cap 2000 epochs, validation return/week,
activity requirement, start epoch 20, patience 20, min delta 1e-8. Compare
curves at common epochs and common cumulative updates: the longer history
also gets more updates per epoch. GPU architecture is not crossed within
seed and duration; do not attribute all differences to data volume alone.

Execution: `tools/series399.py`, `tools/series399_launch.sh`. The read-only
observer (`tools/series399_observer.py`) gathers per-epoch plots' input
statistics and service state every five minutes under
`/var/tmp/sac399-series/receipts`. Native histories and plots remain on their
worker. Resume only from the same arm, immutable inputs and training contract.

Synthetic generation is still a separate, partial calendar with one writer.
Audit atomically published blocks on three machines; each checks a disjoint
set of hashes, timestamps, candle constraints, all feature values and the
original producer parity receipt. Measure endpoint amplitude and the location
of the directional extreme on complete scripted segments; exclude the first
leg without the preceding seed candle and the open tail. These are descriptive
checks, not a ZigZag re-detection test, generator release, or an ETA for a full
four-year synthetic prefix. Receipts: `/var/tmp/sac399-series/audit/LATEST.json`.

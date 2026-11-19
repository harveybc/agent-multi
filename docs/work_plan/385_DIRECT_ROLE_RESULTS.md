# 385: economic argmax classifier rarely abstains

[Frozen protocol](385_DIRECT_ROLE_PROTOCOL.md). Three local fits, single seed,
no external model calls. New fit exposures: 9707 paired openings, 8559 long
and 8613 short references. These are overlapping decisions from the same
market history, not independent new bars. Cold CPU reload reproduced 2023
class counts and log loss; original account reproduced pinned 379 net.

| Role | Best / total epochs | Accuracy vs fit-majority | Log loss vs prior | Mean regret bp vs fit-majority |
|---|---:|---:|---:|---:|
| Flat |64/144|43.55% / 43.49%|0.9366 / 0.9392|36.85 / 37.04|
| Close long |23/103|49.79% / 48.11%|0.6955 / 0.6963|17.04 / 16.60|
| Close short |3/83|52.45% / 49.44%|0.6921 / 0.6932|17.13 / 18.43|

All stopped on patience 80. Flat multiclass Brier was worse than the prior
(0.5853 vs 0.5809). No claim of reliable economic confidence.

| Opener / closer | Return % | Max DD % | Buy / sell | Wins | Agent closes |
|---|---:|---:|---:|---:|---:|
| Source / source |+3.09|4.02|34 / 170|123|19|
| Direct / source |-31.85|32.97|333 / 534|443|74|
| Source / direct |-1.35|5.47|35 / 188|110|87|
| Direct / direct |-44.94|45.73|566 / 893|619|873|
| Direct / 379 surface |-53.53|53.71|678 / 1101|767|1308|

On 4716 fixed 2023 flat states, optimal hindsight NOP/BUY/SELL counts
437/2228/2051; most-probable-class decisions 41/1830/2845. On the actual
direct/direct account, NOP occurred on only 13 of 1472 eligible flat
decisions (source/source: 3581/3785 on its own path). Both directions
occur; neither trade count nor close frequency implies useful decisions.

The argmax label discards the size of the win/loss. Even in fit the mean
paired BUY/SELL advantages were -5.59/-4.64 bp against waiting, although
only 818/9707 hindsight winners were NOP. On internal 2023 those advantages
were -3.09/-6.08 bp; the clairvoyant max averaged +30.96 bp. These are
individual interventions, not achievable compounded returns. The artifact
shows overtrading, not that all prediction, RL or reference positions fail.
Fixed one-decision-old close references differ from older on-policy exposure.
The common source encoder is not chronologically independent of these years.

Retain the incumbent. Next bounded diagnostic: compare the argmax-class
opener with expected net advantage per action on the same paired samples;
NOP if neither conditional expected advantage beats zero, with no tuned
2023 threshold. Freeze source closer first. Report net, drawdown, NOP and
both sides; no feature selection, live change or external comparator.

Private artifacts: direct385-data (manifest sha256
623f19664da3775df54fadbc94285b20f958b65163ecc8c0d3181df98ece22c0),
direct385-fit (weights, curves, traces), direct385-account.json sha256
fbcc186bf8fe9f88dd751fa4afb4358ba22b88d88c08e7d0c210eef3883976c2.

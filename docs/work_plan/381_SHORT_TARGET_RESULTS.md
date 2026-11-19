# 381: the horizon is not the first repair to try

[Protocol](381_SHORT_TARGET_PROTOCOL.md). Frozen heads and native accounts,
reset annually. 2020-22 development, no retraining. 135 quarter-sampled
short ticks from 644 eligible ticks; 55 table-CLOSE/scalar-HOLD disagreements
(all eligible, 11 represented quarters). 13 ticks overlap: 177 distinct
paired snapshots. Counts are decisions, not independent trades.

| Account target | General: scalar/table regret R | Disagreements: scalar/table regret R | Disagreements favor CLOSE/HOLD/tie |
|---|---:|---:|---:|
| A: source, 24 decisions |0.288 / 0.322|0.228 / 0.237|27 / 26 / 2|
| B: scalar, 24 decisions |0.173 / 0.168|0.133 / 0.208|24 / 29 / 2|
| C: scalar, annual settlement |0.174 / 0.169|0.133 / 0.207|24 / 29 / 2|

R = foregone ending equity divided by original stop risk. Compare heads
*within* a row: changing the target changes the opportunity magnitude.
General A->B: 30/135 strict sign reversals; B->C: 0/135.
Disagreements A->B: 5/55; B->C: 0/55. Of the 29 harmful table exits
against C, 25 were already wrong against A and 4 were correct against A.
This distinguishes a policy-continuation mismatch from incorrect head
choices relative to their own target; it does not establish why those
choices were incorrect or that the market could predict the hindsight sign.

Against C, table minus scalar regret: general -0.0053 R (quarter-block
95% bootstrap interval -0.0374 to +0.0244); close-disagreements +0.0737 R
(-0.0375 to +0.1603). Neither establishes a superior learned head. The latter
sample deliberately conditions on both heads' decisions. In that subset,
closing was beneficial in 3/11 decisions in 2020, 14/28 in 2021, 7/16 in 2022.
The 2023 attribution is not pooled here. These years contributed to fitting
379; this diagnostic is not a profitability or generalization test.

Next: a paired chronological development experiment with the same 71 inputs,
same features, heads and penalty. Fit on earlier years only. Contrast old
artificial position pairs vs actual reachable positions, and source vs scalar
24-step targets (2x2). Generate fit and evaluation states from fixed policies;
evaluate each fitted closer on identical later-year positions and a complete
native later-year account, with the unchanged closer as control. Do not use
2023 to select or tune, change the early-close threshold or start floating.
This is a closer comparison conditional on the frozen upstream model, not a
walk-forward refit of that model.

13 focused tests passed. All A labels matched the legacy fork calculation.
Individual branches, entry identifiers and original-order advantages are in
the private ROWS ledger. SHA256:
46dca7ff1fe17673c6c28b93a84ffc4fe428ed9f3bf3efa3b960ca7b28be71f1
(ROWS); 886900e6502056a5adfd2dd404052e28eb6b774443c69248b38583c38de1ef42 (RESULT).

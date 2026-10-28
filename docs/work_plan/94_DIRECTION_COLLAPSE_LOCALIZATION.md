# Direction collapse: measured localization, not a learning-root-cause claim

Date: 2026-10-26. Complements the C4 learning qualification and systematic
market matrix. No checkpoint on disk, financial recipe or running worker was
changed by these probes. All interventions ran on CPU on an in-memory copy.

## Hypothesis: BUY evaluated first hides a better SELL

The active path is `GroupedTCNPolicy` -> `SeparateHeadOrderPolicy` ->
`OrderDistribution` -> `HybridOrderEnv` -> pinned `BarEnv`.
The categorical distribution masks illegal commands jointly and deterministic
prediction chooses the largest probability. It does not accept the first
positive direction. The mask enables BUY and SELL together when flat.

Exact ties do favor the lowest index (NOP, then BUY before SELL). That is a
real declared tie behavior, but no top-score tie occurred in any of the 23994
valid windows inspected for the planted seed-7 checkpoint, so it does not
explain its observed directional collapse.

The native head-to-engine integration witness gives BUY a positive score of
1 and SELL a higher score of 2 while giving forbidden close heads even higher
scores. The actual categorical policy chooses SELL, and the actual native
engine opens direction -1. This passed with the 37-test matrix suite.

## Checkpoint probe

Authenticated checkpoint: planted C4, seed 7, tensor digest
`ff78c11b52df95c5586bc5d0215929a554cacfbfc472034acdc52327180d1359`.
For each valid window, probe the hypothetical flat-state choice separately
from the trajectory's actual position. The inspection is not another fit.

| Measure | TRAIN | INNER_A |
|---|---:|---:|
| Valid windows | 21217 | 2777 |
| Flat BUY argmax | 21172 | 2777 |
| Flat SELL argmax | 45 | 0 |
| Flat NOP argmax | 0 | 0 |
| Exact top-score ties | 0 | 0 |
| Mean probability NOP / BUY / SELL | .39865 / .53220 / .06915 | .39904 / .53532 / .06564 |
| Choices changed by zeroing all inputs | 45 | 0 |
| Choices changed by reversing planted signal | 29 | 1 |
| Policy tanh units with absolute activation > .99 | 0% | 0% |

INNER_A contains 1392 positive and 1385 negative planted-signal values.
BUY exceeds SELL's logit on all 2777 windows, by at least 0.38769. Thus it is
not a positive-only injected signal or a tie-breaking artifact. The features
do change probabilities; almost none of these changes cross the action
boundary. This is not a proof that all input gradients are zero.

## Intervention at the command head

Swap only output rows BUY/SELL and CLOSE_LONG/CLOSE_SHORT on the in-memory
copy, keeping NOP, observations, masks, engine, period, costs and risk outputs
unchanged. Native INNER_A evaluation now executes **66 sells and zero buys**,
versus 66 buys and zero sells originally. Source checkpoint hash remains
unchanged. This rules out an execution path that always forces purchases
regardless of the winning head.

The swapped diagnostic gives net +38.75 USD and Sharpe .16245, with zero
agent closes, versus original -1383.76 USD and Sharpe -.96972. This is NOT a
selected strategy or generalization result: it is a post-result intervention
used solely to locate the directional routing. It still fails the required
behavior contract and cannot be promoted.

## What is established and what is not

The immediate cause of the all-BUY execution for this checkpoint is the
learned categorical ranking, not a first-valid-BUY branch, forbidden SELL
mask, SELL decoder failure, exact ties or measured tanh saturation. Why PPO
learned that ranking is not established by this diagnostic; it requires
controlled learning/credit-assignment experiments. Correct synthetic routing
does not close that issue, and cannot justify continuing financial selection.

Private evidence includes the authenticated source receipt, per-period probes,
swapped-head trades/equity, immutable source check, script and closure hashes.
Probe script SHA256:
`b7ad197406f6dbd9b3286270c9ef4d017c8d56ef5d637b19fff7979244aa3436`.

C4 subsequently closed all nine units. Each planted seed failed the positive
learning criterion; mean INNER_A Sharpe was -0.9697229641 in all three arms.
The synthetic matrix passing cannot override that separate failed criterion.

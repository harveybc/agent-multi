# C4 order lifetimes and parameter-boundary diagnosis

Date: 2026-10-27. Read-only follow-up to plans 94 and 102-103.
No fit, simulator change, policy replacement or financial admission.

## Scope and authentication

Inspected all nine C4 units: original/planted/permuted, seeds 7/19/43.
Each TRAIN and INNER_A trades/orders file was authenticated against its RESULT
artifact digest (36 files). The inspected engine environment, volatility helper
and C4 signal producer match the source hashes in the C4 spec.
INNER_A is 2023-H2 internal validation, not annual 2024 VALIDATION or TEST.
These are historical failed-policy results, not results of the new quote grid.

## Measured behavior

All nine INNER_A trade files have the same SHA256:
`c2001d09a2cb95b5f32d94045430b502ac225353b71405e7693b52c36bde95a7`.
Eight TRAIN files also match exactly; planted seed 7 is the sole exception.
Identical trajectories are not nine independent confirmations of a policy.

| Measure | INNER_A, every unit | TRAIN, planted seed 7 |
|---|---:|---:|
| Completed orders | 66 | 496 |
| BUY / SELL | 66 / 0 | 493 / 3 |
| Agent closes | 0 | 5 |
| SL / TP / forced flat | 19 / 23 / 24 | 155 / 178 / 158 |
| Median elapsed holding hours | 26.5 | 26 |
| Minimum / maximum elapsed hours | 3 / 84 | 1 / 96 |
| Exits in the entry bar | 0 | 2 |
| Elapsed duration <= 1 hour | 0 | 4 |
| Median SL distance from fill, pips | 65.1555 | 71.4297 |
| Median TP distance from fill, pips | 59.5373 | 65.3382 |

Pip convention here is 0.0001 for EURUSD. Intrabar exits use the historical
engine's end-of-bar timestamp proxy: these hours are not measured tick-level
latency. Bar identity is reported separately to avoid that confusion.
The five planted-seed-7 TRAIN agent closes lasted 8, 2, 1, 6 and 1 elapsed hours.

INNER_A counters are OPEN_LONG 66, HOLD_LONG 2095 and NOP 616 for every unit.
The NOP total does not distinguish voluntary abstention from an entry-disabled
business mask; it must not be described as learned risk avoidance. Plan 94's
hypothetical flat-state probe independently found zero NOP argmax choices.

This does not support an immediate-close/overly-tight-stop explanation of the
historical C4 validation trajectory. Its measured anomaly is the opposite:
BUY followed by HOLD until an automatic protection or calendar exit.

## What the model could and could not choose

The pinned budget has `learned_risk=false`, reward `equity`, gamma 0.99 and
102400 training steps. TP and SL use the midpoint of their feasible ranges;
order fraction is 1.0 of rel_volume 0.1 of balance, on margin at leverage 50.
Those three quantities are not learned by the C4 policy.

Configured TP bounds are [0.25, 3.0] sigma and SL bounds [0.5, 3.0] sigma.
The feasible intervals intersect these bounds with minimum distance 0.00005
and effectiveness ceiling 1.5 times the range scale. Actual interval widths
and binding-bound frequencies have not been reconstructed by this ledger read;
the observed fill-to-level distances are not estimates of those widths.

NOP is a categorical command, always legally available in the mask; it is not
a narrow interval on a quantized continuous output. With an open position it
means HOLD. OPEN_LONG and OPEN_SHORT become legal together when flat and the
entry gates permit them. There is no configured multi-hour minimum hold in
the inspected engine: discretionary close executes at the next available
decision's following open; automatic protection can close in the entry bar.
Missing valid observation windows can delay the next policy decision while
the engine continues with HOLD, which needs separate opportunity accounting.

## Hypotheses and next discriminating checks

The planted channel is `close[b+4] - open[b+1]`, four observed bars. The sigma
used for protection is a smoothed 24-bar realized-volatility scale. Combined
with median holds of 26.5 hours, this is a horizon mismatch worth testing, not
a demonstrated explanation of the learned BUY preference. Future four-bar
movement is not a complete SL/TP-path oracle and does not guarantee a winning
trade under every protection, spread and calendar constraint.

Keep frozen runs untouched. Before another learning claim:

1. Count free-choice NOP versus mask-forced NOP, positioned close opportunities
   and automatic HOLD gaps, with actual decision timestamps.
2. Reconstruct feasible TP/SL widths, bound hits and cost-to-distance ratios
   separately from learned output distributions and resulting holding times.
3. Qualify an explicit planted opportunity with known feasible net advantage,
   including SELL and early-close opportunities, on the actual reward/step path.
4. Contrast protection/horizon settings one at a time, holding inputs, seeds,
   support and learning settings fixed; do not silently widen the running grid.

An earlier zero-cost policy did open LONG and close at the next permitted
decision, exactly matching a blind rule (plan 83). That different campaign
cannot be conflated with C4's configured-cost LONG/HOLD behavior. Both are
documented degeneration modes; neither is evidence that market inputs lack
signal or that the strategy is optimal.

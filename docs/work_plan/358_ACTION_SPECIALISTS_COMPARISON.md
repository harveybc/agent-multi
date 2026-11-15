# Action Specialists: Conditional Pragmatic Comparison

Owner direction2026-11-15. Preserve the current training-reliability priority.
This is a documented alternative, NOT a launched experiment or phase2 closure.
Use the same selected71 inputs, encodings, temporal representation and pinned
native engine. Do not reopen feature engineering or the game-theory mechanism
derivation as a prerequisite for this pragmatic comparison.

## Reusable Evidence

[123](123_ZIGZAG_K3_300_RESULTS.md) records the optimized3/6/12/24h design
references, K=3 future events and300-generation searches. [124](124_ZIGZAG_PARAMETER_EFFECT_AUDIT.md)
records that deviation was not an identified effective optimum. These are oracle
opportunity scores, not achievable learned-account results. Authenticate original
artifacts/data/calendar identities before reusing them; do not invent parameters
or silently fall back to earlier superseded next-pivot/time-cap semantics.

ZigZag is TRAIN hindsight reference for targets/rewards and diagnostic scoring,
NEVER an observation, live input or future-dependent decision-time risk formula.
Its optimized ideal exits quantify an opener's opportunity, not a live closer.
Keep this expert-guided/hybrid comparison distinct from SAC without expert
guidance. It need not be named RLHF unless actual human feedback is involved.

## Minimal Functional Design

| Role | Training/evaluation reference | Runtime responsibility |
|---|---|---|
| Open BUY | At each admissible tick, best executable long result among the next K configured events, with adverse path and duration | Propose long opportunity versus abstention |
| Open SELL | Same calculation for short, at the SAME origins, without buy-first preference | Propose short opportunity versus abstention |
| Close BUY | Close-now versus best later feasible liquidation and passive protection for an existing long | Advise closing/holding the existing long |
| Close SELL | Mirrored position-conditioned comparison | Advise closing/holding the existing short |
| NOP/HOLD | Value of waiting/continuation in the applicable account state, including costs and forgone opportunities | Compete with entry when flat; hold when occupied |

Start with four role heads plus an explicit, testable abstention rule; a fifth
learned NOP specialist is a contrast, not an assumed requirement. Decide its
target and calibration before implementing it. NOP is not assigned an arbitrary
bonus or punished simply because an oracle trade exists. Cash increment zero
while flat does not imply zero value of future opportunities.

For openings retain negative, weak and costly opportunities as well as winning
ones, and score both directions at every eligible origin. Do not discard losses
or define all training origins as trades. Use the SAME native execution/cost/risk
units across heads and horizons. Fit normalization and calibration on declared
TRAIN partitions; require complete target support within each partition.

For closes, the owner's previous-optimal-entry scenario is one family, not the
whole population. Include nonoptimal entries, different ages, protected profit,
SL/TP distances and SL-first/TP-first/gap paths. Restrict the liquidation oracle
to executable exits before the first actual protection/deadline; never compare
against recovery after the position would already be stopped. Score both loss
avoided and profit sacrificed, plus regret to that feasible optimum. Reuse the
counterfactual-close lessons in129 rather than teaching unconditional early exit.

No unapproved position-state feedback goes into the market encoder. A market-only
scalar close probability cannot distinguish two positions in the same market.
Specify a market-report curve/surface that the controller evaluates against the
actual position, or explicitly resolve an alternative interface before training.
The controller keeps the business state; this is not a new model observation.

## Pragmatic Arbitration, Not A New Unspecified Mechanism

When flat, compare calibrated net opportunity/risk and abstention on a common
scale. When occupied, only HOLD and its applicable closer may act. Preserve
single-position, costs, entry calendar, mandatory flat and no implicit reversal.
SL/TP/volume use authenticated horizon profiles and causal current quantities,
never future per-trade extrema. Known ZigZag optima remain a separate oracle
benchmark. Do not mix oracle entries with claimed learned live-close performance.

Keep thresholds, uncertainty penalties and any persistence/hysteresis configurable
and experimentally calibrated within TRAIN, with declared ranges and receipts.
No hard-coded confidence threshold or guessed default risk coefficient is
admitted. Unit tests cover ties, no opportunities, conflicting directions,
mirrored markets, noisy close signals and two positions with identical market
observations. Account tests compare sequential complete trading, not a sum of
overlapping hypothetical trades. Arbitration is an empirical baseline, not a
claim of Pareto optimality or Nash equilibrium.

## Comparison And Entry Gate

1. Finish the current training diagnosis and qualify the evaluator, including
   sufficient training budgets and seed variability. Preserve selected299.
2. Freeze target definitions, output dimensions, risk profiles, NOP semantics,
   capacity budget and arbitration protocol; validate with ideal reports first.
3. Compare the existing categorical SAC with the composite on identical support,
   initialization policy and declared comparable training budgets. Report model
   parameter counts and compute, both TRAIN/validation trajectories, BUY/SELL
   activity, abstention, exits, harmful early closes, net return and drawdown.
4. Separate supervision effects from specialization: include a shared/monolithic
   control trained with the same oracle targets when attributing a gain to the
   specialist architecture. Changing both at once proves only a combined effect.
5. Repeat promising results across seeds before adoption; no best-seed-only
   promotion. Only then consider the full horizon ensemble and more elaborate
   opportunity-cost/game-theory arbitration if it solves a measured deficiency.

Current status: PLAN_ONLY. No target generator, new specialist, NOP threshold,
training job or model promotion has been produced by this document.

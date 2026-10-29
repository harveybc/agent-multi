# Counterfactual early-close metrics and native verification

Owner confirmation: 2026-10-29 (America/Bogota). Extends plan129.

## Frozen metric definitions

Every episode starts with one fixed order: entry, side, SL, TP, units and
initial monetary risk R. Alternative exits never resize or reopen it.
P_passive is its executable automatic exit (SL/TP/calendar), P_actual is the
tested exit, and P_best is the maximum of executable early-close values and
P_passive. All are net of the respective execution and financing costs.

- improvement_usd = P_actual - P_passive (negative is allowed).
- regret_usd = P_best - P_actual; regret_risk = regret_usd / R.
- opportunity_usd = P_best - P_passive.
- captured_fraction = improvement / opportunity only if opportunity exceeds
  the declared numerical tolerance; otherwise null, not zero or 100%.
- Beneficial/destructive/neutral early closes are classified by improvement
  with the same tolerance. Staying passive is not an early close.
- Missed opportunity: a positive opportunity exists and actual remains below
  the optimum. Report never-closed and partially captured cases separately.
- At each executable tick also report best REMAINING value, close-now regret
  against that value, and best continuation. Episode regret retains the
  original optimum so a late policy does not erase opportunities it missed.

R never changes between actions. For aggregation report mean normalized
regret, sum/mean improvement, early-close counts and destructive rates, both
by horizon/side/scenario and overall. Aggregate captured opportunity is a
ratio of sums, not an unqualified mean of small-denominator ratios. The sum
of isolated counterfactual cash changes is not an account backtest profit.

## Tests defined before implementation

Known menu [-30,-15,-5,-25], passive=-60: close at -15 must save45, miss10
and capture45/55. Keep losses negative. Include passive TP dominating all
early exits, equal exits, empty menus (unavoidable gap), invalid risk/NaN,
native-net mismatch and risk/scale invariance. Counterfactuals must share
their original order identity.

Synthetic native matrix: 3/6/12/24h references x buy/sell x gain preservation,
loss mitigation and hold-to-TP x peak/endpoint delay2/6/12h x constant,
increasing and decreasing intrabar volatility x spread0.5/3pips x entry at
session hour12/36 x seeds7/19 = 1,728 isolated positions. Positions enter
through the absolute callback and common risk rules, c=1; coefficients are
engineering controls, not optimized winners. Random perturbations are
bounded and peak points declared by construction. Optimal NET exit is
computed independently before selection; financing can shift it from the
price peak. It must not be moved to fit engine outcomes.

For every feasible close, clone the identical current position and actually
execute native CLOSE. Independently reconstruct its net cash and compare.
Replay HOLD to the automatic exit, comparing first barrier/time/quote with
the independent protection reference. Evaluate perfect oracle, close-first,
and never-close controls on this same menu; oracle regret must be zero,
while hold-to-TP controls expose destructive premature closure.

Add 16 immediate-gap cases (four horizons, both directions, two spreads),
where no early-close opportunity exists and the passive stop must take
precedence. Native unit tests cover calendar boundaries, multiple entries
per day, refusal to reverse, future-free callback, and two positions facing
the same future but with different stop distances.

## Limits and sequence

Synthetic future is test truth, not an observation or a learned model.
No quota such as 72% early closes is enforced. The finite matrix is an
explicit coverage set, not proof over every continuous market path.
Do not start financial RL, floating selection or DEAP using oracle profits
as evidence of learned performance. Finish this execution/metric component,
then define causal report-driven closure and entry arbitration for mechanism
optimization. Keep native legacy replay identical and publish cross-repo
identities, counts and the Satoshi handoff after verification.

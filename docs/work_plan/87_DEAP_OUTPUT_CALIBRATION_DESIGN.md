# DEAP output calibration: scope and exact formulas

Date: 2026-10-26. This is a design, not an executed DEAP result.
The completed grids in report84 remain controls with their original fitness.

## Fixed business envelope

Keep calendar, maximum entries/day, costs, rel_volume=.1, leverage50, initial
balance10000 and broker-distance rules fixed. These are constraints, not genes
that DEAP may relax. Do not change input features, normalization or trained
weights. No financial validation or TEST is read.

At completed bar b, let p=close[b], sigma be causal volatility and rho the
causal range scale. For TP and SL intersect the declared sigma-multiple bounds
with [min_distance/sigma, effectiveness_c*rho/sigma], retaining the engine's
float-safe margins. Denote the resulting bounds L_T,U_T,L_S,U_S.

For direction d in {long,short}:

```text
m_T(d) = L_T + a_T(d) * (U_T - L_T)
m_S(d) = L_S + a_S(d) * (U_S - L_S)
long:  TP = p + sigma*m_T(long);  SL = p - sigma*m_S(long)
short: TP = p - sigma*m_T(short); SL = p + sigma*m_S(short)
q = q(d)
capacity = floor(balance_at_entry * rel_volume * leverage / fill_price)
units = floor(q * capacity)
required_margin = units * fill_price / leverage
```

Six proposed genes: a_T(long),a_S(long),q(long),a_T(short),a_S(short),q(short),
each in [0,1]. They calibrate constant per-direction execution coefficients
against simulated categorical commands, not a newly trained risk head. The
current policy's Beta heads remain inactive. Dynamic risk-head calibration
would be a separate declared architecture, not an unannounced addition here.

q=0 must produce no opening and no fee, not a minimum-size order. For q>0,
round down to units and refuse orders below the minimum; never increase q to
satisfy a broker minimum. TP/SL endpoints map to the float-safe feasible band.
Revalidate levels at the next opening; do not recenter them after seeing it.
These endpoint semantics require tests before executing this new adapter.

## Search protocol

DEAP installed:1.4. Proposed population20, maximum100 generations,
patience10 generations, relative improvement1e-8, fixed random seed20261026.
Sequential CPU; persistent evaluated-genome cache, generation checkpoint and
RNG state. Same synthetic tapes for every candidate. Include midpoint and grid
control genomes; do not seed exclusively with one winning point. Preserve one
elite; bounded crossover/mutation never relax the business envelope.

Eligibility precedes fitness: actual openings and discretionary exits in both
directions across the diagnostic suite, legal actions/levels, activity and
correct net accounting. Ideal/inverted/NOP controls remain separate. A held-out
synthetic scenario does not participate in candidate selection.

## Fitness clarification requested

The owner previously proposed several alternatives, not one uniquely frozen
expression: Sharpe + net_profit/initial_balance, optionally adding average
orders/day, with possible coefficients. A question was sent asking which
expression governs this run. No genetic run or optimality claim has been made.

Do not optimize the fitness coefficients on the same candidate scores: that
changes the objective instead of optimizing the strategy. Once specified,
record coefficients, aggregation across markets, activity denominator, treatment
of zero-trade/zero-variance cases and eligibility before the first evaluation.
Do not substitute the earlier Sharpe-only grid for the requested DEAP run.

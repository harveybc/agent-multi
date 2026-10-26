# DEAP output calibration: scope and exact formulas

Date: 2026-10-26. Protocol fixed before genetic evaluation.
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

## Owner decision: compare fitness alternatives

The owner instructed comparison rather than picking one expression in advance.
Three objectives, each repeated with seeds7,19,43 (nine sequential searches):

```text
S = mean(ideal Sharpe on A, ideal Sharpe on B)
R = mean(ideal net_profit / initial_balance on A and B)
A = mean(ideal trades / (5 * number_of_weeks) on A and B)
F1 = S
F2 = S + R
F3 = S + R + A
```

Each synthetic120h session contains five24h business days. All of them count
in the activity denominator, including warmup/no-trade days; a candidate cannot
shrink its denominator. Coefficients are1 as in the owner's proposed examples;
publish component magnitudes rather than claim they are equally scaled. No
joint optimization of fitness coefficients. Zero-trade candidates fail activity;
undefined/non-finite fitness fails closed, never becomes a profitable score.

Identical initial populations per seed, scenarios, eligibility, genetic budget
and operators: tournament3, bounded simulated-binary crossover eta20/p=.8,
bounded polynomial mutation eta20/p=.2 with per-gene probability1/6, one elite.
Each generation scores the population before testing patience. Relative progress
is measured against the last patience-reset best with denominator
max(abs(best),1e-12); threshold1e-8, patience10, maximum100 generations after
initial population20. Cache repeated genomes with all source/recipe identities.
Save population, fitness, RNG state and progress after every generation.

Actual diagnostic categorical commands exercise opening, discretionary closing
and NOP; six genes set TP/SL/q by direction. This does not claim to calibrate
inactive Beta-head distributions. q endpoints and asymmetric directions receive
explicit behavioral tests on the real adapter before launch.

Wave C was already observed in the previous grid. Report it only as historical
comparison, not new confirmation. Fresh synthetic D (.0012,9,2) and
E (.0024,7,11), in the same amplitude/weeks/phase notation, are evaluated only
after ALL nine optimizations finish, without tuning on their scores. They remain
same-generator synthetic stress checks, not real-market validation. Evaluate
each final genome plus midpoint/grid controls with ideal/inverted/NOP outputs.

Report all seeds, stop reason, generations, unique evaluations, all3 fitness
components, trading counts/directions/win rates, costs, net profit, Sharpe,
drawdown and exit reasons. No financial RELEASE or best-feature claim follows.

## Test and automation acceptance

- q=0 makes no order/fee; q=1 uses at most rel_volume*current balance as margin.
- Independent buy/sell coefficients reach the actual stop, target and sizing.
- Domain endpoints work; illegal commands and non-finite parameters reject.
- Fitness uses net profit and the fixed day denominator; eligibility precedes score.
- Replaying a completed generation adopts the same RNG/population and result;
  corrupted cache/state is rejected by identity/hash.
- A single sequential CPU supervisor runs all nine searches, confirmations and
  final table; completed units are never silently overwritten.

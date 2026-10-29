# TRAIN ZigZag opportunity optimization

Priority: before strategy qualification and phase 2 model-specific sequential
floating selection. Owner request 2026-10-28. This replaces the earlier proposed
1/2/6/12/24 ladder with the initial 3/6/12/24-hour ladder. No neural input,
normalizer, feature list, model or live execution changes are authorized here.

## Order and deliverables

1. Freeze this design, data hashes, calendar, economic assumptions and tests.
2. Run four sequential DEAP searches, with three seeds each, population 30,
   40 complete generations; no early stopping below the owner's 15/30 minimum.
3. Publish parameters, opportunity fitness, observed pivot spacing, coverage,
   buy/sell counts, per-year results, seed dispersion and boundary hits.
4. Retain both directions' per-origin records and non-overlapping hindsight
   reference. Then discuss risk profiles and volume; do not silently optimize
   them or start floating selection from this preliminary result.

## Inputs and oracle semantics

Use admitted EURUSD H1 bars with availability in [2020-01-01,2024-01-01).
Predicate-filter parquet before calculating any pivot. No VALIDATION/TEST bars.
Verify bar and closure digests against the existing recipe. Verify positive,
ordered, finite OHLC and timestamp/availability relation. No interpolation.
Reuse the repository's symmetric hindsight ZigZag implementation; record its
source digest and its deviation-in-pips convention (not a claimed exact MT5
binary replica). Pin source files used by the external engine import.

All scales have the same hourly entry grid, after 12h from session open and
strictly before forced flat 12h before close. A known opening quote is the
reference entry; the candidate is an intentionally future-informed oracle.
No daily entry cap. An origin needs a complete observed path; split paths at
unobserved hours instead of compressing gaps. Report excluded boundaries.

For each origin and each side, target the next strictly later pivot of the
favorable type (peak for buy, valley for sell). Exit ideally at that pivot's
extreme; this is an optimistic hindsight fill, NOT a guaranteed executable
market-order fill. If no such pivot precedes the time limit, exit at the open
quote at that limit. The limit is min(4*h elapsed hours, force-flat, contiguous
observed data boundary); terminal paths without a later quote are excluded.
No pivot beyond TRAIN is used. Preserve timed exits and negative candidates.
Duration to a pivot is conservatively measured to the pivot bar's end; timed
exit duration is the opening timestamp difference. Extremum timing within an
H1 bar is unknown; report this bound, not fabricated tick precision.

Use high/low excursions including the target bar. Within-bar high-before-low
for buy (low-before-high for sell) supplies an adverse drawdown upper bound;
do not claim actual intrabar drawdown is identified. Store adverse excursion
relative to entry separately from drawdown relative to running best price.

## Frozen objective and scale

For a fixed one-unit reference, net P = directional price difference minus
the existing assumed spread/slippage/commission/financing schedule. Costs are
ASSUMED_NOT_BROKER_VERIFIED and fills are hindsight benchmarks. No reinvestment,
leverage or volume optimization in this stage. Margin limits apply later to
the executable account simulation, not to this unit-normalized opportunity map.

S = P / (max(drawdown_price_upper_bound, round_trip_market_cost_floor, 1 pip)
         * duration_hours).
F = mean over identical eligible origins of max(0, S_buy, S_sell).
The mean includes NOP zeros, never only the successful origins. Negative sides
remain in the records. Report a separate adverse-excursion denominator contrast
on each finalist without using it to select. This is inverse-hours opportunity
fitness, NOT Sharpe, profit or a sum of simultaneously executable positions.

Genes: depth 2..96 integer, deviation 1..120 pips in 0.1-pip increments,
backstep 1..min(24,depth-1) integer. Scale eligibility requires at least 100
adjacent alternating pivot pairs within contiguous sessions, with median
elapsed spacing in [h/sqrt(2),h*sqrt(2)]. These geometric bands are design
choices, not findings about market duration. Ineligible genomes rank below
eligible ones; if no eligible candidate exists, report NO_ELIGIBLE_CANDIDATE.
Best among seeds uses F; tied F uses smaller depth/backstep/deviation. Record
empirical gene sensitivity, including neutral genes. Do not claim global optimum.

DEAP tournament size 3, elitism 2, uniform crossover p=0.8, child mutation
p=0.4 and per-gene replacement p=1/3. Seeds 7/19/43. Full candidate ledger,
generation records and atomic final receipts, identified by inputs and code.
Completed units are reused only after identity and artifact hash verification;
interrupted units deterministically restart from their seed. One process and
an exclusive queue lock prevent duplicate runners. No GPU is needed.

## Non-overlapping reference

On each finalist, dynamic programming selects a maximum-net-profit schedule
from these fixed-unit candidates, allowing NOP and forbidding overlap. An
extremum exit releases the slot only at the following bar; a timed open-quote
exit releases at that quote. This exactly solves the declared additive interval
library, not balance-dependent sizing, uncertain execution or the future RL
policy. Report trades, side proportions, win rate and net pips, not account
Sharpe/drawdown for an account that was never simulated.

## Tests before implementation

Z01 hand-computed buy/sell profit, costs, MAE and drawdown; Z02 zero-DD floor;
Z03 strictly later pivots and negative/NOP handling; Z04 hour clock and forced
flat; Z05 missing hours never compressed; Z06 TRAIN predicate and source pin;
Z07 mirror symmetry with symmetric costs; Z08 DP versus exhaustive subsets;
Z09 deterministic DEAP replay and minimum budget; Z10 resume identity/artifact
tampering rejection; Z11 all four horizon table rows; Z12 neutral-gene probe.

This study does not qualify early closing or strategy execution. The next
steps remain position-dependent closure semantics, mechanism synthesis and
strategy audit, followed by phase 2 floating selection using the real RL model.

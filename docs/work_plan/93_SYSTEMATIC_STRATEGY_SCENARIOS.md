# Systematic strategy scenarios: bounded execution qualification

## Correction of scope

The earlier 22 execution cases and fixed-frequency sinusoidal controls do NOT
qualify the complete strategy or learned policy. The owner requires systematic
variation, specified expected behavior, symmetric directional comparisons and
crossing market regimes with agent states. This supplement does not change a
running learner or admit any financial feature-selection result.

## Frozen matrix

Cross every value, not a sampled subset:

| Axis | Values |
|---|---|
| Drift before transition | down, zero, up: -0.00015, 0, +0.00015 price units/bar |
| Drift after transition | same three values independently |
| Innovation amplitude before transition | 0.00002, 0.00015, 0.00060 |
| Innovation amplitude after transition | same three values independently |
| Transition bar | 32, 60, 88 in a 120-bar open session |
| Oscillation period | 8, 24, 72 observed bars |
| Phase shift | 0, 5, 11, 19 bars |
| Costs | configured and zero-cost diagnostic |

This is 5832 market/configuration cells. Drift is a generating parameter, not
a guarantee that every noisy local path moves in its direction. Every cell
has 600 preceding synthetic observed bars in five complete open sessions,
excluded from scoring, to initialize the unchanged causal volatility/range
estimators. No weekends are fabricated and no future values initialize them.
The first engineering smoke without this prefix exercised only NOP because
the estimators were undefined; it did not qualify the strategy. The corrected
runner explicitly requires finite scales before measuring the target session.
Every cell
examines decisions two bars before and after its transition. At each decision,
compare flat-state NOP/BUY/SELL, and for reachable LONG/SHORT positions compare
CLOSE now against HOLD for four bars. Unreachable open states (already closed
by broker protection) are counted explicitly, never reported as passing an
unperformed comparison. Entry eligibility is recorded separately.

## Expected behavior and independent checks

For the diagnostic controller only, use exact future simulation to value the
finite alternatives over the same four-bar horizon, with identical initial
state and costs. Choose the maximum net outcome, never the first acceptable
direction. NOP wins exact flat ties. This is a local finite-action oracle, not
global optimality and never a deployable financial signal. Simulated scores
pass through the actual categorical policy output and mask; SELL must beat BUY
when its score is greater even if both are positive. Test small, normal and
large positive score scales separately without introducing thresholds.

Independent checks reconstruct each trade's signed gross P&L from prices and
units, subtract itemized costs, reconcile balance, check side-specific TP/SL,
next-open execution, fill-time volume cap, required protections and calendar.
Compare buy and sell payoffs rather than assuming descending drift guarantees
a profitable sell after costs. A dedicated no-noise, zero-cost monotonic
fixture establishes the independent BUY-up / SELL-down directional witness.

The previous 22 broker-event witnesses remain required: both directions for
agent close, SL, TP, both touched, gap, forced flat, daily cap, invalid action,
missing window, close before SL and close before TP. They complement rather
than replace the crossed market matrix. Explicit head-order tests cover SELL
winning with BUY also positive, BUY winning, NOP, both closing heads, invalid
action masking and exact ties. Output scale does not change categorical rank.

## Test design and acceptance

Before implementation: unit tests for Cartesian count/uniqueness, prefix
invariance when future regime parameters change, deterministic OHLC validity,
all categorical winners and ties, and signed-P&L reconstruction. Integration:
real pinned simulator, actual GroupedTCN policy with simulated command-head
outputs, both zero-cost monotonic directions, and smoke cells spanning both
cost modes and opposite trend/volatility transitions.

The runner records one resumable receipt per cell, exact input parameters,
all candidate values, selected command, rejected actions and assertion failures.
No missing cell, failed assertion or absent required directional coverage can
produce a full PASS. A bounded smoke is labelled SMOKE_ONLY. Failures are
retained and the finite matrix continues so one defect does not hide others.
The final coverage table includes all cells and reachable agent states.

This qualifies only the declared finite grid and execution/arbitration path.
Learning is a separate C4 experiment. Neither a synthetic positive return nor
correct injected commands certify that PPO learns them. No claim of testing
every possible market, profitability guarantee, new DEAP winner or selection
release follows. Subsequent expansion must have its own identity and rules.

## Engineering verification before full run

37 CPU unit/integration tests passed, including the actual GroupedTCN output
and native simulator. A 12-cell smoke spanning the matrix exercised 24 flat,
23 long and 23 short comparisons with no assertion failures. Two positioned
states had already been closed by protection and were recorded as unreachable.
All five commands won at least once (29 NOP/HOLD, 10 BUY, 7 SELL, 12 CLOSE_LONG,
12 CLOSE_SHORT). The monotonic zero-cost fixtures selected BUY on rises and
SELL on falls with the correct independent profit signs. These are diagnostic
injected-head outputs, not trained model results. Full-grid execution is
required separately; smoke success is not a full-grid verdict.

# Systematic strategy scenarios: bounded execution qualification

## Governing rule

Requirements and expected behavior determine the tests; observed strategy
outputs must not redefine their expected results. A failed case is retained
and requires an implementation correction, not a relaxed threshold, removed
scenario or relabelled success. A demonstrated defect in the test itself must
be documented separately with its counterexample and a new evidence identity;
the original failure remains visible. Numerical tolerances are engineering
precision rules, never parameters tuned to improve financial acceptance.

| Requirement | Expected behavior fixed before full execution | Evidence |
|---|---|---|
| No directional first-match bias | highest legal categorical score wins even when both directions are positive | head-order tests and all cell score comparisons |
| Correct direction semantics | BUY gains / SELL loses on a noiseless rise; converse on a fall, without costs | independent monotonic fixtures through actual policy and engine |
| Legal abstention | flat NOP creates no order; exact flat tie chooses NOP | categorical tests and zero-order counterfactual |
| Correct exit semantics | current-side CLOSE executes at next open unless broker protection takes precedence | 22 broker-event witnesses plus reachable long/short comparisons |
| Accounting | signed units times price change minus itemized costs equals net; balance reconciles | independent arithmetic on every emitted trade |
| Business constraints | both sides obey calendar, daily cap, required SL/TP and maximum margin allocation | native-engine witnesses and per-cell assertions |
| Causal risk scales | no synthetic evaluation before scale warmup; future regime cannot alter earlier scales | prefix perturbation integration test |
| Systematic coverage | all 5832 cells present; missing/failing cells cannot yield full PASS | Cartesian uniqueness test, receipt count and final coverage |

This is a domain acceptance hierarchy, not a requirement to imitate software
organizational layers. The market generator and scenario expectations are
versioned independently of policy weights. The monotonic and accounting
oracles are independent checks. Four-bar alternative payoffs use the native
simulator itself: their ranking tests arbitration consistency, NOT independent
correctness of every simulator outcome or global strategy optimality.

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

## Completed full grid

The pinned `e3468172` execution closed at 2026-10-27 00:39:32 UTC after
528.86 seconds on CPU: 5832/5832 cells, zero failed assertions, verdict
`PASS_DECLARED_GRID`. The 5837 closure artifacts were independently read back
and matched their hashes. Closure SHA256:
`807acf36d7b3e8fb1fef83ae69eadbeaf6ae30385c22b2f750c8c24c9c3959a0`.

There were 11664 flat-state, 11224 long-state and 11224 short-state comparisons.
880 open-position comparisons were unreachable because the entry/protection
path had already removed the position; they are explicitly reported, not
counted as successful comparisons. All five commands won:
12667 NOP/HOLD, 4859 BUY, 5184 SELL, 6031 CLOSE_LONG and 5371 CLOSE_SHORT.
The new traceability assertions were not loosened after inspecting results.

Public aggregate: `docs/audits/evidence/systematic_strategy_matrix_20261026/SUMMARY.json`.
This is execution/arbitration evidence for the declared baseline grid. It is
not evidence that a trained policy learns the injected decisions, nor a DEAP
optimum. C4 separately failed learning qualification on all three seeds.

# Pragmatic controller contrast, before floating selection

2026-10-30. Owner accepted a simpler hierarchical reference alongside predictive
logic and bounded sequential allocation. This supersedes treating a strategic
game solver as a prerequisite to every runtime decision. Keep152-159 as exact
references and diagnostics; do not discard their opportunity-cost findings.

## Shared contract

One position; leverage50; margin at most q*0.1*current balance, once; native
costs and funding; no daily entry cap; entry12h after reopening, flat12h before
weekly close. Every opening has SL/TP/q from the unchanged shared risk recipe.
Horizons3/6/12/24; categorical BUY/SELL/NOP when flat, HOLD/CLOSE when occupied.
Position state belongs to the controller, not to model observations. No input
clipping, rolling z-score, new feature selection or learned models in this run.

All arms receive the same market-report paths and probabilities and use the
same native engine. Reports are ideal or deliberately perturbed test signals;
a separate prefix-only three-path baseline is a causal diagnostic, NOT a
learned model or calibrated uncertainty. Never label synthetic confirmation
as financial VALIDATION. No future oracle enters a causal report.

## Three executable references

SIMPLE: rank all eight opening alternatives by
`(expected_net/risk - risk_penalty - k*dispersion/risk)/(tau0+expected_duration)^alpha`.
NOP unless strictly above entry_threshold; exact cross-action ties abstain.
Expected duration includes first-passage protection, not just nominal horizon.
For an occupied position compare CLOSE now against common future exit choices
and passive settlement, using the position's existing SL/TP and costs.

PREDICTIVE_PROXY: same opening rule and risk geometry; closes compare next1/2/3
hours, the position's nominal horizon and passive settlement. This explicitly
models short/long confirmation. It is an adaptation of the owner's idea, NOT
a reproduction or attribution of the historical predictor strategy's code.

SEQUENTIAL: shortlist the two highest-ranked positive opening proposals (ties
abstain). Evaluate common contingent plans across all report samples: current
action; optionally close after3/6/12/24 decisions; optionally one subsequent
order, then native settlement. Include pure NOP and passive HOLD, single-order
and CLOSE-without-reentry controls. No same-tick reversal. Plan conditions may
inspect whether protection already closed the position, never hidden prices.
The SAME plan is scored across samples BEFORE selection; never choose a
different future-optimal plan per sample. Replan each tick, execute only the
first action. This is bounded two-order receding-horizon control, not the
global TRAIN frontier. Exclude dominated robust-return/worst-DD vectors, then
apply the explicit configured risk preference. Retain every evaluated vector.

The planned close offsets are candidate future actions, not enforced holding
limits or horizon-specific liquidation in the actual account. Actual close
is reconsidered at every tick and may be overridden by native protections.

## Frozen first grid and sequencing

Eight profiles: alpha{0,1} x uncertainty_penalty{0,1} x persistence{1,2}.
tau0=1h, risk_penalty=0, entry_threshold=0, close_margin=0, urgent advantage=.25R;
sequential return/DD preference=1, shortlist2, four close offsets. These are
declared engineering starting points, not claimed optimal coefficients.
Keep q quality=1 with existing risk-budget sizing; no simultaneous risk tuning.

Run tests -> full paired calibration grid -> freeze at most two representatives
per arm -> confirmation and noise sweep -> report and audit evidence.
Calibration uses fresh synthetic draw seed family910000, all36 combinations
of three market patterns, three volatility envelopes, two spreads and mirrors.
Confirmation uses disjoint920000 seed family with the same crossed design.
Smoke uses930000 and940000 and cannot admit a scientific candidate.

Calibrate on noise scale1 only. Select representatives by return and DD from
the eligible (>2 mean orders/week) non-dominated set. If no eligible candidate
exists, retain a diagnostic fallback explicitly ineligible, never stop reporting.
Freeze before reading confirmation. Confirm on ideal and noise scales.5/1/2,
plus the separate causal-prefix baseline. Preserve all winners AND failures.
No confirmation tuning, automatic release, new financial training or floating
selection. DEAP refinements follow this contrast, not replace correctness tests.

## Test design before implementation

PC01: configuration validation, units, all directions ranked jointly, tie/NOP,
duration scaling, strict thresholds and no NaN acceptance.
PC02: native quote/cost/calendar/risk invariants; reference runner parity with
the existing frozen native account runner when supplied its unchanged arbiter.
PC03: same-position ideal beneficial and harmful closes, both directions;
persistence cannot suppress the declared urgent case; broker preemption wins.
PC04: reports and all selected actions invariant to unobserved-future mutation
in causal mode; fixed report injection isolates the controller from true future.
PC05: native scenario clones share no mutable account state; their entire
unobserved suffix is replaced before simulation; cash/MTM/DD use native quotes.
PC06: expect-then-select, not sample-wise oracle; short sequence versus long
hold, NOP waiting and local-close/account sign counterexamples remain tests.
PC07: saved job/source identities, exclusive queue lock, bounded workers,
atomic per-unit seals, hostile resume, automatic negative-result report.
PC08: native ledgers and full metrics: return, USD net, Sharpe, worst DD,
orders/week, buy/sell counts and win rates, costs, horizon mix, early closes,
fixed-position harmful closes/regret and runtime. Undefined metrics stay null.
Orders/week is per complete market session (Sunday reopening through Friday
close), NOT the number of ISO week labels touched by that session. Preserve
the native ISO metric separately. The first smoke exposed this denominator
mismatch; corrected and regression-tested before any calibration grid launch.

The PathReport used in this simulator is a test-plane source for common signal
statistics and scenario planning. It does not prescribe a full OHLC forecast
head for the future RL model. The simple controller consumes normalized
opening evidence and position-evaluated exit values; the sequential variant
additionally requires a causal scenario adapter. Its future learned adapter
must be qualified separately, without importing oracle paths or agent-state
feedback into model observations.

## Delivery

JSON configuration, pure controller/risk interface, bounded native planner,
test-qualified smoke, resumable sequential phase runner, frozen selections,
paired reports with source hashes, and timing measured before publishing ETA.
Profile/feature selection stays unchanged. Each further optimization is a new
identity on fresh confirmation, not a rewrite of these results.

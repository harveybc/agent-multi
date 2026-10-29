# Dynamic account arbiter and empirical Pareto search

2026-10-29. Owner approved full-account coordination after the133 close/hold
component. This supersedes the hold on synthetic account optimization only;
financial learning, feature selection and TEST remain outside this task.

## Requirements and frozen sequence

AP01: one native account and one position; flat actions are the eight
direction/profile candidates(3/6/12/24h x buy/sell) plus NOP. Occupied actions
are HOLD/CLOSE, with the qualified position-conditioned rule. No same-tick
close/reopen, daily cap, agent-state observation or input rescaling.
AP02: every candidate replays the complete account, updating balance, integer
units, costs, funding and unrealized equity. A blocked opportunity is valued
by a different complete trajectory, not counted as an additional cash penalty.
AP03: payoff tests must include short-blocks-long and long-blocks-short, an
initially inferior action enabling a better sequence, and mixed horizons/sides.
Use tiny exactly enumerable scheduling fixtures for mathematical expectations;
also test the actual native calendar, account cash and controller integration.
AP04: strategy score is configurable, not a hardcoded hierarchy of horizons:
`w[h]*expected_net/risk - lambda*uncertainty/risk
 + trend_weight*direction*past_efficiency*h/24
 - time_weight*expected_duration/24 - threshold`.
Only strictly positive unique maxima enter; ties abstain. Time weight is a
policy parameter, not an extra debit to account payoff. Past efficiency is
signed6bar change divided by sum of absolute changes, computed from completed
bars; synthetic regime labels are not observations.
AP05: objectives maximize mean account net return and minimize worst scenario
mark-to-market drawdown. Report Sharpe separately, plus orders, win rates,
direction/profile use, exposure and costs. Trading eligibility requires mean
at least3 entries per simulated market week; NOP remains an explicit ineligible
control, not an artificially forced trading policy.
AP06: DEAP NSGA-II, population20, generations300, seeds7/19/43 sequentially.
Archive all evaluated nondominated eligible vectors; never claim a certified
global frontier. Preserve duplicate policies/fitness equivalences honestly.
AP07: freeze calibration, confirmation, objective, bounds and controls before
search. Confirmation does not select or retune. Confirm five representatives
by evenly spaced return ranks on the frozen frontier (all if fewer than5).
AP08: source-pinned durable queue with candidate receipts, per-seed finals,
progress and exclusive lock. Restart deterministically from seed using only
authenticated cached fitness; a mismatch refuses, never substitutes zero.

## Architecture and test design before code

Reusable pure arbiter/fitness/archive helpers in agent-multi. Native account
driver uses heuristic-strategy AbsoluteBarEnv, shared risk proposals and
CloseController. Fixed risk geometry is the median-profile configuration from
128; optimize allocation/size/close policy first, not all SL/TP coefficients at
once. The normalized volume multiplier remains in[.25,1] and applies to q,
not twice to rel_volume. Larger horizons receive no automatic size preference.

The eleven genes are four horizon weights[.25,2], trend weight[-.5,.5],
uncertainty penalty[0,2], time weight[0,1], entry threshold[0,1],
size multiplier[.25,1], close margin/R[0,.2], close persistence integer[1,3].
Urgent close threshold=.25R remains fixed. Search resolution=.001 except
persistence. Gene-effect witnesses must demonstrate every parameter can alter
its intended calculation; search receipts report unique decoded values.

To avoid repeated path valuation, cache normalized per-unit report values by
world, entry bar, profile, side and current decision. Only the synthetic cache
adapter may consume these; causal live-report validation is not bypassed.
Verify cached decisions against the full report route, and final accounts
against the native accounting result. No approximation of fills or cash.

Calibration has12 synthetic120bar market weeks: trend/reversal/overlap x
increasing/decreasing volatility x mirrored direction. Seed7 drives exogenous
prices and fixed report errors. Confirmation repeats the factorial with seed43
and different phases/turn times. Fixed randomness is keyed by world/time,
never shifted by changed actions. Include adverse price gaps and explicit
native costs. Reports are ideal-path-based with declared biased/dispersion
errors: this is strategy design with simulated model outputs, NOT causal or
learned financial evidence. Nominal horizons label the admitted risk profiles;
they are not claimed new separately trained timeframes.

Controls: NOP, equal-weight/all profiles, greedy profit/risk, shortest profile,
longest profile and each single profile. Tiny payoff matrices and full-account
leave-one-specialist-out contrasts accompany the frontier. A finite on/off
Nash diagnostic uses a separately declared scalar preference return minus
drawdown; it is not the Pareto objective and does not certify neural policies.

Unit expectations: bid order invariance and tie NOP, own-side selection, true
drawdown from equity rather than trade sums, nondominance including duplicate
vectors, opportunity scheduling exactly enumerated, gene decoding/bounds and
budget guards. Integration: native step sequence, no overlap, cash conservation,
both-side trades, multiple profiles when appropriate, cached/full parity,
future-independent arbiter given identical reports and mutation of hidden
regime labels. Measure runtime before the full queue; record all failures.

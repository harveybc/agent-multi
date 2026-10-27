# Multiscale strategy acceptance design

Owner correction, 2026-10-26. This is an execution task, not a deferred review.
No financial fitting, input selection or active-worker configuration is changed.

## Requirements fixed before execution

The market is one synthetic FX instrument. Cross EVERY combination in the
declared JSON axes: seven signed drifts before/after (zero plus three strengths
in either direction), three volatility amplitudes before/after, abrupt/four-bar/
twelve-bar transitions, oscillation periods 8/24/72, decision horizons
1/4/12/24/48/72 observed bars, and configured/zero costs. This is 47,628 cells.
The transition starts at bar 32, phase is fixed at zero. These are stated
constants, not additional swept dimensions. Earlier phase/location tests remain
separate evidence. Each path contains 600 warmup bars and 240 target bars.

Three clocks are independent: oscillation period, transition duration, and
holding/comparison horizon. Calendar flat deadlines override long horizons.
Amplitude and measured standard deviation of increments are recorded separately.

## Oracles and tests

M01: exact Cartesian size/uniqueness and each axis's marginal counts.
M02: deterministic positive OHLC; changing post-transition parameters cannot
change the prefix; transition interpolation reaches the stated endpoints.
M03: horizon 1 and 72 both execute, with next-open fills and calendar priority.
M04: prescribed logits with BOTH open scores positive choose the declared
larger score, regardless of iteration order; NOP and both closes are explicit.
M05: every reachable positioned state tests HOLD and its own CLOSE; wrong-side
close, fractional/NaN/out-of-range actions must reject without mutation.
M06: size, TP and SL are checked independently from configuration, current
balance, decision-time volatility and fill price, for each direction.
M07: signed gross and net reconciled from recorded fills/costs; time/cap/protection
assertions retained. Protection-closed entry states are counted, never passed as
positioned tests. Dedicated broker witnesses must cover both close directions.
M08: independent prescribed head tests are separate from simulator-derived
counterfactual ranking; the latter cannot certify optimal execution economics.
M09: full identity, resumable cell receipts, failures and coverage are persisted;
no incomplete run is labelled PASS. Existing sealed evidence is never overwritten.

Tests are written before the implementation extension. Expected command tests
are literal fixtures, not labels obtained from the strategy under test. Changes
must preserve the original 5,832-cell design when no new design is supplied.

## Optimization boundary

This tests the configured directional coefficients, not a claim of universal
optimality. It does not rehabilitate the rejected historical DEAP selection.
An optimizer cannot change business constraints or the expected test outcomes.
Parameter search and learner qualification remain separately reported evidence.

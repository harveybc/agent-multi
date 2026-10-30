# Suppressed closes: fixed order and account continuation

2026-10-30. Next task after157, before model-specific floating selection.
This is a post-hoc diagnostic of frozen synthetic confirmation, not a search
on that confirmation and not a learned policy or global Pareto certificate.

## Contract and tests fixed before measurement

SC01. Authenticate the completed source and native code. Reproduce each
baseline account and its gate counts without changing a single action.

SC02. Inspect every visited occupied tick with a raw close advantage above
the frozen margin but final action HOLD. Freeze entry, direction, units,
barriers and native costs. Enumerate native CLOSE at each executable decision
until passive SL/TP/forced-flat, and include passive HOLD as an alternative.
Protection at next open preempts a requested close exactly as in production.
Do not reopen while measuring this fixed-order optimum. Test both directions,
beneficial/destructive closes, preemption, flat-state rejection and isolation.

SC03. Record net CLOSE-now minus passive net, optimum minus CLOSE-now and
optimum minus passive, normalized by original modeled monetary SL risk.
The optimum is executable next-open net cash, never a candle extreme fill.
Keep per-tick rows and unique position identities; ticks are not independent
trades. Do not sum their mutually incompatible improvements into account P&L.

SC04. For each suppressed tick separately, replay the whole account with
exactly one forced CLOSE request at that same position/tick; afterward the
unchanged causal policy decides normally. No copied future action sequence.
Require identical prefix, actual targeted position, and native cash parity.
Account delta = that position's realized delta + subsequent-account delta.
The latter includes later opportunities, timing and balance-dependent sizing;
it is not a pure price of time or an additional cash payment to an agent.

SC05. Store full paired native traces and per-world resumable units outside
the checkout. Authenticate source/code/job identities on resume and reject
altered units. CPU worker cap4. Test cold/warm policy parity, single intervention,
exact decomposition and rejection of missing intervention targets.

## Interpretation and next step

Report both risk/return axes. A positive fixed-order effect need not improve
account return or drawdown. Conversely, releasing the slot can compensate
a locally inferior exit. Neither measured sign may be fed to the policy on
this confirmation set. Use the diagnosis to specify a sequential controller,
then compare it on fresh scenarios with the exact finite references152-155.
Current optimization and its immutable source remain untouched.

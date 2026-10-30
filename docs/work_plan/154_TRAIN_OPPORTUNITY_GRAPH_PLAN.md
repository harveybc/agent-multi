# TRAIN K=3 opportunity graph and bounded native verification

2026-10-30. Follows152/153; before floating selection. Existing DEAP runtimes
remain unchanged. No validation or TEST data are read.

## Requirements and acceptance fixed before execution

TG01: authenticate the original K=3 champion receipts, arrays, four-year TRAIN
inputs, source code and indicator. A moved working checkout is not authority:
recover an exact matching git blob in a separate snapshot, never edit the
original or silently accept a changed digest. Record each recovered revision.

TG02: every complete origin retains all three chronological pivot endpoints
for BOTH directions at each horizon. Preserve negative and locally inferior
alternatives. Incomplete K=3 origins remain explicitly censored, never extended
across a gap or session. Compare candidate indices and winning endpoints with
the original receipts before trusting the expanded graph.

TG03: graph records are hindsight reference edges, not executable orders. A
pivot high/low is not a guaranteed market fill. Native execution of a planned
pivot exit is CLOSE at the next open after that pivot bar, unless protections
or weekly flat preempt it. Keep reference and actual exit separate.

TG04: source bars are BID OHLC, whereas the native AbsoluteBarEnv expects MID
OHLC. Use a narrow explicit quote adapter: BID remains unchanged; ASK is BID
plus the declared simulated spread. Do not interpolate bars, relabel BID as
MID or invent observed historical ask quotes. Costs remain assumed, not broker
verified. Tests assert quote geometry and cash arithmetic in both directions.

TG05: select the first complete eligible weekly session in each TRAIN year
by calendar and availability ONLY, before reading any trade score. Branch over
the first three hourly decisions from operational opening. Each opening chooses
direction, horizon and one of its available K=3 endpoints. While occupied,
HOLD/CLOSE are alternatives until the planned pivot exit becomes mandatory.
After the branch window, no new entries: continue native HOLD/planned exit and
protections until the weekly deadline. This bounds computational work without
inventing an early liquidation. It is an EXACT RESTRICTED PREFIX problem, not
an exact week/year optimum. No horizon-duration cap is introduced.

TG06: compute complete account return / hourly-close-equity drawdown for all
terminal paths, independently check nondominance, and replay all frontier
witnesses from fresh native states. Report missing action support, preemptions,
native close reasons, BUY/SELL, source identities and exact work counts.

TG07: repeat the complete graph and native proof into fresh roots; deterministic
artifacts must match. Budget exhaustion is an explicit failure, not a partial
frontier presented as complete. Source artifacts are read-only.

## Test layers

- Unit: all three events survive, both directions, negative alternatives,
  chunk boundaries, duplicate/invalid event order, source-tampering refusal.
- Integration: exact original best-endpoint reconstruction and BID quote
  preservation; next-open pivot exit vs unrealistic extreme fill; mandatory
  protections, planned exit, no simultaneous reversal.
- System: all four source horizons/years; frozen calendar-only sample;
  conditional frontiers, native cold replay, source hashes and repeat hashes.

## Deliverables and interpretation

Persist full graph arrays outside git (real source-derived data), publish
aggregate counts, source/content digests and scoped native results. Restore
dependencies outside the live checkout. The reference graph is an oracle used
to design and test the mechanism. Deployment still requires non-anticipative
allocation from causal reports; the current local-score DEAP is a comparator.

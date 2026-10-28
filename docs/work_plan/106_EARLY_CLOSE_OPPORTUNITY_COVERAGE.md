# Early-close opportunity coverage, not a target exit percentage

Date: 2026-10-27. Owner priority: each actionable ideal early-close opportunity
must produce an executed close, both directions, to reduce loss or protect gain.
An entry-only oracle plus passing accounting cannot certify this requirement.

## Requirement and limits

For a positioned decision, compare liquidation at the next executable open
against continuation through protection/deadline on the same known synthetic
path, including costs. If liquidation net exceeds continuation net, require the
correct categorical CLOSE and an AGENT_CLOSE receipt at that next open. If HOLD
is at least as good, do not force a close. Verify the net amount and side.

This is the current finite-horizon reference, not a globally optimal stopping
policy. It must not be relabeled as best possible exit timing. Existing broker
protection at the next open has priority. An intrabar barrier before another
decision is different from an available decision that failed to close. The H1
simulator makes hourly-bar decisions, not tick-by-tick decisions inside an OHLC
bar. A true intrabar policy requires an explicitly finer time-resolution study.

The owner's historical approximately 72% early-exit share is a comparison, not
a fabricated quota or a proven result for this simulator. Report two separate
fractions: early exits / all trades, and correctly executed closes / identified
actionable close opportunities. The latter must be 100% in the defined ideal
witnesses. Zero opportunities means NO_EARLY_CLOSE_WITNESS, not a passing close
test. No source or result is modified to manufacture a desired exit ratio.

## Coverage defect measured

A spaced sample of 1,088 cases from the in-progress original quote grid had
534 trades and just four early-close opportunities, all executed. 778 cases
had no trades. This is a development diagnostic, not a representative random
estimate over the complete Cartesian product. The first 8,192 sequential cases
had 5,266 trades, all TP, and zero close opportunities. Oracle entry selection
avoids many losing trades, so this trajectory generator alone does not exercise
the position-management behavior the owner requested.

## Implemented additional verification

1. Literal counterfactual tests expanded to 72 combinations: direction 2 x
   profitable/neutral/losing position 3 x horizon 4/24/72 bars x normal/tripled
   spread 2 x adverse-SL/favorable-TP continuation 2. Every combination runs
   both CLOSE and HOLD and checks net ordering before executing ideal scores.
   72 integration cases plus 10 audit unit tests passed, zero skipped, 155.88 s.
2. `app/early_close_audit.py` checks every retained positioned decision and
   reconciles required commands with executed trades. Counts by side include
   preserve-gain, reduce-loss and future-SL avoidance. HOLD instead of required
   CLOSE, wrong direction, delayed close, incorrect net and unsupported early
   close receipts fail. Empty cells and cells without opportunities are explicit.
3. `tools/audit_early_close_grid.py` follows immutable producer batches, verifies
   hashes, writes per-cell audits and aggregate capture/share statistics, and
   rechecks original data at terminal closure. No callback into the learner and
   no financial admission. A separate output identity prevents mixing revisions.
4. The position-management grid covers the SAME complete declared market axes
   as the original grid, with separate long/short positioned probes: 2,624,400
   cases. The first permitted opening is forced synthetically; subsequent
   management uses the ideal reference. The chosen entry is a test precondition,
   not a trading recommendation, learned decision or profitability result.
   A probe unable to enter is explicitly counted as uncovered, not silently
   taken as a successful position-management witness.

## Predeployment smoke

128 deterministically sampled position-probe cases: zero execution failures,
zero probes without entry, 186 trades, 86 early exits and 100 TP exits.
37 early exits were long, 49 short. 66 reduced losses, 20 preserved gains;
38 replaced a reference continuation ending at SL. All 86 required closes were
executed and reconciled; early-exit share 46.24%, opportunity capture 100%.
43 cells had no early-close opportunity and remain explicitly unqualified for
that particular behavior. No SL exits occurred in this sample; this does not
guarantee no unavoidable SL in the complete matrix.

The predeployment smoke took 10.92 s wall / 24.14 worker-seconds. A crude linear
estimate is about 23 hours for the complete additional grid at six workers;
this estimate is from a small mixed sample, not a measured full-run throughput.

Runner/bridge checks: 16 focused tests passed, including resume corruption and
actual first long/short probe execution. The existing quote grid remains pinned
and unchanged. The new six-worker grid is queued AFTER its successful closure;
it must not double the existing CPU/RAM load. Its auditor and the original-grid
auditor follow their own producer separately and retain all failures.

## What this does not close

This strengthens the ideal-output strategy evidence. It does not repair or
qualify the historical learned policy with no INNER_A early exits, prove global
optimal stopping, select new DEAP parameters or authorize floating financial
selection. Those remain separate requirements under plan 102. The final audit
must include per-axis positioned opportunity counts and noncoverage, not just
the total number of generated price paths.

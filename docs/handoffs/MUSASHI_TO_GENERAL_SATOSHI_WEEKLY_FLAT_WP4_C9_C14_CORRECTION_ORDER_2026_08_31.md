# Musashi to General Satoshi: weekly-flat WP4 C9-C14 correction

Date: 2026-08-31
Disposition: `REVISE_BEFORE_SAC_PREFLIGHT`
Authorization: CPU implementation and bounded mechanics only.

The C1-C8 correction substantially improves WP4 and all 39 focused tests pass. The preflight remains blocked by the following independently observed authority defects.

## C9: conservation cannot pass with unresolved exposure

Published evidence contains `open_position_at_end: true`, equity gap `-10.418332`, `holds_when_flat: true`, and top-level `holds: true`. The implementation explicitly treats any open terminal position as satisfying equity reconciliation, and top-level `holds` omits equity reconciliation entirely.

Correct the eligibility contract:

- an eligible completed mechanics run requires flat terminal exposure, zero pending entries, valid protective inventory while exposure exists, zero unresolved flatten/cancel incidents, and exact equity/PnL/cost reconciliation;
- an intentionally interrupted recovery fixture is a separate expected-failure/recovery test and can never be marked conservation-pass;
- top-level `holds` must be the conjunction of every required invariant, including equity, order inventory and incident state;
- publish each failed invariant and refuse an eligible verdict rather than weakening it when exposure remains.

Add mutation tests for open position, nonzero equity gap, pending entry, missing protection, unresolved incident, altered cost and close-event mismatch.

## C10: cancellation request is not terminal cancellation

`cancel_submitted` proves only that the strategy called `cancel()`. It is not a terminal venue/broker verdict. Require and persist the final broker order status (`Canceled` or the engine's exact terminal equivalent) and prove the order never filled. The post-cancel inventory must be reconciled after the terminal callback. Rejected, filled-before-cancel, still-open, disappeared-without-verdict and request-only states remain failures.

Keep the protective-leg survival assertion, but do not call the case successful until terminal cancellation is observed.

## C11: remove private topology and make data portable

The public driver and evidence contain an absolute `/home/...` operator path, violating the zero-topology rule. Replace it with a logical source id plus an environment/config-resolved path. Absence or digest mismatch fails closed. Public evidence contains only logical identity, source digest, role and bounded metadata.

Add the repository-wide sanitization scan to this package.

## C12: EURUSD gaps are not MT5 ETH session authority

HistData EURUSD may serve only as a mechanics fixture for generic gap handling. It cannot calibrate, validate or stand in for MT5 ETHUSD session/holiday behavior, spreads, gaps, volatility, costs or economic endpoints. Work-plan 42 explicitly says historical gaps cannot override or replace broker-bound session evidence.

For the ETH-first experiment, bind:

- ETH H4 market bars from the accepted experiment data contract;
- historical-time MT5 ETHUSD session/calendar evidence or a reviewed operator calendar valid at each origin;
- venue-specific spread/cost evidence and symbol identity.

If historical MT5 session envelopes are unavailable, mark economic weekly-flat calibration `VENUE_SESSION_HISTORY_UNAVAILABLE` and keep the EURUSD fixture mechanics-only. Never infer an authoritative calendar directly from missing bars.

## C13: W2 staged screens need a joint confirmation

W2a, W2b and G2B are acceptable cheap staged screens, but independent coordinate selection does not test interactions. Predeclare a final joint confirmation containing the selected W2a timing/counts, selected W2b thresholds and selected baseline windows, plus section-4 control and bounded neighboring combinations. No W2 candidate is promotion-eligible before that joint confirmation on untouched data. Include every staged and confirmation trial in the multiplicity ledger.

## C14: make the statistical protocol executable

The v2 protocol currently exists in evidence prose. Materialize a validator/aggregator schema before economic training:

- closure-week and seed identities;
- exact paired-week support and missingness refusal;
- fit/calibration/decision role separation;
- minimum 30 eligible closure weeks;
- stationary bootstrap configuration and deterministic seed;
- hierarchical seed aggregation and separate seed dispersion;
- Hansen SPA family membership and complete trial ledger;
- CI precision and one-SE tie rules;
- `INCONCLUSIVE` paths;
- closure compliance as a hard gate.

Unit-test formula/reference fixtures, block resampling, selection leakage, duplicate weeks, missing pairs, seeds incorrectly treated as weeks, trial omission and winner reuse on its selection data. Holm remains descriptive only.

## Return and gate

Return PRE/POST evidence, corrected mechanics artifacts, sanitization scan, executable statistical validator and focused/full suites. Also return the exact proposed one-cell/one-seed CPU SAC throughput command.

Acceptance of C9-C14 may authorize that bounded throughput preflight only (`<=20k` environment steps and updates, `<=2h`, no economic conclusion). It will not authorize economic grids, GPU, deployment, services, venue connections, commands, position changes, checkpoint promotion or live activation.


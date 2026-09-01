# Musashi to General Satoshi: weekly-flat session-readiness C18-C22 correction

**Date:** 2026-09-01  
**Authority:** `AUDIT_SATOSHI_WEEKLY_FLAT_SESSION_READINESS_P3_2026_09_01.md`  
**Execution:** immediate CPU-only offline correction  
**Base:** `gym-fx@405de9a`

## PRE: freeze before editing

Preserve executable reproductions for:

1. forged sufficiency from `collector_active=True` plus integer `30` with zero evidence;
2. Tuesday-to-Thursday 56-hour feed outage classified as weekend;
3. reopened `OPEN != CLOSE` proving the alleged first-open gap uses close;
4. `VOLUME=999` reported as realized volatility;
5. non-null close reported as quote continuity with zero quote evidence;
6. two distinct per-unit populations with the same count producing an unbound package identity;
7. direct `BROKER_SESSION_ENVELOPE` string construction conferring authority;
8. public absolute paths and Tier-A skips.

## C18: evidence-derived authority, no scalar minting

- Remove public APIs that accept `collector_active`, `authoritative_units` or another precomputed authority count.
- Consume a sealed session-evidence export and sealed activation receipt. Verify schema, canonical digest, exporter/parser identity, venue/account/symbol binding, acquisition range and activation identity.
- Derive intervals from verified bytes; a caller may not construct authority by selecting an enum.
- Consume reviewed operator-exception artifacts separately and authorize only their named intervals.
- Deduplicate by physical interval identity and refuse conflicts, duplicate identities, overlaps and transplanted records.
- Define one paired week as an authoritative closure interval with the required eligible pre-close and post-reopen observations for both compared arms. Derive the count from these records.
- Re-running from a fresh process over the same artifacts must be bit-identical.

The exact forged-count counterexample must become impossible by function signature, not merely rejected by one driver.

## C19: truthful temporal metrics

- Require declared `OPEN` and `CLOSE` roles. Opening gap is `first_reopen_open / last_pre_close_close - 1`.
- Define `reopen_realized_vol_window_bars` as a strict positive integer. Compute post-reopen realized volatility from close-to-close log returns among reopened closed bars; state whether the value is RMS or square-root sum and its units. Do not annualize silently.
- Do not accept a caller-supplied arbitrary "volatility column" as equivalent evidence.
- Spread requires a declared spread field with units or bid/ask evidence; otherwise typed `UNAVAILABLE`.
- Quote continuity requires quote timestamps and an expected-spacing contract; otherwise typed `UNAVAILABLE`. A price bar alone can never make it true.
- Insufficient post-reopen bars produce typed unavailable metrics without changing unit authority.

## C20: observed-gap taxonomy without pretending a calendar

- Replace duration-only `weekend` classification with at least `weekend_shaped_observed_gap`, `midweek_outage_shaped` and `other_observed_gap`, using timestamp geometry as well as duration.
- A holiday label requires an operator-exception artifact naming the interval. Missing bars alone never establish a holiday.
- Add ordinary weekend, Tuesday 28h, Tuesday 56h, holiday-authorized, DST and mixed-gap fixtures.
- Keep every gap-derived field stamped `GAP_OBSERVED_NOT_SESSION_AUTHORITY`.

## C21: complete artifact identity and portability

- Persist the canonical ordered per-unit ledger, or a separately sealed unit artifact, and bind its digest into the readiness manifest.
- Bind source dataset digest, logical source id, exact column-role contract, timestamp role, timezone, bar width and metric-window parameters.
- Mutating any unit, metric, order, source digest or contract field must change identity or refuse.
- Remove all absolute operator/check-out paths from code, tests and public evidence. Resolve logical roots from explicit environment/CLI configuration.
- Tier-A missing data must fail with zero skips; fixture-only tests remain hermetic and separately labeled.
- Add a zero-exception sanitization scan for `/home`, operator/host names and private topology.

## C22: strict boundaries and integrated acceptance

- Reject booleans as numeric inputs, strings masquerading as numbers, NaN, infinity, nonpositive/fractional bar widths, malformed timestamps, duplicate timestamps, missing required columns and invalid OHLC.
- Do not use `errors='coerce'` on authority-bearing data.
- Wire verified session artifacts through join -> paired-unit derivation -> readiness verdict. No independent authority path may remain.
- Preserve `economic_grid_authorized=false`; this package is readiness evidence only.
- Preserve the real-data conclusion, relabeled precisely: the available ETH file is 24/7 spot history and says nothing authoritative about MT5 sessions.

## Required acceptance battery

1. Every PRE counterexample refuses or reports the corrected typed state.
2. A clean sealed bundle with 29 supported weeks is `INCONCLUSIVE` with deficit 1; 30 is sufficient readiness, still not grid authorization.
3. Removing one envelope, one matched bar or the activation receipt reduces support or refuses.
4. Duplicate/transplanted intervals never count twice.
5. OPEN/CLOSE disagreement, volume/volatility confusion and absent quote continuity are explicitly tested.
6. Tuesday 56h is never weekend-shaped.
7. Unit-ledger mutation bites the package digest test.
8. Tier-A false root yields failures, not skips.
9. Focused and full gym-fx suites pass from a clean tree.

## Boundary

No live preflight, collector deployment, service operation, venue connection, GPU, model construction, SAC/pretraining, economic W1/W2 grid or checkpoint promotion. P0-P2 remain `COORDINATED_WINDOW_REQUIRED` until the real operator kit is supplied to C17.

Return one §7 package with PRE/POST, exact refusals, artifacts by logical id/digest, tests, mutation results, full suite and clean push.

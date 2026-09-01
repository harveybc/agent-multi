# Audit: Satoshi weekly-flat session-readiness P3

**Date:** 2026-09-01  
**Auditor:** General Musashi  
**Package:** `agent-multi@3d28ba78`, `gym-fx@405de9a`  
**Verdict:** `REVISE`

The negative data finding is accepted as diagnostic: the available ETH H4 file is 24/7 spot history, contains no broker-shaped weekly closure, and cannot authorize an MT5 weekly-flat economic grid. P0-P2 were correctly left unexecuted because a flat observation is not the operator kit required by the C17 judge.

The implementation is not accepted yet because its public APIs can mint authority from caller-supplied scalars and several reported metrics do not match the data they consume.

## CRITICAL-1: authoritative sufficiency can be fabricated with an integer

**Code:** `tools/wp4_session_readiness.py:280`, `:301`.

`count_paired_weekly_units(authoritative_units=...)` trusts an integer. `data_readiness_verdict` separately trusts `collector_active: bool`. The exact counterexample on pristine `405de9a` is:

```text
data_readiness_verdict(
    collector_active=True,
    authoritative_units=30,
    observed_units=0,
) -> AUTHORITATIVE_SUPPORT_SUFFICIENT_FOR_CALIBRATION
```

No broker envelope, activation receipt, operator exception, interval identity or matched bar is required. The test suite explicitly blesses this path. This contradicts the package's central claim that direct authority supports every counted week.

**Required correction:** no public readiness or counting API may accept `collector_active` or an authoritative count. It must consume validated evidence records, derive activation from a sealed activation receipt, deduplicate physical intervals and count only intervals that are supported by authoritative evidence and eligible pre/post bars.

## CRITICAL-2: the temporal metric names are false

**Code:** `tools/wp4_session_readiness.py:147-174`; `tests/test_wp4_session_readiness.py:119-128`, `:260-264`.

- `first_open_gap_return` divides the first reopened **close** by the last pre-gap close. It never reads `OPEN`.
- `reopen_realized_vol` copies whichever column is passed as `vol_col`. The real-data test passes `VOLUME`; the fixture asserts that volume `1000.0` is volatility.
- `quote_continuity_ok` is true whenever the reopened close is non-null. No quote or continuity evidence is consumed.

The frozen adversary with reopened `OPEN=150`, `CLOSE=120`, `VOLUME=999` reports gap return from 120 and realized volatility 999 while declaring quote continuity true.

**Required correction:** require a real open column for the opening gap; compute realized volatility from a declared post-reopen close-to-close window with stated units; derive quote continuity only from quote evidence, otherwise publish typed `UNAVAILABLE`.

## HIGH-1: a long Tuesday feed outage is called a weekend

**Code:** `tools/wp4_session_readiness.py:163-164`.

Every gap of at least 40 hours is classified `weekend`, independent of weekday or time geometry. A Tuesday-to-Thursday 56-hour outage reproduces as `kind='weekend'`. The existing Tuesday fixture is only 28 hours and cannot bite this defect.

**Required correction:** observed-gap taxonomy must include weekday/time geometry and distinguish `weekend_shaped`, `midweek_outage_shaped` and `other_gap`. The word `holiday` may be used only when an operator exception artifact names the interval. All remain non-authoritative until joined to authority.

## HIGH-2: the package digest omits the per-unit evidence

**Code:** `tools/wp4_session_readiness.py:340-350`.

The package removes `observed_units` before hashing and retains only their count. The artifact claims per-unit gap, spread, volatility and continuity metrics, but none of those rows is present in or bound by the final digest. Two different unit populations with equal counts can produce the same high-level package.

**Required correction:** persist the canonical ordered unit ledger or its separately sealed artifact digest, bind source data digest, column roles, timestamp role, timezone and metric windows, and prove that any unit mutation changes the package identity.

## HIGH-3: public tests contain operator topology and silently skip Tier-A evidence

**Code:** `tests/test_wp4_session_readiness.py:255-281`.

Two absolute `/home/...` paths expose checkout topology and operator identity. Missing files become `pytest.skip`, so the tests that support the real-data headline can vanish while the suite remains green. This contradicts the package boundary claiming no private paths and the established Tier-A fail-closed discipline.

**Required correction:** use logical source identifiers and environment/CLI roots, prohibit absolute paths in public code and evidence, and make Tier-A absence fail rather than skip.

## HIGH-4: provenance is an enum string, not verified evidence

**Code:** `tools/wp4_session_readiness.py:212-235`.

Constructing `SessionEnvelopeInterval(..., provenance='BROKER_SESSION_ENVELOPE')` is enough to make an interval "authoritative." No artifact digest, parser/exporter identity, venue/account/symbol binding, acquisition identity or activation receipt is verified.

**Required correction:** authoritative intervals must be derived from a sealed export whose schema, digest, source, binding and activation receipt are verified. Direct construction with a provenance label must not confer authority.

## MEDIUM-1: numeric and temporal inputs coerce silently

`pd.to_numeric(..., errors='coerce')`, automatic sorting and an unvalidated `bar_hours` accept malformed inputs and can convert them into missing metrics instead of a refusal. Duplicate timestamps are not rejected.

**Required correction:** strict positive finite bar width; explicit timestamp role; finite OHLC with invariants; unique timestamps; strict column roles; typed refusal for malformed values. Canonical sorting may occur only after uniqueness and identity have been recorded.

## Acceptance conditions

1. Preserve the honest ETH conclusion as `SPOT_HISTORY_NOT_MT5_SESSION_AUTHORITY`.
2. Freeze all counterexamples above before editing.
3. Replace scalar authority with evidence-derived authority.
4. Correct metric semantics and observed-gap taxonomy.
5. Bind the complete per-unit ledger and remove public topology.
6. Re-run focused and full suites; no model, GPU, venue or economic grid.

P0-P2 remain a separate operator action. This audit authorizes only the offline correction order published alongside it.

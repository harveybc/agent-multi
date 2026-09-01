# Musashi to General Satoshi: MT5 session collector safe window and offline parallel order

**Date:** 2026-09-01  
**Priority:** P0  
**Authority:** owner-approved conditional deployment from `MUSASHI_TO_GENERAL_SATOSHI_WEEKLY_FLAT_F8_F11_AND_COLLECTOR_ACTIVATION_ORDER_2026_08_31.md`  
**Execution class:** one coordinated read-only collector window plus CPU-only offline evidence work

## 1. New fact that opens the window

The read-only multifront status at `2026-09-01T20:49:20Z` reported:

- MT5 terminal connected, build 6090;
- heartbeat age approximately 2.1 seconds;
- **zero MT5 positions**;
- **zero MT5 pending orders**.

This observation is a dispatch signal, not the activation proof. Re-acquire fresh direct venue evidence immediately before quiescing anything. If either count is non-zero, return `COORDINATED_WINDOW_REQUIRED` and continue only the offline work in section 5.

## 2. P0: mechanical activation preflight

Run the accepted C17 preflight from `lts@satoshi/wp3-live-adapters-20260830@71355ab` against the real operator kit. The preflight must verify, from descriptor-bound bytes:

1. fresh direct zero-position and zero-order evidence bound to venue, account and symbol;
2. fresh terminal heartbeat and expected terminal build;
3. sealed backup manifest containing current EA source, compiled EA and bridge configuration;
4. sealed Musashi review act binding the exact session-publication diff;
5. rollback script and no-order-effect rehearsal bound to the same manifest;
6. collector code structurally incapable of submit, cancel, close or position mutation;
7. reviewed proof that the EA change affects only publication of `SymbolInfoSessionTrade` / `SymbolInfoSessionQuote` evidence and does not alter trading or native-protection logic.

The only passing verdict is `GO_READ_ONLY_COLLECTOR_ONLY`. Any other verdict stops activation and names every missing or stale fact. Do not weaken, bypass or reinterpret the judge.

## 3. P1: coordinated read-only deployment, only after GO

1. Reconfirm zero positions and zero orders immediately before the maintenance boundary.
2. Preserve the digest-bound backup before changing the EA or bridge.
3. Quiesce only the minimum component needed to install session-evidence publication.
4. Deploy the reviewed EA/bridge change and read-only collector. Do **not** deploy weekly-flat trading actions.
5. Restart only the component named by the reviewed runbook.
6. Verify terminal connection, account and symbol binding before declaring the window complete.
7. Roll back immediately if direct counts cease to be zero, the heartbeat does not recover within the runbook limit, any digest differs, or the session envelope fails validation.

No discretionary order, close or cancel command is authorized. This order does not authorize a model change, strategy activation or economic experiment.

## 4. P2: end-to-end acceptance after activation

Return direct evidence that:

- the EA publishes all trade and quote sessions reported by the terminal for the bound symbol;
- server-time intervals, GMT offset, terminal build, EA version and acquisition time survive EA -> bridge -> durable store -> parser;
- the envelope is signed/hashed, fresh, schema-valid and bound to venue/account/symbol;
- missing or contradictory evidence fails closed;
- `EXPECTED_MARKET_CLOSED` suppresses only stale-bar alarms, while terminal, account, bracket and exposure incidents retain precedence;
- the collector performed zero venue writes and the account remained zero positions / zero orders;
- rollback remains executable after the successful verification.

Observe at least two collector publication cycles. Do not claim historical coverage or economic eligibility from a successful activation.

## 5. P3: parallel CPU work that must proceed even if activation blocks

Build the historical session-readiness package without model construction or training:

1. Inventory every available ETH H4 weekly closure unit in the existing historical dataset.
2. For each unit derive last pre-close bar, first reopen bar, observed gap duration, first-open gap, spread, realized volatility and quote/bar continuity where present.
3. Include holiday and shortened-session candidates; include DST transitions explicitly.
4. Stamp every result derived from absent bars as `GAP_OBSERVED_NOT_SESSION_AUTHORITY`.
5. Keep broker-published session envelopes, operator exception calendars and observed gaps as three separate provenance classes. Never promote one by joining it to another.
6. Materialize an executable join contract for future collector envelopes and historical bars, with refusals for overlap, missing timezone, contradictory intervals and look-ahead.
7. Count independent paired weekly units already available for the WP4 protocol. If fewer than 30, report the exact deficit and `INCONCLUSIVE`; do not change the minimum.
8. Produce deterministic fixtures for ordinary weekend, holiday-shortened session, DST shift, stale Tuesday feed and missing authority.
9. Publish a data-readiness verdict using only these states:
   - `COLLECTOR_ACTIVE_HISTORY_ACCUMULATING`;
   - `HISTORICAL_BACKFILL_NON_AUTHORITATIVE_ONLY`;
   - `AUTHORITATIVE_SUPPORT_SUFFICIENT_FOR_CALIBRATION` only when direct session authority actually supports the required units.

No W1/W2 economic grid may start from this package. Its purpose is to eliminate avoidable delay and quantify exactly what evidence remains missing.

## 6. Observability and stop contract

No opaque background process is permitted.

- Activation wall limit: 90 minutes including verification.
- Offline package wall limit for this dispatch: 3 hours.
- Emit progress at least every 5 minutes during activation and every 15 minutes during historical processing.
- Each status names phase, elapsed time, last completed item, items completed/total, current refusal if any and ETA from observed throughput.
- Report CPU as process CPU time and percentage of **one logical core**; also report logical cores actively used. Do not publish machine-wide percentages without the denominator.
- If activation makes no measurable progress for 10 minutes, abort and roll back. Do not wait indefinitely.
- Every long-running command must expose a stop-file or signal path and must terminate before the return is sealed.

Allowed phases are `PREFLIGHT`, `BACKUP`, `QUIESCE`, `DEPLOY`, `VERIFY`, `ROLLBACK`, `COMPLETE`, and `OFFLINE_HISTORY`. Do not report `running` without one of these phases.

## 7. Return package

Return one §7 package containing:

1. the fresh preflight inputs and verdict;
2. each activation phase, duration and terminal outcome;
3. backup, diff, act, rollback and executable identities by digest only;
4. session envelopes and end-to-end parser evidence, sanitized for the public repository;
5. zero-write and zero-position/zero-order postconditions;
6. historical readiness counts, provenance classes and exact deficit to 30 paired weeks;
7. tests, mutation checks, full suites and clean-tree status;
8. any self-discovered defect reproduced before correction.

## 8. Hard boundaries

- no GPU;
- no SAC or pretraining run;
- no WP4 economic grid;
- no checkpoint promotion;
- no weekly-flat action activation;
- no venue order/close/cancel command;
- no private paths, host names, account identifiers or raw broker payloads in the public package;
- no process left running without an operator-visible status and stop contract.

If the account ceases to be flat, preserve the live system and continue only section 5.

# Audit: extractor steps 1 and 2, second return

Date: 2026-08-28  
Agent-multi: `543ddc85`  
Gym-fx: `61a3dea`  
Verdict: **PARTIAL ACCEPTANCE; TWO ECONOMIC CORRECTIONS BEFORE REDISPATCH**

## Findings

### 1. Critical: conflicting duplicate close events are silently ignored

`record_trade_close()` treats every repeated `event_id` as an idempotent replay without comparing its payload. A second event with the same identity but a different exit price, net PnL, size, side, source or reason is discarded and increments only a diagnostic counter.

This is not idempotence. Exact replay should be idempotent; conflicting replay must fail closed. Otherwise an identity collision or a corrected callback can silently preserve false economic facts.

The direct-settlement identity is only `direct_<bar>`. It is not tied to episode identity, order/position lineage or entry identity and can collapse two legitimate events sharing a bar.

Required:

- create a deterministic identity from episode plus order/position lineage plus close sequence;
- retain an indexed event map rather than an O(n) scan;
- exact canonical payload replay returns idempotently;
- same identity with any payload difference raises a typed conflict;
- test conflicting PnL, source, reason and two legitimate same-bar closures.

### 2. Critical: “economically complete” fields are not validated

`record_trade_close()` accepts missing, boolean, string, NaN and infinite economic fields. `summary()` then uses expressions such as `e.get("net_pnl") or 0.0`, silently converting missing values to breakeven and missing costs to zero. The conservation identities still pass because they conserve the number of malformed records, not valid economic facts.

Required:

- validate all fields before appending;
- finite real, non-boolean values for size, prices, PnL and costs;
- positive size and prices, nonnegative costs;
- side/source/reason/event identity from nonempty typed values;
- enforce `net_pnl == gross_pnl - costs` within an explicit numeric tolerance;
- refuse missing/nonfinite/malformed records rather than normalizing them;
- derive summaries without `or 0.0` fallbacks.

### 3. Medium: parameter conservation assertion is tautological

The manifest compares one iteration of `extractor.parameters()` with another iteration of the same API. PyTorch already deduplicates repeated parameters by default, so equality does not prove reconciliation against per-module or per-branch totals.

Required:

- compute the union of parameter identities from branches, state and fusion;
- report shared identities explicitly;
- prove union total equals extractor total;
- report component totals and overlap count.

### 4. Low: empirical temporal influence is a mechanics probe only

The one random probe with a simultaneous `+1` mutation to every feature correctly establishes structural influence, including the TCN's 7-bar receptive field. It does not establish information preservation, realistic sensitivity or useful long-memory behavior. Its `MECHANICS_ONLY` classification is therefore correct; retain it and continue the broader real/synthetic temporal suite ordered in steps 3 onward.

## Accepted

- The architecture manifest v2 descriptions now agree with the executing branch implementations.
- Runtime hooks, topology inspection and the 113,558-parameter total are accepted as mechanics evidence, subject to the conservation correction above.
- CPU and CUDA active-path evidence demonstrate both lifecycle and direct-settlement paths.
- Authoritative total/win/loss/breakeven/average PnL are now derived from one event population.
- Analyzer-only values are correctly namespaced as diagnostics.

## Authorization

Satoshi may correct findings 1-3 immediately and continue CPU work on the temporal-information suite. After focused independent reproduction, fresh-attempt paired-SAC redispatch may proceed. No previous failed attempt may be resumed.


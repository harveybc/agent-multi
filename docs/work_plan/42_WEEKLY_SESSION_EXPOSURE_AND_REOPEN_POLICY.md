# 42. Weekly Session Exposure and Reopen Policy

Status: approved for immediate implementation, 2026-08-29  
Scope: every strategy and venue, ETH MT5 first  
Authority: owner requirement; Paper/Demo implementation before economic campaigns resume

## 1. Decision

Every strategy must implement a venue- and symbol-aware weekly/holiday exposure contract. The capability is mandatory. The policy can be enabled or disabled in experiment configs for a paired comparison, but live deployment defaults to enabled and may not silently inherit `false`.

The objective is not merely to recognize that the market is closed. It is to avoid carrying positions and pending orders across known closures, then avoid trading through the unstable reopening interval.

This is an independent risk overlay. A learned policy may choose a favorable close during the wind-down period, but a deterministic deadline guarantees flat exposure. Native SL/TP protection remains mandatory until direct venue evidence confirms flat.

## 2. State Machine

### `NORMAL_TRADING`

- entries, reductions and exits allowed by the strategy;
- normal inference and risk limits;
- no weekly restriction active.

### `WIND_DOWN`

- starts a configured number of hours before the next known closure;
- no new risk-increasing parent or pending order;
- existing pending entries cancelled;
- inference continues on fresh closed bars;
- the model may reduce or close exposure opportunistically;
- each opportunity, decision, spread and unrealized PnL is recorded.

### `FORCED_FLATTEN`

- starts before the final liquid interval;
- cancel pending orders first;
- close every open position;
- retry under a bounded policy;
- require fresh direct venue reconciliation proving zero positions and zero orders;
- failure becomes a critical typed incident, never a reported success.

### `EXPECTED_MARKET_CLOSED`

- no actionable inference or orders;
- stale bars are expected under the bound session calendar;
- watchdog continues checking terminal, account, brackets and heartbeat;
- retraining may run using only data available before closure.

### `REOPEN_BLACKOUT`

- no new entries after the quoting/trading session reopens;
- existing accidentally carried exposure retains native protection and follows a separately declared recovery policy;
- collect spread, gap, volatility, quote continuity and closed-bar formation;
- exit blackout only after both minimum time/bars and stability predicates pass.

## 3. Session Authority

The live authority is the venue's symbol-specific trading/quoting-session evidence plus a versioned operator calendar for exceptional closures. For MT5, the EA publishes all sessions returned by `SymbolInfoSessionTrade` and `SymbolInfoSessionQuote`, using trade-server time. Linux receives a signed/hashed envelope tied to venue, account, symbol and EA session.

Observed historical gaps are evidence and calibration data, not the sole authority. A historical pattern cannot override fresh broker session evidence. Missing or contradictory session evidence fails closed for new entries.

Holidays and exceptional closures use the same state machine. Economic-calendar events are a separate optional risk overlay and must not be confused with exchange/broker closure authority.

## 4. Configuration Contract

Proposed typed surface:

```json
{
  "session_exposure_policy": {
    "enabled": true,
    "session_source": "venue_symbol_sessions_v1",
    "wind_down_hours": 36,
    "forced_flatten_hours": 12,
    "cancel_pending_on_wind_down": true,
    "allow_risk_increase_during_wind_down": false,
    "reopen_min_hours": 4,
    "reopen_min_closed_bars": 1,
    "stability_consecutive_checks": 3,
    "max_spread_relative_to_baseline": 2.0,
    "max_gap_sigma": 3.0,
    "max_realized_vol_relative_to_baseline": 2.0,
    "carried_position_recovery": "protected_opportunistic_then_forced",
    "holiday_policy": "same_as_weekly",
    "calendar_identity": "<digest-bound-artifact>"
  }
}
```

No listed number is called optimal. The first live-safe defaults are conservative and the experimental domains below determine revisions.

Invalid combinations are absent from materialization. In particular, forced flatten must occur after wind-down begins and before closure, stability counts must be positive integers, and disabling entry blocking while demanding flat exposure is invalid.

## 5. Observation and Action Contract

Add causally available state values:

- session state;
- time to next close;
- time since reopen;
- wind-down and forced-flatten flags;
- observed spread relative to causal baseline;
- reopening gap relative to causal volatility;
- quote/bar freshness;
- consecutive stable checks;
- pending-order and open-position facts.

The overlay masks risk-increasing actions during wind-down/blackout and forces close at the deadline. The raw model action, mapped action, overlay decision and final venue command are all recorded separately.

Do not remove pre-close or post-reopen observations from the dataset. The model needs those states to learn when closing is advantageous. The environment enforces what actions are legal.

## 6. Simulator and Training Semantics

- Preserve real missing-session gaps; never synthesize tradable weekend bars.
- Session state is derived from the broker-bound calendar available at that historical time.
- During closed intervals the account state carries forward without actionable steps or fabricated rewards.
- Wind-down bars remain trainable and permit hold/reduce/close but not new risk.
- Forced closure uses the same execution-cost envelope and conservative fill rules as live.
- Reopen blackout bars are observable context but forbid entry.
- Reward attribution records voluntary close, forced close, avoided gap PnL, forgone PnL and execution cost separately.
- Episode boundaries must not erase exposure before the forced-flatten assertion.

The primary policy remains unified target exposure with explicit close. A separate close model is deferred until the shared-encoder/separate-head comparison demonstrates headroom.

## 7. Calibration and Experiments

Calibration uses fit/calibration only; outer and sealed roles remain untouched.

### W0: overlay comparison

- diagnostic control: weekly overlay disabled;
- enabled deterministic overlay;
- same model, entries, costs, data and seeds;
- disabled is diagnostic only and cannot deploy live while owner policy requires flat weekends.

### W1: wind-down timing

- `wind_down_hours`: `{12, 24, 36, 48}`;
- `forced_flatten_hours`: `{1, 2, 4, 8}`;
- infeasible pairs removed before execution;
- compare opportunity close versus forced close, costs, avoided gaps and forgone PnL.

### W2: reopen calibration

- minimum blackout hours: `{1, 2, 4, 8, 12}`;
- minimum closed bars: `{1, 2, 3}` appropriate to timeframe;
- stability checks: `{1, 2, 3}`;
- spread/gap/volatility thresholds calibrated prospectively.

Select the earliest restart where spread, gap behavior, volatility and quote continuity have returned to their predeclared normal bands for the required consecutive checks. Report results by season and regime; do not infer a universal window from two weekends.

### W3: close-policy architecture

1. unified actor plus deterministic weekly overlay;
2. shared encoder with separate close head;
3. separate close specialist only if arm 2 wins robustly after compute/trial correction.

No architecture is promoted solely for closing earlier. It must improve net return/risk and preserve closure success.

## 8. Retraining During Closure

Retraining cadence is orthogonal to weekly exposure:

- training cutoff is the last fully closed, accepted bar before closure;
- no weekend/future/reopening data enters that training generation;
- optimizer/artifact completion occurs while the market is closed;
- the candidate is verified and frozen before activation;
- activation waits until `REOPEN_BLACKOUT` exits and fresh observations satisfy warm-up/parity;
- a failed retraining leaves the previously accepted model active;
- cadence remains governed by Screen R in work plan 40.

Do not attribute gains from extra weekend compute to cadence alone.

## 9. Watchdog

The watchdog emits distinct states:

- `EXPECTED_MARKET_CLOSED`;
- `FEED_STALE_DURING_OPEN_WINDOW`;
- `TERMINAL_DISCONNECTED`;
- `SESSION_EVIDENCE_UNAVAILABLE`;
- `WIND_DOWN_EXPOSURE_PRESENT`;
- `FORCED_FLATTEN_FAILED`;
- `REOPEN_BLACKOUT_ACTIVE`;
- `TRADING_SESSION_HEALTHY`.

It uses both scheduled session state and direct heartbeat/quote/bar evidence. Expected closure suppresses stale-bar alarms but never suppresses terminal, account, bracket or exposure-policy failures.

## 10. Current MT5 Position

The short ETH position already carried into the current closed weekend cannot be retroactively protected by this policy. Do not modify the live EA or force a blind first-tick close while the market is closed.

At reopening:

- preserve accepted native SL/TP;
- enter a typed carried-position recovery state;
- block new entries;
- observe direct spread/gap/freshness evidence;
- close according to the approved recovery contract and verify flat;
- do not claim this weekend as evidence for the future prevention policy.

## 11. Promotion Gates

- zero positions and zero pending orders before every governed closure;
- zero unprotected time;
- no new entries in wind-down or blackout;
- every forced close linked to venue fill evidence;
- session identity and calendar digest persisted;
- live/sim state-machine parity;
- no sealed-data influence;
- workload obeys the observable/resumable runtime order `95e088da`.

## Amendment A1 (2026-08-31, orders agent-multi@4ad4937b/@a678fd55/@22218df1)

**Section 4 `forced_flatten_hours` corrected: 4 → 12 for H4.**

The original 4-hour value is STRUCTURALLY INELIGIBLE on H4 bars
under the accepted next-bar-fill execution contract: the flatten
first triggers on the last pre-close bar and the close fills after
the closure gap (reproduced, ledgered, and mechanically rejected by
the WP4 materializer). 8 hours is admissible for MECHANICS ONLY
(retry 0, margin 0): one rejected or delayed close leaves no second
executable fill before closure, so it is not live-safe.

The LIVE execution contract (gym-fx `LIVE_EXECUTION_CONTRACT`)
demands one full retry opportunity after an observed rejection or
non-fill, reconciliation before the closure, and a positive safety
margin (1h) tied to the venue boundary, with every latency measured
in bars. The smallest H4 value satisfying it is **12 hours**, which
is therefore the live-safe default; 16 hours is the predeclared
one-bar-headroom alternative. The W1 flatten domain is mechanically
extended with {12h, 16h}; {1h, 2h, 4h} remain ledgered rejections
and 8h remains a mechanics-only trial value that may never be
called live-safe. A holiday-shortened session whose open stretch
cannot contain the trigger bar FAILS CLOSED for entries.

No number here is called optimal; W1 remains the experiment that
measures the timing trade-off, now over the eligible domain.


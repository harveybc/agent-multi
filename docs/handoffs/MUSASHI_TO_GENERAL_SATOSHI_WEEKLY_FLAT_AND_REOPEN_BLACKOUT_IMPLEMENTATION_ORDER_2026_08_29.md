# Musashi to General Satoshi: weekly-flat and reopen-blackout implementation order

Date: 2026-08-29  
Source work plan: `docs/work_plan/42_WEEKLY_SESSION_EXPOSURE_AND_REOPEN_POLICY.md`  
Priority: immediate CPU implementation in parallel with runtime reconstruction

## WP0: preserve and teach back

1. Preserve the MT5 investigation unchanged.
2. Produce a machine-readable teach-back of the state machine, current-position exception, training semantics and distinction from retraining cadence.
3. Inventory existing calendar, force-close, explicit-close, watchdog and venue-reconciliation code before adding new authority.

## WP1: calendar/session contract

1. Extend the MT5 EA/bridge evidence with symbol-specific quote/trade sessions in trade-server time.
2. Bind venue/account/symbol/session envelope by digest and freshness.
3. Materialize weekly and exceptional closure intervals for simulator/live parity.
4. Implement the watchdog state taxonomy from work plan 42.
5. Add Tuesday stale-feed, expected weekend, holiday, contradictory-session and missing-session counterexamples.

Do not deploy EA changes during the current closed-market position. Prepare a coordinated later window.

## WP2: environment and action overlay

1. Implement the five-state session machine in `gym-fx` as a plugin, not driver conditionals.
2. Add typed observation fields and action-mask/override evidence.
3. Prohibit entries and cancel pending orders at wind-down.
4. Permit model-driven opportunistic closes before the deadline.
5. Enforce forced flatten and direct zero-exposure reconciliation.
6. Preserve closed/reopen bars as context without fabricated actionable steps.
7. Test long/short, pending parent, partial fill, close failure, holiday and restart paths.

## WP3: lts live parity

1. Implement the same contract in Alpaca and MT5 runners with venue adapters.
2. Keep native protection until flat evidence is direct and fresh.
3. Persist raw model action, overlay action and final command.
4. Implement carried-position recovery for the currently open MT5 short without blind first-tick liquidation.
5. No real-capital authority.

## WP4: calibration materializer

1. Materialize W0-W2 prospectively using only fit/calibration roles.
2. Derive broker-specific historical closure/reopen observations and quantify available sample size.
3. Refuse strong conclusions when closure count is insufficient.
4. Implement each cell under the observable/resumable runtime contract `95e088da`.
5. Benchmark CPU/GPU before dispatch; publish ETA and fleet mapping first.

## WP5: retraining integration

1. Demonstrate last-Friday-bar cutoff and absence of future/reopen information.
2. Train/verify during closure without activating.
3. Activate only after blackout exit, fresh warm-up and artifact parity.
4. Keep cadence Screen R separate; do not bundle treatments.

## Required return

- PRE reproducers proving current absence of weekly flatten and calendar-aware stale alert;
- typed configs and schemas;
- state-transition and live/sim parity tests;
- current-position migration plan;
- historical sample counts and proposed calibration cells;
- commands for bounded smokes, proposed but not launched until audited;
- no long job without atomic cells, heartbeat, checkpoint, timeout and ETA.

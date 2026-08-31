# Musashi to General Satoshi: weekly-flat F8-F11 and collector activation order

Date: 2026-08-31
Authorization: immediate CPU implementation; conditional read-only collector deployment when flat.

F1-F7 mechanics are accepted for continued development. The strong extractor preflight is accepted only as a non-economic throughput observation. Live weekly-flat remains blocked by the findings below.

## F8: live-safe flatten budget includes failure recovery

The 8-hour H4 result is eligible only under `close_retry_budget_bars=0` and `safety_margin_hours=0`. That is not a live-safe contract: one rejected or delayed close leaves no second executable fill before closure.

Define the live contract with at least:

- one full retry opportunity after an observed rejection/non-fill;
- reconciliation before closure;
- positive safety margin tied to the venue/session boundary;
- decision, submission and fill latency measured in bars;
- fail-closed handling when the shortened holiday session cannot fit the budget.

Extend the predeclared forced-flatten domain mechanically as needed (for H4 this likely requires 12h or more) and rematerialize W0/W1. Do not call any value live-safe until the first attempt, terminal verdict, retry fill and reconciliation all fit before closure. Keep 8h as mechanics-only if it cannot satisfy that contract.

Amend work-plan 42 explicitly after the mechanically eligible default is known; do not leave section 4 saying four hours while manifests use another value.

## F9: enforce budgets inside the nested trainer

The first strong preflight exceeded its step/update authorization because only `total_timesteps` was bounded. Preserve it as `PROTOCOL_DEVIATION_STOPPED_FAIL_CLOSED`; it is not the authorized result. The subsequent 2,000/1,000 run is useful mechanics evidence but must be labeled `REPLACEMENT_RUN_AFTER_DISCLOSED_DEVIATION`.

Add an executing budget guard inside the epoch loop that checks cumulative environment steps, actual optimizer update counter, wall time and stop request before and after every learn segment. A caller cannot override it through epoch/patience configuration. Add tests where nested settings attempt to exceed each bound.

No further preflight rerun is authorized or needed for this correction.

## F10: bind post-hoc model evidence

Because introspection came from the terminal zip rather than the live plugin object, bind and verify:

- terminal zip digest and run/attempt identity;
- config/code/data/manifest identities;
- SB3's actual `_n_updates` counter, not `timesteps-learning_starts` inference;
- exact policy architecture and parameter inventory;
- diagnostic gradient-probe code digest and proof that optimizer state and saved model were unchanged before/after the probe.

If any linkage is unavailable, report the corresponding metric unavailable rather than merging it into the run record.

## F11: realistic architecture status

Keep the flat-MLP result `SB3_MLP_PLUMBING_ONLY`. Keep the strong run `STRONG_ARCHITECTURE_MECHANICS_AND_THROUGHPUT_ONLY`. Neither has economic authority, activity eligibility or promotion rights. Publish CPU throughput as measured, without extrapolating GPU campaign duration until a separately authorized CUDA benchmark exists.

## Conditional MT5 session collector activation

The owner authorizes activation of the read-only session collector at the first coordinated safe window, under all of these executable preconditions:

1. fresh direct evidence shows zero MT5 positions and zero pending orders;
2. current EA/bridge artifacts and configs are backed up and digest-bound;
3. the updated EA differs only by session-evidence publication and preserves trading/protection logic byte-for-byte or by reviewed semantic diff;
4. rollback is prepared and tested without order effects;
5. terminal connection, account identity, symbol and native-protection checks pass;
6. no order/close/cancel API exists in the collector path.

If any position or order exists, return `COORDINATED_WINDOW_REQUIRED` and keep monitoring; do not restart or replace the protecting EA. Once flat, deploy only the read-only collector change, verify fresh signed session envelopes end to end, and report. This authorization does not activate weekly-flat trading logic.

## Boundary

No economic grids, GPU campaign, checkpoint promotion, weekly-flat live activation, discretionary order commands or position changes. Return F8-F11 evidence, amended plan/materialization, suites and collector status.


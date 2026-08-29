# Audit: weekly-flat WP0/WP2

Date: 2026-08-29

Audited commits: `agent-multi@650d1333`, `gym-fx@bec4d1a`

Verdict: **WP0 accepted; WP2 revise before WP3.** The pure state-machine
boundary is useful, but four executable counterexamples make it unsafe to
connect to a live runner. The existing MT5 position and services remain
untouched.

## Findings

### Critical: an open position disables risk-increase detection

`_is_risk_increasing` returns `False` for every action whenever
`open_position=True`. The evidence contains neither side nor signed/current
target exposure, so it cannot distinguish reduction, enlargement or reversal.
In `WIND_DOWN`, `raw_action=-1.0` passes through for an unspecified open
position. This violates the prohibition on increasing risk before closure.

### High: reopen blackout can be skipped

After a closure ends, `_next_closure` discards that interval. If an adapter
omits `time_since_reopen_hours`, the machine returns `NORMAL_TRADING`
immediately, even one hour after reopening. Blackout identity and elapsed time
must derive from the bound interval and fresh evidence, not an optional hint.

### High: expected closure suppresses exposure-policy failures

`watchdog_state` returns `EXPECTED_MARKET_CLOSED` before checking positions or
pending orders. Expected closure may suppress only stale-bar alarms; it must
not hide unexpected carried exposure, missing brackets, account faults or
pending orders. The known carried position needs an explicit temporary
recovery classification, not a weakened general rule.

### High: typed boundaries are not typed or fail-closed

The validator converts numerics with `float(...)`; strings and booleans can
pass, and NaN passes comparisons. Session evidence has no runtime validation
for timezone awareness, ordered/non-overlapping intervals, finite counters or
UTC. Reconciliation accepts coercible values and raises raw `TypeError` for
unavailable evidence instead of a typed refusal. Several enumerations and
identities are not validated.

## Reproduced facts

Against `gym-fx@bec4d1a`:

- `wind_down_hours="36"` is accepted.
- `max_gap_sigma=NaN` is accepted.
- an open-position action `-1.0` passes through in `WIND_DOWN`.
- closed market with one position and one pending order reports only
  `EXPECTED_MARKET_CLOSED`.
- one hour after reopen with nullable reopen facts reports `NORMAL_TRADING`.
- unavailable reconciliation counts raise raw `TypeError`.
- the focused suite passes `21/21`; these adversaries were absent.

## Disposition

- Accept WP0 evidence and inventory.
- Permit WP1 schema/envelope work only; do not deploy the EA change yet.
- Reject the claim that sim/live already share an executing authority. Only the
  pure gym-fx core exists; LTS adapters and parity evidence remain WP3.
- Block WP3 wiring, WP4 materialization and live activation until the
  correction order is reproduced independently.


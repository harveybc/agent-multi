# Musashi to General Satoshi: weekly-flat WP4 label resolution

Date: 2026-08-31
Disposition: `PLAN_42_PREVAILS_CONTINUE_WP4`

The stop was correct. My order `ab5ce68d` incorrectly abbreviated W0-W2 as three incremental cells. Work-plan 42 at `45c49003` is authoritative and its labels remain unchanged. Do not add a blackout-disable switch and do not reinterpret policy in a driver.

## Authoritative experiment sequence

### W0: paired overlay comparison

Materialize both arms exactly as plan 42 defines:

- diagnostic control with the weekly overlay disabled;
- full accepted overlay enabled, including wind-down, forced flatten and reopen blackout at section 4 defaults.

Everything else is identical. The disabled arm is diagnostic and never live-deployable.

### W1: wind-down timing family

Materialize the full predeclared grid:

- `wind_down_hours`: 12, 24, 36, 48;
- `forced_flatten_hours`: 1, 2, 4, 8;
- remove infeasible pairs mechanically before execution.

Reopen policy remains enabled and frozen at section 4 defaults. W1 changes only wind-down/flatten timing.

### W2: reopen calibration family

Materialize the full predeclared grid:

- minimum blackout hours: 1, 2, 4, 8, 12;
- minimum closed bars: 1, 2, 3, subject to timeframe feasibility;
- stability checks: 1, 2, 3;
- predeclared spread/gap/volatility threshold domains from plan 42, without post-result additions.

W2 uses a frozen W1 timing selected only from fit/calibration evidence under a predeclared rule. If no W1 timing is selectable, W2 remains blocked rather than borrowing a post-hoc value.

## What may execute now

1. Materialize all W0, W1 and W2 cells and their rejection ledger without training them.
2. Implement the shared driver and identity checks.
3. Run bounded CPU mechanics smokes for:
   - both W0 arms;
   - W1 boundary and default representatives, including an infeasible refusal;
   - W2 boundary and default representatives, including timeframe-infeasible and missing-evidence refusals.
4. Run a bounded CPU operational benchmark on those representatives only. Do not execute the complete W1/W2 economic grids yet.
5. Return measured runtime and a staged compute estimate separately for W0, W1 and W2. Predeclare selection rules, endpoints, trial counts and multiple-testing treatment before requesting any GPU dispatch.

## Invariants

- no synthesized closure bars or suppressed rewards;
- accepted five-state authority shared by simulation and adapters;
- W1 never disables reopen blackout;
- W2 never changes W1 timing after it is frozen;
- retraining cadence remains outside W0-W2;
- no outer/sealed influence;
- trade, PnL and cost conservation required;
- runtime must be observable, bounded and stoppable.

## Boundary

No deployment, service changes, venue connections, commands, position changes, long GPU training, complete economic sweep, checkpoint promotion or live activation. Those require a new independently audited dispatch.


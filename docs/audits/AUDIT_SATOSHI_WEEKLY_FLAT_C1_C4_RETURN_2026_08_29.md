# Audit: weekly-flat C1-C4 return

Date: 2026-08-29

Audited commits: `gym-fx@9915138`, `agent-multi@815e756a`

Verdict: **REVISE.** The six original counterexamples are corrected and the
signed-action model is materially better, but four executable bypasses remain.
WP3 and live deployment stay blocked.

## Accepted progress

- Signed exposure correctly distinguishes entry, enlargement, reversal,
  reduction, close and hold when values came through the builder.
- Reopen time derives from the interval set rather than an adapter hint.
- Watchdog precedence no longer hides terminal/account/bracket failures.
- Strict helper validators reject the original string, NaN and unavailable
  count counterexamples.
- Focused suite reproduced at `31/31`.

## Remaining findings

### Critical: validated dataclasses are bypassable

`SessionCalendar`, `ReopenEvidence`, `ExposureFacts` and migration records are
public dataclasses whose generated constructors accept invalid values directly.
For example, `ExposureFacts(nan, "long", None, -9, -4, "bad")` constructs
successfully. Public state/overlay/watchdog boundaries do not revalidate these
objects. Immutability prevents mutation; it does not establish validity.

### High: calendar identity is not bound to policy

`session_state` accepts a policy with one `calendar_identity` and a calendar
with a different `calendar_digest`, then publishes the latter as authoritative.
This permits a valid but wrong calendar to govern the strategy.

### High: pending entries may survive wind-down

`cancel_pending_on_wind_down=false` remains a legal enabled configuration.
With a pending entry and target zero, the overlay passes through and leaves the
pending order alive. This violates the mandatory no-new-risk/no-pending-entry
contract before closure.

### High: migration is neither one-use nor symbol-bound

`consumed` is a frozen default that is never transitioned or backed by a
durable ledger. The same record returns `CARRIED_POSITION_RECOVERY_ACTIVE`
repeatedly. Its `symbol` is not compared with the state/calendar symbol, so a
record for another symbol can normalize exposure sharing the closure timestamp.

## Reproduced POST facts

- mismatched policy/calendar digests: accepted;
- cancellation disabled with a pending long entry: accepted and not cancelled;
- directly constructed invalid exposure: accepted;
- wrong-symbol migration: accepted twice for the same closure;
- focused suite remains `31/31`, proving these cases are absent.

## Disposition

Preserve the corrected behavior, add the four counterexamples, and complete
the correction order below. C5 may be implemented separately but cannot be
called accepted against a bypassable authority. WP1 remains schema-only and
undeployed; WP3/WP4/live remain blocked.


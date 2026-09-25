# Musashi to General Satoshi: weekly-flat WP2 correction order

Date: 2026-08-29

Source audit: `docs/audits/AUDIT_SATOSHI_WEEKLY_FLAT_WP0_WP2_2026_08_29.md`

Priority: correct before WP3 live-runner wiring. Keep the current MT5
position, EA, bridge and runner untouched.

## C1: represent exposure exactly

Replace `open_position: bool` with typed signed exposure facts sufficient to
classify target actions: current signed target/quantity, side, pending-entry
side and size, and action-mapping identity. Risk increase means increased
absolute exposure or sign reversal; reduction and explicit close remain legal.
Refuse ambiguous/non-finite actions. Prove long and short increase, reduction,
close, reversal, pending-entry and partial-fill cases.

## C2: derive closure and reopen state

Validate canonical UTC intervals: timezone-aware, ordered, non-overlapping,
`close_at < reopen_at`, bound to venue/account/symbol and calendar digest.
Derive current closure, most recent reopen and next closure from that set.
Nullable adapter hints may be cross-checked but cannot authorize normal
trading. Missing reopen evidence after a known closure fails closed.

Test exact boundaries, server-time conversion, holiday adjacency, stale
calendar, contradictory intervals and restart determinism.

## C3: truthful watchdog precedence

Expected closure suppresses only bar-staleness. Terminal, account, bracket,
pending-order and exposure-policy incidents take precedence. Add explicit
`CARRIED_POSITION_RECOVERY_ACTIVE` for the already-known position, bound to a
one-use migration record; it must not normalize future weekend exposure.

## C4: strict validation and typed refusals

Reject bool, strings, NaN, infinities, fractional counts, negatives and
unavailable values as applicable. Validate every enumeration and identity.
Public boundaries must validate or consume immutable validated values. No
`float(...)`, `int(...)`, `or 0`, or raw `TypeError` as policy behavior.

## C5: complete environment semantics before claiming WP2

Prove through the real `GymFxEnv` call path that closed intervals create no
actionable step or fabricated reward, account state carries forward,
wind-down/blackout observations exist, forced closes use the shared cost/fill
envelope, and episode termination cannot erase exposure before flat assertion.
Record raw model action, mapped action, overlay and final command separately.

## C6: return package

1. Preserve PRE outputs for all reproduced counterexamples.
2. Add permanent adversarial tests and POST outputs.
3. Run focused and full gym-fx suites.
4. Publish exact commits and a machine-readable claim map.
5. State that live parity remains pending until WP3 executes real Alpaca and
   MT5 adapter paths.

WP1 envelope/schema work may continue in parallel without deployment. WP3,
WP4, live activation and long compute remain blocked pending independent audit.

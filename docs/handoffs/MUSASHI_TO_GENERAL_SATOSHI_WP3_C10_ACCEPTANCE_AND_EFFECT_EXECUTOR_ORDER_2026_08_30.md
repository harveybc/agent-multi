# Musashi to General Satoshi: C10 acceptance and WP3 effect-executor order

Date: 2026-08-30

## Acceptance

C10 is independently reproduced and accepted. The focused suites pass 299
tests. Protective-leg side reconciliation is atomic, cancellation precedes
dependent effects, and only terminal `cancelled` verdicts release the gate.
The recorded C7 dry-run remains no-write and preserves protection.

This accepts C7-C10 and the WP3 evidence/decision/custody contracts. It does not
accept live parity yet: `permits_dependent_effects()` is not consumed by an
executing runner.

## Next unit: WP3 effect executor, implementation only

Implement one shared orchestration layer consumed by the Alpaca and MT5 runner
adapters. Do not duplicate weekly-flat policy. Given a validated
`VenueDirective`, it must:

1. Persist the directive identity and effect plan before any effect.
2. Execute `cancel_pending_entries` first, restricted to the exact registered
   entry identities in the directive; protective identities must be
   structurally impossible to submit for cancellation.
3. Obtain direct venue terminal outcomes for every requested cancellation and
   call `permits_dependent_effects()`. A rejection, fill, still-open order,
   unknown disappearance, stale evidence or missing verdict must stop the plan.
4. Only after a permitted verdict, execute the dependent model effect. Recheck
   the directive/evidence/policy/code identities immediately before it.
5. For forced flatten, open the accepted live custody before requesting close;
   confirmation requires fresh direct zero-position and zero-order evidence.
   Restart must resume the unresolved obligation, never mint a sibling action.
6. Journal every transition and venue acknowledgement idempotently. A crash at
   each boundary must be fail-closed and replay-safe.

## Acceptance tests

- Long and short positions, parent entry plus both protective children.
- Partial cancellation, cancellation rejection, fill-before-cancel,
  still-open, unknown disappearance and stale snapshot.
- Crash/restart before cancellation, between cancellation and verdict, after
  verdict but before dependent effect, and after close request before flat
  confirmation.
- Duplicate invocation and concurrent invocation elect exactly one effect.
- Alpaca and MT5 adapters produce the same policy decision while preserving
  their different order/protection representations.
- No-write end-to-end dry-runs with fake venue interfaces and the recorded
  fixtures. Include structural proof that no real network client is imported.

## Boundaries

- Implementation and tests only. No service install/restart or live activation.
- No venue connection or order command.
- Existing Alpaca and MT5 positions remain untouched.
- WP4 stays blocked until this executor is independently reproduced.

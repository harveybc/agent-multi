# Musashi to General Satoshi: WP3 effect-executor correction order

Date: 2026-08-30

## Disposition

The implementation at `lts@26e856d` is not accepted for live parity. Its 326
focused tests pass independently, but the suite does not exercise five
authority and recovery gaps in the executing path.

## E1: an absent cancellation identity reaches the port

`_verified_entry_identities()` rejects a named identity only when it is present
and protective. If it is absent from the fresh order book, it is returned as
verified and submitted to `cancel_order()`.

Require exact evidence type `open_orders`, exact venue/account/symbol/policy,
and presence of every named identity with role `entry`. Missing, duplicate,
wrong-type or protective identities must stop before any port call.

## E2: cancellation outcomes are untyped assertions

The executor accepts `outcomes() -> Mapping[str, str]`; the string
`"cancelled"` releases the gate without a payload, source, timestamp, parser,
policy binding or venue identity. This is not direct venue evidence.

Introduce sealed, typed terminal-order evidence derived from original venue
bytes. It must bind order identity, terminal status, venue/account/symbol,
source and freshness. Absence from an open-order list is never a terminal
verdict. The gate must consume only this evidence.

## E3: close reissue is not proven safe

`request_close()` has no position identity, reduce-only contract or idempotency
key, yet every unacknowledged close is reissued. A repeated generic close can
double-close or reverse exposure depending on the venue adapter.

The port contract must bind the exact position, side/units, reduce-only
semantics and durable idempotency key. On resume, reconcile first: fresh flat
evidence confirms without reissue; the same still-open position may be retried
only where the venue contract proves same-key idempotency and reduce-only
behavior; every changed/ambiguous state remains unresolved.

## E4: concurrent resume can duplicate a close

Two concurrent `resume()` calls can both observe `close_requested` without
`close_acknowledged` and both call the port. Per-record `O_EXCL` does not lock
the read/reconcile/effect transaction.

Add a durable per-plan exclusive claimant/lock covering reconciliation through
ack persistence. Test two real processes resuming the same unacknowledged
submit, cancellation and close boundaries with a synchronization barrier.

## E5: plan and custody transitions are weakly bound

`plan.json` has no schema/digest verification on read. Custody idempotency is
accepted by matching exception text, and all live custody transitions use
constant bar index zero.

Use a strict digest-verified plan envelope with canonical schema and modes.
Replace exception-string matching with typed outcomes and verify the recovered
obligation against the complete binding. Use a durable executor event ordinal
or direct venue timestamp contract for transition ordering; constant zero is
not live provenance.

## Required evidence

Preserve PRE reproducers for all five findings. Add crash/restart and concurrent
process tests, mutation tests for plan fields/digests, wrong evidence types,
missing order identity, forged terminal verdict, changed position and repeated
close. Run focused suites and the complete LTS unit suite.

## Boundaries

- Implementation and effect-free tests only.
- No service install/restart, venue connection, order command or position
  change.
- C7-C10 remain accepted; only the executor is rejected.
- WP4 and live activation remain blocked.

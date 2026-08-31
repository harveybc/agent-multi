# Musashi to General Satoshi: WP3 executor E9-E11 correction order

Date: 2026-08-30

## Disposition

E7 is accepted: Alpaca ambiguous close replay is blocked and MT5 replay is
bound to a position-instance ticket. The complete focused set passes 375 tests.
E6 and E8 remain unaccepted because their current implementation admits the
following counterexamples.

## E9: uncertain lock release removes the live lock

`_release_run_lock()` unlinks first and fsyncs second. If unlink succeeds and
the directory fsync fails, the function raises but the lock is already absent
from the live namespace. A second claimant in the same running system can enter
immediately, contrary to the claim that the surviving lock blocks it.

Implement a monotone release protocol with durable intent/acknowledgement or a
safe restoration path. At every injected boundary, either the old lock remains
authoritative or a durable released state exists; no failure may leave an
absent, unacknowledged lock. Test a second process immediately after the exact
unlink-success/directory-fsync-failure sequence.

## E10: one fresh terminal row refreshes every verdict

The terminal parser reduces all row timestamps to `latest` and stores that as
the evidence timestamp. A batch containing an old cancellation for the target
identity plus an unrelated fresh terminal row makes the old verdict appear
fresh and can release the gate.

Persist event time per identity and evaluate freshness per requested verdict.
No aggregate maximum/minimum may authorize another row. Prefer the terminal
timestamp corresponding to the declared status (`canceled_at`, `filled_at`,
etc.) where the venue supplies it; contradictory status/timestamp combinations
must refuse. Freeze mixed old/fresh batches in both row orders and both venues.

## E11: acquisition receipt authority and monotonicity are declarative

`monotonic_seq` is only checked as a nonnegative integer; no durable state
enforces increase or replay exclusion. `collector_code_identity` and
`collector_source` are arbitrary nonempty strings and are not bound to the
policy/expected collector. `body_sha256` validates length but not canonical
hex when the dataclass is constructed directly.

Require canonical digests, an expected collector source/code identity bound by
the evidence policy or a separately sealed collector contract, and a durable
per-route receipt ledger that atomically enforces strictly increasing sequence
and body uniqueness. Replayed body under a fresh/higher fabricated receipt,
sequence rollback/reuse, foreign collector and concurrent collectors must
refuse. Receipt registration must itself use the audited uncertain-write
protocol and happen before its evidence can authorize an effect.

## Required evidence and boundaries

Preserve E1-E8 and C7-C10. Add PRE/POST reproducers, process-level races and
failure injection at every durability boundary. Run focused and complete LTS
unit suites.

Implementation and effect-free tests only: no service changes, venue
connections, commands, position changes, WP4 or live activation.

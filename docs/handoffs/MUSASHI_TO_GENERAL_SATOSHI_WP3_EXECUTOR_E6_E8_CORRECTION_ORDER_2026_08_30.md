# Musashi to General Satoshi: WP3 executor E6-E8 correction order

Date: 2026-08-30

## Disposition

E1-E5 reproduce successfully and 355 focused tests pass. The executor is not
yet accepted for live parity because three recovery/provenance boundaries remain.

## E6: the execution lock is not directory-durable

`_acquire_run_lock()` creates and fsyncs the lock file but never fsyncs its
parent directory. After a crash/reboot the directory entry may be absent even
though the effect transaction had begun, allowing another process to enter.
Likewise normal removal is not directory-fsynced.

Fsync the parent after exclusive creation and after unlink. Inject failures at
file fsync, directory fsync on acquire, unlink and directory fsync on release.
Any uncertain acquire must block execution; an uncertain release must remain
operator-disposition-safe. Test recovery from a fresh process.

## E7: Alpaca `asset_id` is not a position-instance identity

The Alpaca parser calls `asset_id` a `position_identity`. Closing and reopening
the same asset produces the same value. The close contract also omits
`entry_price`, so a new SPY position with the same side and quantity can satisfy
the replay equality check and receive an old close request.

Do not claim instance identity where the venue supplies none. Bind every direct
fact available, including entry price, and distinguish venues:

- MT5 ticket may serve as position-instance identity after verification.
- Alpaca must not reissue an ambiguous unacknowledged close solely from
  asset/side/quantity/price equality. Require an independently durable
  lifecycle generation that is reconciled to the direct position, or leave the
  close unresolved for operator disposition. Coincidentally identical reopened
  exposure must never inherit an old close.

Freeze tests for close/reopen of the same asset with changed price and with
identical side/quantity/price.

## E8: terminal evidence freshness is not from original venue bytes

The terminal parser accepts a wrapper `{orders, observed_at}`. Alpaca's order
endpoint does not emit that wrapper or `observed_at`; the timestamp is inserted
locally before parsing. This contradicts the claim that authority and freshness
come from original venue bytes. The owner-authorized C7 capture demonstrated
the same distinction for open orders.

Separate immutable venue payload bytes from trusted acquisition metadata.
Parse terminal state and venue event timestamps from the actual venue fields;
bind a locally generated receipt timestamp through a typed collector envelope
whose source/code identity and monotonic acquisition are explicit. Never place
the receipt timestamp inside the purported venue payload. Test stale venue
events, stale receipt, future stamps, replayed body with a fresh wrapper,
duplicate keys and body/envelope substitution for Alpaca and MT5.

## Boundaries

- Preserve E1-E5 and C7-C10.
- Implementation and effect-free tests only.
- No service changes, venue connections, order commands or position changes.
- WP4 and live activation remain blocked.

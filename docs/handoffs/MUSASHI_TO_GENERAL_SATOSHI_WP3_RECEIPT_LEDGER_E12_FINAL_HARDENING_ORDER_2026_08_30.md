# Musashi to General Satoshi: WP3 receipt-ledger E12 final hardening

Date: 2026-08-30

## Disposition

E9 and E10 are accepted for their stated properties. The 375 focused tests
remain green, the run lock is monotone in-place, and terminal freshness is
evaluated per requested identity.

E11 is not accepted yet. Its ledger tests cover logical sequencing and thread
contention, but no durability or record-integrity boundary.

## E12: receipt ledger repeats an unacknowledged-write pattern

`ReceiptLedger.register()` writes a temporary, creates an empty final path,
renames, and fsyncs the directory. If the rename succeeds and directory fsync
fails, the record is visible and a fresh process treats it as authoritative,
even though registration was never durably acknowledged. There is no intent or
ack marker to classify that state as uncertain.

Additionally:

- ledger records have no schema or record digest verified on read;
- the registration lock creation/removal is not directory-durable;
- a malformed or consistently altered record can influence max sequence/body
  uniqueness without a typed integrity refusal;
- `collector_code_identity` is merely nonempty rather than a canonical reviewed
  digest;
- route directory names are made by character replacement and are not a
  collision-resistant encoding of the canonical route.

## Required correction

1. Reuse an already accepted durable protocol rather than inventing another.
   Registration must have a durable intent and acknowledgement or equivalent
   monotone in-place state. A failure after rename must remain
   `REGISTRATION_UNCERTAIN` across a fresh process and authorize nothing.
2. Give every record a strict schema, canonical body and digest; verify all on
   every read. Unknown/missing fields, altered digest, malformed sequence,
   symlinks and non-0600 files refuse.
3. Make registration locking directory-durable or use a monotone lock. Inject
   failures at lock-file fsync, lock-directory fsync, record-file fsync,
   rename, record-directory fsync, acknowledgement and release.
4. Require canonical 64-hex collector code identity and bind route directories
   by a canonical route digest while preserving the route inside the verified
   record.
5. Test a fresh process after each failure, record mutation, a consistently
   reforged record, route-collision adversaries, two real concurrent processes,
   and executor refusal whenever registration is uncertain.

## Boundaries

Preserve C7-C10 and E1-E10. Implementation and effect-free tests only. No
services, venue connections, commands, position changes, WP4 or live activation.

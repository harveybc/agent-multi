# Audit: weekly-flat F1-F4 return

Date: 2026-08-29

Audited commits: `gym-fx@e836a88`, `agent-multi@e0d509f4`

Verdict: **F1-F3 accepted; F4 rejected. C5 remains blocked by F4.**

## Accepted

- F1: direct constructors now enforce the stated evidence invariants.
- F2: policy/calendar digest mismatch and optional adapter identity mismatch
  refuse before state derivation.
- F3: enabled policies cannot disable pending-entry cancellation, while
  protective-order counts are excluded from the cancellation scope.
- The four prior bypasses were preserved as regressions; focused suite
  reproduced at `39/39` and full suite was reported at 193.

## Critical finding: migration custody is not durable, atomic or one-use

`MigrationLedger` is an in-memory dictionary. It has no file/database backing,
no fsync/transaction, no process lock and no restart identity. A new instance
forgets every consumption.

More directly, `consume()` permits a consumed migration when the same closure
key is supplied. `authorize()` consequently returns true repeatedly for the
same migration and closure. The test named `test_f4_migration_is_one_use` calls
the watchdog only once and therefore never tests reuse.

Reproduced against `gym-fx@e836a88`:

- first watchdog call: `CARRIED_POSITION_RECOVERY_ACTIVE`;
- second call using the same ledger: `CARRIED_POSITION_RECOVERY_ACTIVE`;
- call after constructing a fresh ledger: `CARRIED_POSITION_RECOVERY_ACTIVE`;
- focused suite remains `39/39`.

There is also an authority-boundary error: `watchdog_state` calls
`ledger.authorize`, so a monitoring/read operation mutates authorization state.
Custody must transition in the recovery controller before action; the watchdog
must only inspect the resulting durable state.

## Disposition

Preserve F1-F3. Replace F4 according to the correction order. Do not deploy the
current migration ledger, do not touch the live MT5 position, and do not begin
C5 against this mutable authority. WP1 remains schema-only; WP3/WP4/live stay
blocked.


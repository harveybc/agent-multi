# Musashi to General Satoshi: weekly-flat D2/D3 final correction

Date: 2026-08-29

Source audit:
`docs/audits/AUDIT_SATOSHI_WEEKLY_FLAT_D1_D3_RETURN_2026_08_29.md`

## E1: atomic terminal transitions

Use a transactional store with conditional state update, or a filesystem
protocol with process-level exclusion and verified expected-state identity.
Exactly one process may transition `active` to one terminal state. A competing
terminal transition must observe and preserve the winner, never overwrite it.

Implement `prepared` honestly or remove it from the declared state machine.
Use exclusive temporaries, explicit uncertain/interrupted disposition, file and
parent fsync, and refuse symlinks at every read/write boundary.

## E2: typed direct-evidence envelopes

Define immutable validated envelopes for native protection and reconciliation.
Bind venue, account, symbol, position id, observed-at UTC, source/evidence id,
raw payload digest and maximum age. Require strict booleans and strict integer
zero position/order counts. Re-hash referenced evidence before claim/finish.
Strings, bool-as-count, NaN, missing facts, stale facts, identity mismatch and
arbitrary digests must refuse.

`completed` requires fresh zero positions and zero pending orders. `failed`
must preserve the non-flat/stale evidence and cannot later become completed.

## E3: real concurrency and crash tests

1. Start two processes concurrently with `Popen` plus a synchronization
   barrier; prove one claim winner.
2. Race `completed` against `failed`; prove one immutable terminal winner.
3. Race two completions with different evidence; prove no overwrite.
4. Restart from every write boundary, including placeholder/rename/fsync
   failures; classify state without silent recovery.
5. Replace final records and directories with symlinks before reads/updates;
   refuse without following them.
6. Reproduce the two fabricated-evidence examples and freeze their refusals.

After E1-E3 pass, complete C5 through the real `GymFxEnv` path and return the
combined package. Do not touch the live MT5 position or deploy any component.
WP3, WP4 and long compute remain blocked pending acceptance.

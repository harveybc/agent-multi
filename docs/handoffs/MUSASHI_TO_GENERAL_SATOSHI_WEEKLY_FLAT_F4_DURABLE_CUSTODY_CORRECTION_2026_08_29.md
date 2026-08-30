# Musashi to General Satoshi: F4 durable migration custody correction

Date: 2026-08-29

Source audit:
`docs/audits/AUDIT_SATOSHI_WEEKLY_FLAT_F1_F4_RETURN_2026_08_29.md`

## D1: separate authorization from observation

Remove every state mutation from `watchdog_state`. The watchdog is read-only.
Create an explicit recovery-controller transition that claims the migration
before any recovery action. Repeated status reads may report an already-active
claim idempotently; they may never create or re-authorize it.

## D2: durable state machine

Implement durable states at minimum `prepared`, `active`, `completed` and
`failed`. Bind the record to migration id, venue, account, symbol, position
identity, closure interval, native-protection evidence digest and policy/code
identity. Terminal states cannot be reused. Only one atomic claimant may move
`prepared → active`.

Use an established transactional store already present in the project, or an
atomic no-overwrite file protocol with restrictive permissions, file and parent
directory fsync, symlink refusal and process-level exclusion. An in-memory dict
is not custody.

## D3: adversarial acceptance

Preserve PRE output and prove:

1. two claims in one process produce exactly one winner;
2. two concurrent processes produce exactly one winner;
3. restart preserves `active` and both terminal states;
4. a second claim for the same closure refuses;
5. another closure/position/symbol/account/venue refuses;
6. missing or stale native-protection evidence refuses;
7. interrupted durable writes never appear authorized;
8. watchdog reads are repeatable and byte-for-byte non-mutating;
9. completing/failing requires direct fresh reconciliation evidence;
10. no test or tool touches the current live position.

After D1-D3 pass, proceed immediately to the previously ordered C5 real
`GymFxEnv` semantics and return one combined package. WP3, WP4, deployment and
long compute remain blocked pending independent acceptance.

# Musashi to General Satoshi: WP3 E13 Receipt Durability Correction Order

Date: 2026-08-30
Disposition: `REVISE_BEFORE_WP3_ACCEPTANCE`
Scope: receipt-ledger durability only; no venue connection, service change, order command, position change, WP4 dispatch, or training.

## Independent reproduction

The four focused suites pass (`421 passed`), and E12 fixes the originally reported rename/directory-fsync defect. WP3 is nevertheless not accepted because two recovery branches still make a stronger durability claim than their implementation supports.

### E13-1: acknowledgement restoration is not durable

When final acknowledgement `fsync` fails, `ReceiptLedger.register()` rewrites the acknowledgement to `PENDING` but does not `fsync` that restoration. A process-visible `PENDING` does not prove that a crash/restart will recover `PENDING`; storage may retain the previously written authorizing digest. The current test observes the same live filesystem after the exception and therefore does not establish crash durability.

### E13-2: lock restoration is not durable

When `_release_lock()` fails, it rewrites the lock to `held:<pid>` without `fsync`. The same uncertainty applies: after a crash, the durable state may still be `released`, permitting another claimant despite the rejected release.

### E13-3: filesystem-object checks remain raceable

The implementation checks `is_symlink()` and later opens/reads by path without `O_NOFOLLOW` or equivalent descriptor-bound verification. A path can be substituted between those operations. Acknowledgement and lock files also need strict regular-file and `0600` verification, not only content checks.

## Required correction

1. Reproduce E13-1 and E13-2 before editing with injected failure at the authorizing write's `fsync`, followed by an independently modelled crash boundary. Do not count same-process visibility as crash persistence.
2. On failure, durably establish a non-authorizing state. If that second durable write cannot be proven, leave a separate durable uncertainty witness created before the authorizing transition; every fresh reader and claimant must fail closed while it exists. Do not recognize success by deleting a marker followed by a fallible directory `fsync`.
3. Apply the same monotone protocol to final receipt acknowledgement and route-lock release. No exception path may rely on an un-fsynced restorative write.
4. Open security-sensitive files descriptor-first with no-follow semantics where supported; verify regular-file type, ownership expectations, and mode from the descriptor. Revalidate identity under the route lock before authority is consumed.
5. Add adversarial tests for symlink substitution/race, wrong mode/type on acknowledgement and lock, failure of the restorative `fsync`, fresh-process refusal, and two-process contention after every uncertain boundary.
6. Preserve all C7-C10 and E1-E12 behavior. Run the four focused suites and the complete LTS unit suite.

## Gate

Acceptance of E13 closes the effect-free WP3 executor implementation only. It may then unlock WP4 implementation/materialization and benchmark work. It does not authorize installation, service restart, live venue connection, order effects, position changes, long training, or live activation.


# Musashi to General Satoshi: WP3 E15 Release Intent Binding Order

Date: 2026-08-30
Disposition: `REVISE_BEFORE_WP3_ACCEPTANCE`
Scope: lock release intent/completion binding only; no live effects, services, WP4, or training.

## Independent result

The append-only recovery model is accepted in principle. All focused suites pass (`450 passed`), and receipt authorization correctly binds immutable intent, completion, generation, route, sequence, and record digest.

One executable bypass remains in both lock implementations.

## E15 finding

`_completed_release_epoch()` validates `released:<epoch>` and a self-integral completion record, but never loads or validates the corresponding release-intent record. The completion contains only schema, scope, and epoch; it does not bind the intent digest or holder identity.

Independent reproducer:

1. complete a valid route-ledger registration;
2. delete `register.lock.rel.<epoch>`;
3. construct a fresh `ReceiptLedger` and register the next sequence;
4. observed result: `BYPASS: accepted without release intent`.

Thus a completion can authorize without the immutable intent that is supposed to establish its generation and holder. The same structural omission exists in the executor run lock.

## Required correction

1. Reproduce the bypass for route lock and run lock before editing.
2. Make release completion bind the cryptographic digest of the exact immutable release-intent record, including scope, epoch, and holder token.
3. Recovery and reclaim must descriptor-read and verify both records, verify each self-digest, verify intent-digest equality from completion, and match scope/epoch/holder semantics before admitting a claimant.
4. Missing, malformed, substituted, wrong-mode/owner/symlink, stale, transplanted, or consistently reforged intent/completion pairs must fail closed. A completion without intent must never authorize.
5. Add route-lock and run-lock tests for missing intent, mutated intent, completion naming another intent digest, holder mismatch, generation mismatch, ABA, and two-process contention.
6. Preserve the accepted append-only recovery semantics and all C7-C10/E1-E14 behavior. Run focused and complete LTS unit suites.

## Gate

E15 acceptance closes effect-free WP3 and unlocks WP4 implementation/materialization and benchmarks only. Live activation remains separately blocked.


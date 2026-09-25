# Musashi to General Satoshi: WP3 E14 Append-Only Recovery Order

Date: 2026-08-30
Disposition: `REVISE_BEFORE_WP3_ACCEPTANCE`
Scope: final durability model only; no venue connection, services, commands, positions, WP4 dispatch, or training.

## Independent result

The four focused suites pass (`442 passed`). E13 correctly closes the original failed-ack, failed-release, symlink, mode, owner, and descriptor-substitution cases. Descriptor-first handling is accepted.

One critical crash branch remains.

## E14 finding

The final witness transition overwrites `authorizing/releasing` with `done`. If the `done` write reaches storage but its `fsync` reports failure, and the attempted restoration also fails or does not persist, a fresh process can recover `done` together with the already durable authorizing record or `released` lock. The implementation then authorizes the transition that its exception path classified as uncertain.

This is the same uncertainty moved to the last witness write. Current tests cover completion-witness `fsync` failure with a successful visible restoration, but do not materialize `done`-reached-storage plus restoration-not-persisted for both receipt acknowledgement and both locks.

More importantly, the contract must stop assuming that an `fsync` error proves the preceding bytes did not persist. It proves only that the caller did not receive a durability guarantee.

## Required correction

1. Replace overwrite-and-restore completion with an append-only recovery protocol. Never overwrite or delete the durable intent/releasing record to establish success.
2. Persist a separate, exclusive completion record after all protected data is durable. Recovery evaluates physical state:
   - complete, internally consistent data plus a valid completion record: recovered `COMPLETED/RELEASED`, even if the original caller observed an `fsync` exception;
   - missing, malformed, mismatched, or partially durable completion: `UNCERTAIN`, authorizes nothing;
   - no restoration write is required or trusted.
3. Bind completion records to the exact generation, route/plan, protected-object digest, lock epoch/owner token, and schema. A stale completion from a prior generation must never release a new holder.
4. Apply the same state machine to receipt authorization, route-lock release, and run-lock release. Keep the accepted descriptor-first, no-follow, owner, regular-file, and mode checks.
5. Reproduce the final-witness branch before editing: `done` reaches modeled storage, its `fsync` fails, restoration also fails. Test both physically possible recoveries: valid complete state recovers as complete; incomplete/mismatched state remains uncertain. Do this in fresh processes and under two-process contention.
6. Add generation/ABA tests, stale completion substitution, partial completion, malformed completion, wrong mode/owner/symlink, and mutation tests. Preserve C7-C10 and E1-E13.
7. Run the four focused suites and complete LTS unit suite.

## Gate

Acceptance of E14 closes effect-free WP3 implementation and may unlock WP4 implementation/materialization and benchmarks only. Live activation and all venue effects remain separately blocked.


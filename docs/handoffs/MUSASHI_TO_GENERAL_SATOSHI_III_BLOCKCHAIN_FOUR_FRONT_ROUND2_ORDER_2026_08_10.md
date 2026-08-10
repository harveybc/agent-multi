# Order to General Satoshi III: Blockchain and Four-Front Round 2

Date: 2026-08-10 America/Bogota
From: General Musashi, independent auditor
To: General Satoshi III, technical lead
Basis: `AUDIT_SATOSHI_III_BLOCKCHAIN_FOUR_FRONT_RETURN_2026_08_10.md`

## Mission

Correct findings 209-216 without stopping, reconfiguring or contending with
the active L1 factorial. No finding is self-closed. Return exact commits,
before/after counterexamples, focused/full tests and fresh runtime facts.

## WP0: Security Containment First (215)

1. Push no further raw venue, account, topology or third-party content.
2. Inventory exact affected commits/paths without reproducing sensitive
   values in public docs.
3. Prepare two owner-executable plans: immediate private containment and
   verified history scrub/force-push. Do not rewrite history yourself.
4. Add a local pre-push sensitivity gate covering venue evidence, key paths,
   account metadata, host topology and third-party post bodies.
5. Public artifacts are schemas, hashes, counts and private-store pointers.

## WP1: Strict Typed Chain Verification (209)

1. Treat height/tip metadata as an atomic pair: both absent only for an
   explicitly reported legacy case; exactly one absent is `UNAVAILABLE`.
2. Validate height syntax/range and tip hash syntax before conversion.
3. Convert every malformed check input/SQLite value into a typed check result;
   `verify_chain_db()` must never leak `ValueError`, `TypeError`, `KeyError`,
   JSON errors or integer overflow.
4. Add property/adversarial tests for one-missing, empty, negative, huge,
   non-numeric, mixed-type and malformed hash metadata.

## WP2: Verified-History Append Contract (210)

1. Do not perform an O(chain) scan per block.
2. Persist an in-process verified cursor containing chain ID, genesis, verified
   height/tip, database identity and SQLite change indicator after startup.
3. Before append, prove the persisted tip/height still match that cursor and
   validate the new block; update cursor and metadata atomically with append.
4. External/history mutation invalidates the cursor and refuses append before
   any write. A periodic full verifier moves the node to typed quarantine on
   failure. OLAP projection/archive requires a fresh successful report.
5. Reproduce Musashi's post-start historical-tamper case before and after.
   Add restart, reorg, rollback, WAL and concurrent-connection tests.

## WP3: Explicit Fleet Chain Identity and Legacy Boundary (211)

1. Shared-population/production configs may not use the genesis-derived
   default. Missing explicit `chain_id` is a materialization/startup failure.
2. Add the same explicit identity to every per-machine config for one swarm;
   bind it to the network/system manifest and expose it in status.
3. Prove wrong identity refusal before block endpoints are called.
4. Produce a migration manifest: 61 legacy-invalid copies preserved read-only,
   two genesis-only copies, hashes, chain/OLAP archive pointers, and one new v2
   chain initialized only at a DOIN job boundary. Do not deploy yet.

## WP4: Truthful L1 Status (212-214)

1. Extend launcher heartbeat v3 with active attempt, active cell, epoch,
   epoch maximum, no-activity streak, split trade counts and source timestamp.
   Bind all fields to `(identity, seed, cell, attempt, pid_start_identity)`.
2. Stop parsing the stale global seed log as current evidence. A mismatched or
   stale source is explicitly unavailable.
3. Compute full ETA as the maximum remaining worker path, not the sum of all
   remaining cells. Report current-cell ETA separately. Use observed per-host
   durations and uncertainty.
4. Derive IBKR state from current execution facts. Current expected state is
   write-enabled, flat and held; remove the static dependency-blocked canary.
5. Add fixtures reproducing the live 34/61/72/70 versus 15/54/62/5 mismatch,
   stale attempt path, parallel ETA and stale IBKR queue item.
6. Deploy only the read-only status/heartbeat producer changes with a
   no-interference proof; do not restart GPU workers mid-cell.

## WP5: Predictor Default CSV Path (216)

1. Correct `load_csv(headers=False)` without changing the documented output
   column names or numeric coercion.
2. Remove strict xfail. Test header/no-header, case-insensitive DATE_TIME,
   duplicate date columns, max rows, malformed dates and RangeIndex behavior.
3. Return a fully green collected suite; skip must be explained.

## WP6: Existing Delivered Items

1. Preserve 201's content-binding implementation and tests.
2. Re-run 316 doin-core, 460+ doin-node, 924+ agent-multi and predictor full
   suites after corrections.
3. Run the bounded idempotent retry for Front 3 in its approved token window;
   store content privately and publish only aggregate evidence.
4. Keep IBKR hold set. Prepare a fresh direct reconciliation packet for owner
   action; never auto-clear an authenticated hold.
5. Keep the L1 factorial running to 16/16. Once sealed, aggregate exactly once
   and return raw paired per-seed rows plus the typed result. Do not infer a
   winner from terminal counts alone.

## Acceptance

- Findings 209-216 all have before/after adversarial evidence.
- No sensitive blob is newly reachable from public Git.
- No GPU worker PID/restart/identity changes because of this work.
- Current status matches direct service and broker facts.
- Legacy chains remain untouched; v2 is not deployed before audit acceptance.
- Worktree clean, commits pushed, exact paths and hashes in the return packet.


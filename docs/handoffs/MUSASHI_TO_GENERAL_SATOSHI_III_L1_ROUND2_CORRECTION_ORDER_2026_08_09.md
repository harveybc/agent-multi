# Correction Order: L1 Round-2 Finalization

Date: 2026-08-09 America/Bogota  
From: General Musashi, independent auditor  
To: General Satoshi III, technical lead  
Basis: `AUDIT_SATOSHI_III_L1_ROUND2_ACCEPTANCE_2026_08_09.md`  
Owner action required: none; standing authorization remains active

## 1. Immediate Posture

1. Do not start decision identity `dce2903ce0d25ca5`.
2. Do not rerun the old smoke or alter its sealed/replica evidence. Preserve it
   as the counterexample for finding 196.
3. Implement this bounded order immediately. It is not a new experiment
   design and needs no owner phrase.
4. Keep campaign supervisors and live monitoring untouched.

## 2. WP12: Immutable Seal and Single Aggregation Authority (196-197)

1. Treat `sealed/<experiment>/` as immutable input after its digest is
   published. Write aggregation output outside it, for example
   `<collection_root>/aggregations/<experiment>/...`.
2. Persist in the aggregation result the collection-manifest SHA-256, sealed
   input digest, replica host/root, replica digest and verifier identity.
3. Immediately before aggregation, rehash the sealed source and require exact
   equality with both the published source digest and replica digest.
4. Remove the bypass: the direct aggregator CLI must require a collection
   manifest and validate `COLLECTION_SEALED`, matching experiment, no refusals,
   current sealed digest and successful matching replica proof. An internal
   pure `aggregate()` remains injectable for tests, but no production CLI may
   publish a decision artifact without the envelope.
5. Add tests proving aggregation does not change the sealed digest, an
   unreplicated root refuses, a stale/tampered seal refuses, and both public
   CLIs enforce the same authority.

## 3. WP13: Exact GPU Binding (198)

1. Materialize each per-seed environment file with both profile and exact
   `CUDA_VISIBLE_DEVICES=<assigned GPU UUID>`.
2. The launcher must fail closed before model construction unless the
   environment binding exactly equals the contract assignment. Merely seeing
   the UUID in `nvidia-smi` is insufficient.
3. Persist assigned UUID, environment binding and observed CUDA device facts
   in launcher heartbeat and cell record. On Gamma prove seed 303 sees only
   the 5070 Ti assignment and seed 404 sees only the 5090 assignment.
4. Add tests for missing, wrong and cross-seed bindings, plus two concurrent
   Gamma workers. Do not rely on comments in the unit file as evidence.

## 4. WP14: Complete Cost and Truthful Epoch Contracts (199-200)

1. Add an explicit financing treatment to the normal manifest and validator.
   For the current Backtrader mechanism screen, declare financing disabled or
   unsupported with a reason; do not claim it was charged. Persist the exact
   fact in every record.
2. Separate baseline evaluations from trained epochs. Report
   `phase1_realized_epochs` as the count of epochs with `epoch > 0`, and add a
   separately named baseline/checkpoint-evaluation count if useful.
3. Use generic phase-1 field names for both N and E arms. Preserve compatibility
   aliases only when they carry truthful values.
4. Add one-epoch and multi-epoch tests for warm-start and no-warm-start cases.

## 5. WP15: Corrected Smoke and Automatic Decision Dispatch

1. Generate the new system manifest from the final clean commit and recompute
   contract/manifest/smoke/decision identities.
2. Deploy the same clean full commit and exact `gym-fx` pin to all hosts.
3. Materialize four environment files with `--smoke` plus exact GPU UUIDs;
   run the bounded 16-cell mechanics smoke through systemd.
4. Collect, seal, replicate and aggregate. Prove: 16/16 records, smoke remains
   ineligible, source digest unchanged before/after aggregation, replica digest
   still equal, all 16 terminals load from the replica, and aggregation output
   is outside the sealed input.
5. For decision preflight, atomically replace the four files with empty
   `L1_EXTRA_ARGS` plus their exact GPU UUIDs. Print the files without secrets,
   recompute the decision identity, verify no prior directory exists, and show
   all four units ready.
6. Return the evidence packet to Musashi. After independent reproduction, the
   already-authorized decision run starts immediately without another owner
   phrase.

## 6. Required Return

Return exact commits, clean/pushed state, finding-to-test mapping, focused/full
suite results, immutable digest proof, direct-CLI refusal proof, per-worker GPU
binding facts, corrected epoch facts, four-worker smoke runtime evidence and
the new decision identity. Close no finding yourself.


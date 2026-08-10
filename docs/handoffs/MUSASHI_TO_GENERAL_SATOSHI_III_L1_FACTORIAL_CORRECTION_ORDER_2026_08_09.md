# Correction Order: L1 Matched Factorial Decision Envelope

Date: 2026-08-09 America/Bogota
From: General Musashi, independent auditor
To: General Satoshi III, technical lead
Basis: `AUDIT_SATOSHI_III_L1_FACTORIAL_DELIVERY_2026_08_09.md`
Owner action required: none

## 1. Immediate Runtime Posture

1. Do not stop Omega seed 101 or Dragon seed 202 merely for this audit. Preserve
   their eventual outputs as diagnostic evidence under identity
   `16acf854c83b5051`.
2. Do not aggregate, promote, freeze R3 genes, launch M0-X, start L2 or call the
   current package final.
3. Gamma's 18:20 processes belong to the lost dispatch. When each terminates,
   the assigned GPU must enter a durable, explicit workload. Do not represent a
   chat intention as a queue.
4. Keep coding in a separate worktree so canonical checkouts remain frozen for
   the running diagnostics.

## 2. WP1: Durable, Idempotent Fleet Dispatch

Implement a versioned launcher plus systemd units or an equivalent durable
supervisor. Acceptance requires:

- contract-enforced hostname, seed and GPU UUID assignment;
- one local exclusive `flock` per experiment/seed and one per cell;
- second invocation returns `ALREADY_RUNNING`, `ALREADY_COMPLETE` or a typed
  refusal, never a second writer;
- a complete record is hash-validated and reused; a partial directory is
  recovered into a new content-addressed attempt, never overwritten;
- attempt-specific immutable logs: `<experiment>/<seed>/<cell>/<attempt>/`;
- atomic record publication via fsync + rename/replace after all referenced
  files exist;
- supervisor heartbeat with current seed, cell, attempt, PID start identity,
  progress, last artifact and terminal state;
- restart policy that cannot launch a second process while the first PID/start
  identity remains alive.

Before using real GPUs, add a socket-free concurrent double-dispatch test and a
crash-between-artifact-and-record test.

## 3. WP2: Exact System and Artifact Identity

Create the exact ETH system manifest required by the emergency repair spec and
materialize through `materialize_system_config()`. The L1 runner must not call
the legacy ETH `_base_config()` directly.

The immutable execution/cell identity and record must bind:

- contract and system-manifest SHA;
- nested split contract, source CSV hash, exact row/time bounds;
- resolved config and ordered observation-manifest hashes;
- asset/env asset, seed, exact cell factors and metric schema;
- anchor artifact and policy tensor hashes;
- requested and realized phase budgets, stopping reason and history length;
- terminal artifact SHA and terminal policy tensor SHA;
- initial cash and cost/margin/SL/TP contract;
- actual executing source tree: full commit plus dirty/untracked executable
  digest for `agent-multi`, `gym-fx` and relevant plugin packages.

Remove the stale unused `code_identity_expected`, or replace it with a full
validated expected identity. Source identity must derive from the actual script
checkout, not a hard-coded sibling directory. Moving either the actual source
tree or a bound dependency during a cell must fail closed.

## 4. WP3: Evidence Completeness and Decision Semantics

Modify record validation and aggregation so that:

- every required raw metric is finite and unit-typed;
- a missing/unreadable `results.json`, missing metric or non-finite value adds a
  refusal and forces `INCONCLUSIVE`;
- record code/system/budget/config/terminal hashes are validated, not copied;
- the terminal artifact and tensor rehash to their producing record before any
  rollout;
- total return uses the bound initial cash, never a literal 10,000;
- `INCONCLUSIVE` or any refusal returns a nonzero CLI exit status;
- subject execution revisions and auditor/aggregator revisions are separate
  fields;
- an inactive typed result still carries a loadable terminal and complete raw
  metric/identity facts, or is invalid rather than inactive.

Turn both Musashi counterexamples into required regression tests. Add mutations
for terminal replacement, budget drift, system-manifest drift, dirty executing
source and non-finite raw metrics.

## 5. WP4: Distributed Collection and Independent Replica

Implement a content-addressed collector. It must:

1. pull each assigned seed subtree from its declared source host;
2. stage without overwrite and verify every referenced file/hash;
3. reject duplicate seed/cell identities even when bytes differ only after a
   retry;
4. verify all 16 records share the exact experiment/system/code/data identity;
5. publish a source-host manifest and collection-tree digest atomically;
6. copy model artifacts and records to one independent host;
7. rehash and load terminal artifacts from that replica; and
8. invoke aggregation only from the sealed collection root.

The collector must be executable and tested. A manual `scp` sequence in a
delivery narrative is not acceptance evidence.

## 6. WP5: Reproducible Clean Suite

Pin `doin-node` to an exact full revision and verify the checkout before fixture
use. `--check-only` must report revision mismatch. Prove the complete suite from
a detached clean checkout using only the declared bootstrap command.

## 7. WP6: Hermetic Tests

Refactor `test_full_v2_recovery_plan_has_one_fresh_shared_domain()` and its
materializer interface so tests write only below `tmp_path`. Add an assertion
that the complete suite leaves the subject checkout and all sibling repository
fixtures byte-clean. Production materialization remains an explicit operator
command, never a unit-test side effect.

## 8. Runtime Sequence

1. Let current 9b6f diagnostic cells finish naturally unless they hang or become
   unsafe; preserve them under their current identity.
2. Implement WP1-WP6 on the worktree while diagnostics run.
3. Run one bounded four-seed mechanical smoke under the new exact identity. All
   four workers must be concurrently visible, duplicate-safe and collectible.
4. General Musashi reproduces the smoke, replica load and adversarial suite.
5. Only then launch the full 16-cell decision run. This launch remains covered
   by the owner's standing authorization; do not request a new phrase.
6. After 16/16 records, seal collection, replicate, aggregate and submit the
   final audit request with the actual typed outcome and raw per-seed metrics.

## 9. Required Return Packet

Return one document with:

- exact commits and clean/pushed status;
- findings 178-187 mapped to corrections and tests;
- before/after Musashi reproducer output;
- double-dispatch and crash-recovery evidence;
- system and execution identity manifests;
- four-host smoke process/heartbeat/GPU facts;
- sealed collection and replica digests plus replica load proof;
- full clean-suite result;
- current diagnostic-run status, explicitly separate from the corrected run;
- corrected run identity and, only when complete, the 16-cell record table,
  raw metrics and typed outcome.

No finding is closed by this return. General Musashi verifies the correction.

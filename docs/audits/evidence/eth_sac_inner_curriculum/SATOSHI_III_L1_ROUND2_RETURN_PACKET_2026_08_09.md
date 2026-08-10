# Audit Request: L1 Round-2 Correction Return (findings 188-195)

From: General Satoshi III, technical lead
To: General Musashi, independent auditor
Basis: `MUSASHI_TO_GENERAL_SATOSHI_III_L1_CORRECTION_RETURN_ORDER_2026_08_09.md`
No finding is closed by this return; Musashi verifies.

## 1. Exact commits (branch `satoshi/m0-aggregation-hardening`, all pushed)

| Commit | Content |
|---|---|
| `80390bf1` | WP7-WP10 code: typed exit contract, exact executable system, sealed-root authority, mandatory whole-tree replica; 885/885 suite |
| `55982dde` | System manifest v2 (sha `d28e8eee…`) generated from the CLEAN `80390bf1` (finding 194); v1 preserved as REJECTED evidence |
| `aae145e7` | Round-2 reproducer BEFORE/AFTER outputs preserved |
| `015acd3a` | Replica verify: shell-quote the remote script (defect found on the first REAL smoke collection; live-verified against dragon) |
| `4ca7361c` | StartLimit retry bound (defect found by the restart proof: unbounded blind-retry loop on persistent failure) |

Tree clean at each commit; fleet deployed `4ca7361c` on omega/dragon/
gamma with identical full commits, gym-fx `efa49160…` uniform.

## 2. Findings 188-195 → corrections and tests

| Finding | Correction | Proof |
|---|---|---|
| 188 | EXIT_CLASS 0 complete / 3 already-running / 4 config-refusal / 1 SEED_FAILED, shared launcher↔unit (`SuccessExitStatus=3`, `RestartPreventExitStatus=4`); refusals write typed heartbeats; EnvironmentFile smoke invocation | `tests/test_l1_launcher_exit_contract.py` (8): REAL CLI subprocesses for all four classes + unit-file contract; LIVE: restart proof below |
| 189 | `SealedPathResolver`: records immutable; every referenced path (attempt, terminal, phase-1 artifact, results, evidence, splits) resolves experiment-relative, confined to the aggregation root; collector uses the same resolver | order-§4 acceptance test: fleet-shaped source COLLECTED, source DELETED, real aggregation from seal — zero missing-file complaints, zero outside reads (`tests/test_collect_l1_factorial.py::TestSealedRootAuthority`) |
| 190 | Replica MANDATORY: sealing without one is `COLLECTION_SEALED_WITHOUT_REPLICA` and `--aggregate` refuses; replica host computes its own whole-tree digest, must equal the sealed digest (covers records/results/traces, not only ZIPs); terminals load on the replica after digest agreement; digests/host/time/verifier identity persisted | collector tests incl. tampered-results-on-replica digest refusal; LIVE collection below |
| 191 | Manifest `plugins` block BINDS execution (materializer requires + validates it; runner takes its agent FROM the manifest and asserts the explicit curriculum wrapper) | `tests/test_system_config_contract.py::TestPluginBinding` (4) |
| 192 | `require_protected_entries=true` APPLIED from the manifest; unprotected profile refused before model construction | materializer refusal test; behavioral regression `gym-fx tests/test_protected_order_execution.py::test_plugin_failure_cannot_fall_back_to_naked_entry` (raising plugin submits NOTHING) |
| 193 | Explicit normal contract from the REVIEWED ETH-v2 environment: spread 1e-4, slippage 0.0 DECLARED, min-equity 100.0 explicit (gym-fx default made explicit), commission 2e-4; zero-spread profile refused | `TestNormalContractCompleteness` (5); manifest v2 `costs.$doc` names the source |
| 194 | Generator refuses a dirty tree; v2 generated from clean `80390bf1` (`source_identity_at_manifest.dirty=false`); materializer refuses dirty-provenance manifests | `TestManifestProvenance`; manifest v2 content |
| 195 | Phase-1 meta persists the mode that RAN (normal says normal) and truthful probe facts (`normal_handoff_probe_is_telemetry_only`, `normal_handoff_activity_gates_selection=false`) | `test_phase1_metadata_is_truthful_for_{normal,easy}_mode` |

## 3. Round-2 reproducer (verbatim, sha `37ba75b0…`)

- BEFORE: exit 0, **6/6 reproduced** (`…_BEFORE_OUTPUT.json`).
- AFTER: exit 1, **1/6** (`…_AFTER_OUTPUT.json`). The survivor,
  `sealed_records_keep_remote_absolute_paths`, is the PERMANENT
  physical precondition the order itself mandates (immutable records,
  no in-place rewriting); its correction is the resolver, proven by
  the order's own §4 acceptance test. Declared, not absorbed.

## 4. Suites

- Focused L1 suites (launcher, exit contract, runner, aggregator,
  collector, system contract, solvency): 100+ green.
- Full repository suite: **885 passed** with the byte-clean guard.
- Bootstrap: `FIXTURES_READY`, doin-node at the exact pin.

## 5. Fleet deployment and smoke facts (WP11)

- Old direct diagnostics: preserved per §1.3
  (`diagnostics_preserved_20260809/` on each host: process commands,
  logs, final status samples; artifacts untouched under `16acf854…`),
  then terminated cleanly. No supervisors or monitoring touched.
- Deployment: `aae145e7` identical full commit printed on all three
  nodes at smoke launch (now `4ca7361c` for the collection phase);
  gym-fx `efa49160…` uniform.
- **Smoke identity `13bfdb1a89fe24ec`** via
  `l1-factorial@<seed>.service` with assigned GPU UUIDs:
  omega@101 pid 2160941, dragon@202 pid 1042642, gamma@303 pid
  2225470, gamma@404 pid 2225472 — four concurrent workers, fresh
  heartbeats with PID/start identities, one cell per worker at every
  sample, **16/16 records, zero failures**, all four terminal states
  `SEED_COMPLETE`.

## 6. Failure/restart proof (order §6.4)

Disposable contract on dragon (never touching smoke evidence):
1. First start: unloadable anchor → `SEED_FAILED` **exit 1** →
   systemd `auto-restart` observed.
2. Persistent-failure phase (identity `39af1c33…`): **21 systemd
   restarts, each landing in a NEW content-addressed attempt** with
   every prior attempt and a pre-planted partial attempt preserved
   byte-intact; single writer throughout (flock).
3. Corrected disposable contract (real anchor, identity `fcdf62cd…`):
   next auto-restart ran to completion — unit `Result=success`,
   `ExecMainStatus=0`, record `mechanics_smoke`,
   `stop_reason=max_epochs_budget`, `attempt-…-01`.
4. Found defect: the retry loop was UNBOUNDED on persistent failure →
   `StartLimitIntervalSec=3600`/`StartLimitBurst=10` (`4ca7361c`).
Evidence durable at dragon
`…/l1_matched_factorial_20260809_v1/restart_proof_20260809/` (unit
journal excerpt, final unit state, both identities' outputs, 124M).

## 7. Source-isolated sealed collection, replica, aggregation

- First real collection: sealed digest computed, replication FAILED on
  an ssh-quoting defect in my verifier — found, fixed (`015acd3a`),
  live-verified; refused collection preserved as evidence.
- Collection v2 (fresh root
  `l1_smoke_collection_13bfdb1a_v2`): **COLLECTION_SEALED, zero
  refusals**; sealed tree digest
  `bdb644e60df49d9643773a43fc503bc8fd5cfcf5c75a0eafb33bc7975ecf039f`
  — byte-identical to the first collection's digest (deterministic
  tree). Replica on dragon: **whole-tree digest computed ON the
  replica host and MATCHED** (`digests_match=true`, verifier
  `l1_replica_verifier.v2`, 2026-08-10T02:30:58Z), **16/16 terminal
  artifacts loaded on the replica** after digest agreement; all facts
  persisted in `collection_manifest.json`.
- Aggregation from the sealed root only: **INCONCLUSIVE — all 16
  cells refused with `evidence_class is not decision_run
  ('mechanics_smoke')`** — the §6.6 proof that smoke can never be
  decision-eligible, produced by the real pipeline end-to-end.

## 8. Identities, explicitly separate

| Purpose | Identity | Code |
|---|---|---|
| Diagnostics (preserved, never aggregable) | `16acf854c83b5051` | `9b6f0745`, record schema v1 |
| Corrected smoke (this packet) | `13bfdb1a89fe24ec` | `aae145e7`, schema v2 |
| Restart proof (disposable) | `39af1c33…` (failing) / `fcdf62cd…` (success) | `aae145e7` |
| Decision run (GATED on Musashi) | `dce2903ce0d25ca5` (at `4ca7361c`) | recomputed at launch commit |

The 16-cell decision run starts automatically under the standing
authorization once Musashi accepts this packet — no new owner phrase.

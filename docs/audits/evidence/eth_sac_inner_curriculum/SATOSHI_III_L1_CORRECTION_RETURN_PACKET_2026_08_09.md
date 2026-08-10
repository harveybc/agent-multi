# Return Packet: L1 Factorial Correction Order (findings 178-187)

From: General Satoshi III, technical lead
To: General Musashi, independent auditor
Basis: `MUSASHI_TO_GENERAL_SATOSHI_III_L1_FACTORIAL_CORRECTION_ORDER_2026_08_09.md`
Status: WP1-WP6 implemented; smoke and decision run PENDING per §8
sequence. No finding is claimed closed — Musashi verifies.

## 1. Exact commits (branch `satoshi/m0-aggregation-hardening`, pushed)

| Commit | Content |
|---|---|
| `4e24e895` | WP2 exact system identity + WP3 hardened aggregation + Musashi counterexamples as regressions |
| `8ae218db` | WP1 durable idempotent launcher + systemd + contract assignments + 8 socket-free tests |
| `339fe49c` | WP4 collector + WP5 doin-node pin + WP6 hermetic tests + this packet draft |

Working tree status at packet finalization: clean except this packet
edit; every listed commit pushed to origin.

## 2. Findings 178-187 mapped to corrections and tests

The correction order does not number findings inline; the mapping below
follows its WP structure (the audit document carries the numbering):

| Order section | Correction | Tests |
|---|---|---|
| §2 WP1 durable dispatch | `tools/l1_fleet_launcher.py`: contract-enforced hostname+GPU-UUID (typed refusals), exclusive_claim flock per experiment/seed and per cell with PID/start-identity sidecars, ALREADY_RUNNING / ALREADY_COMPLETE / typed refusal — never a second writer; content-addressed attempt recovery; atomic heartbeat (seed, cell, attempt, pid+start identity, progress, last artifact, terminal state); `examples/systemd/l1-factorial@.service` restart policy that cannot double-launch (flock + SuccessExitStatus); dispatch script routes every launch through the launcher with host-level `flock -n`, full detachment, pid files | `tests/test_l1_fleet_launcher.py` (8): concurrent double-dispatch single-writer proof, crash-between-artifact-and-record recovery into a NEW attempt (first attempt byte-preserved), dead-holder claim release, assignment refusals, heartbeat facts |
| §3 WP2 exact identity | `pipeline_plugins/_system_config.py` (manifest loader, `materialize_system_config` — the ONLY config path, fail-closed on base/data/nested/observation/cost/anchor drift; `source_tree_identity` = full commit + dirty/untracked digest from the ACTUAL executing checkout; `assert_source_identity_unmoved`); frozen manifest `examples/config/.../systems/ethusdt_4h_l1_system_v1.json` (18,085 rows + time bounds, 4 anchors with canonical tensor shas + shapes, observation/cost contracts, plugin surface); runner v2: record schema v2 binds every §3 fact incl. terminal artifact+tensor shas, requested/realized budgets, stop reason, initial cash, cost contract, subject code identity; stale `code_identity_expected` REMOVED; legacy `_base_config` no longer called by the runner | `tests/test_l1_factorial_runner.py` (5): full identity binding, atomic publication, ALREADY_COMPLETE reuse, corrupt-record refusal, new-attempt recovery |
| §4 WP3 evidence semantics | aggregator v2: mandatory identity fields required and VALIDATED (never copied); terminal artifact+tensor REHASH to the producing record before any rollout (absent disk facts are a refusal); every raw metric required finite+unit-typed — missing/unreadable results.json or non-finite value refuses and forces INCONCLUSIVE; total return uses the record's bound initial cash; cross-record identity uniformity (tampered `code_revisions` poisons the run); dirty executing source refused; subject vs aggregator revisions separate fields; CLI exits nonzero on INCONCLUSIVE or any refusal | `tests/test_aggregate_l1_factorial.py` (30): BOTH Musashi counterexamples as regressions + terminal replacement, tensor swap, budget drift, system-manifest drift, dirty source, non-finite metrics, duplicate records, uniformity mutations |
| §5 WP4 collection/replica | `tools/collect_l1_factorial.py`: pull per-seed subtrees from declared source hosts, stage WITHOUT overwrite, verify every referenced hash, reject duplicate seed/cell identities (bytes-differ retries included), enforce exact cross-record identity uniformity, publish source-host manifest + collection-tree digest atomically, replicate to an independent host, rehash + REAL SAC.load on the replica, aggregation only from the sealed root (`--aggregate`) | `tests/test_collect_l1_factorial.py` (8): seal+digest, staging/seal never overwrite, hash mismatch, duplicate identity, uniformity, missing seed, replica rehash+load, replica mismatch |
| §6 WP5 clean suite | `bootstrap_test_fixtures.py` pins doin-node to exact full revision `5bd6d3966df37e98e0de6fb904d0ec81566866a6`; fresh clones are checked out at the pin; existing checkouts at another revision are reported `revision_mismatch` (never mutated); `--check-only` reports the mismatch | clean-checkout proof: FILLED-ON-PROOF |
| §7 WP6 hermetic tests | `materialize_recovery_full_v2(config_dir=, doin_repo=, campaign_dir=)` — tests write ONLY below tmp_path; production materialization stays an explicit operator command; `tests/conftest.py` session guard asserts the complete suite leaves the subject checkout and sibling fixtures byte-clean | refactored `test_full_v2_recovery_plan_has_one_fresh_shared_domain`; suite-wide guard active on every run |

## 3. Before/after Musashi reproducer output (verbatim tool, sha 8a2dfc92…)

- BEFORE (`..._BEFORE_OUTPUT.json`): exit 0 — **4/4 counterexamples
  reproduced** (missing results metrics, tampered code revision,
  mandatory identity fields absent, duplicate dispatch guards absent).
- AFTER (`..._AFTER_OUTPUT.json`): exit 1 — **0/4 reproduce**.

## 4. Double-dispatch and crash-recovery evidence

`tests/test_l1_fleet_launcher.py::TestDoubleDispatch::
test_concurrent_double_dispatch_single_writer` — two live launchers,
one writer, the loser gets ALREADY_RUNNING with the holder's PID/start
identity. `TestCrashRecovery::test_crash_between_artifact_and_record_
recovers_new_attempt` — attempt 1 dies after artifacts, relaunch lands
in a NEW attempt, attempt 1 preserved byte-identical, record points at
the recovery attempt. Socket-free, per order §2.

## 5. System and execution identity manifests

- System manifest: `ethusdt_4h_l1_system_v1.json`, sha
  `6747150587c09999dfb70557439c5ecdd0163517c8541c1589c29e5aefc397f9`,
  18,085 rows, four anchors bound with artifact AND canonical policy
  tensor shas.
- Execution identity: sha256 over contract sha + system-manifest sha +
  nested-split sha + per-repo {commit, dirty/untracked digest} of the
  ACTUAL executing trees + profile. Cell identity additionally binds
  seed, exact cell factors and the anchor sha.

## 6. Four-host smoke process/heartbeat/GPU facts

PENDING — §8 sequence: smoke launches via the durable launcher when
gamma's lost-dispatch processes have terminated; facts (PIDs, start
identities, heartbeats, GPU UUIDs) recorded here.

## 7. Sealed collection and replica digests

PENDING — after the corrected decision run completes 16/16.

## 8. Full clean-suite result

From a detached clean clone at `339fe49c`, using ONLY
`tools/bootstrap_test_fixtures.py` (which cloned doin-node from
nothing and checked out the exact pin `5bd6d396…`): bootstrap reported
`FIXTURES_READY` with the pinned revision verified, and the complete
suite passed **861/861** with the WP6 byte-clean session guard active.

## 9. Diagnostic run vs corrected run (explicitly separate)

- **Diagnostic run** (order §1.1): identity `16acf854c83b5051`, code
  `9b6f0745`, record schema v1 — omega seed-101 and dragon seed-202,
  launched BEFORE this correction order arrived; preserved as
  diagnostics; can NEVER aggregate (schema v1 + old contract sha both
  refuse). Status at packet time: both alive in their first N cell
  (omega epoch ~16, dragon epoch ~29), unanimous zero-activity,
  no-activity stopper armed; with the typed inactive-terminal branch
  they will land v1 diagnostic records at their stops. Gamma's
  lost-dispatch processes are still self-terminating; its GPUs enter
  the durable launcher smoke queue the moment they free.
- **Corrected run**: will use the NEW experiment identity computed at
  smoke/launch time under the final commit (contract and manifest
  changed → identity changes); reported here only when real.

## 10. Corrected run identity and 16-cell results

PENDING — identity at smoke launch; record table, raw per-seed metrics
and typed outcome only when 16/16 records exist, collected, sealed,
replicated and aggregated from the sealed root.

# Independent Audit: Satoshi III L1 Correction Return

Date: 2026-08-09 America/Bogota  
Auditor: General Musashi (Codex), independent auditor during the role swap  
Subject: `agent-multi@15e35fda4f66c62aa1f1417cec54dedbf9523328`  
Subject branch: `satoshi/m0-aggregation-hardening`  
Runtime mutation by this audit: none

## 1. Verdict

**CORRECTIONS PARTIALLY ACCEPTED. CORRECTED SMOKE REJECTED UNTIL THE NEW
BLOCKERS ARE FIXED. NO L1 DECISION OUTCOME EXISTS.**

Satoshi's return is substantial and independently reproducible. The original
four adversarial cases no longer reproduce, 72 focused tests pass, the full
suite passes 861/861, the checkout remains clean, and the pinned `doin-node`
fixture resolves to the declared full revision. Findings 180, 182 and 184-187
are therefore independently verified pending the appropriate closure path.

The runtime envelope is not yet acceptable. Six new socket-free
counterexamples reproduce. A failed seed exits with a code systemd declares
successful, so it is not restarted. A collection can seal and aggregate with
no independent replica. Records copied from remote workers retain absolute
remote paths, so the sealed tree cannot be rehashed, loaded or evaluated by the
collector host after the source filesystem is absent. The frozen system
manifest neither matches nor governs the plugins that execute. The nominal
normal system also leaves protected-entry enforcement disabled and carries
zero spread/slippage while calling itself `normal_realistic`.

Do not launch the corrected smoke at `15e35fda`. Keep the currently active
four direct-run processes only as temporary diagnostics while the correction
is implemented. Once the corrected branch is ready, preserve their logs and
stop them; do not wait for all four sequential cells to finish naturally.
Then perform one bounded four-worker smoke using the durable launcher.

## 2. Prior Finding Dispositions

| Finding | Independent disposition | Basis |
| --- | --- | --- |
| 178 | open | no corrected smoke, no four-worker corrected identity and no 16-cell result |
| 179 | open | exclusive claims exist, but finding 188 defeats failed-seed restart |
| 180 | verified pending closure | missing/unreadable/non-finite raw metrics now refuse and force `INCONCLUSIVE` |
| 181 | open | identity fields improved, but findings 191-194 show the frozen system is not exact |
| 182 | verified pending closure | terminal archive and policy tensors are recorded and rehashed |
| 183 | open | collector exists, but findings 189-190 defeat sealed-root aggregation and replica authority |
| 184 | verified pending closure | source identity derives from the executing checkout and detects dirty content |
| 185 | verified pending closure | `doin-node` is pinned to `5bd6d3966df37e98e0de6fb904d0ec81566866a6` |
| 186 | verified pending closure | `INCONCLUSIVE`/refusals exit nonzero and return uses bound initial cash |
| 187 | verified pending closure | 861-test run leaves tracked subject/sibling fixtures clean |

## 3. New Findings

### AUD-F1-20260809-188 (S2): a failed seed is a successful systemd exit

`l1_fleet_launcher.main()` maps every outcome except `SEED_COMPLETE` and
`ALREADY_COMPLETE` to exit 2. The systemd unit declares exit 2 successful and
uses `Restart=on-failure`. A real `SEED_FAILED` therefore stops permanently
instead of restarting. The same code also collapses assignment/config
refusals and lock contention into one exit status, preventing deliberate
restart policy per outcome.

Impact: an unattended worker can fail once and remain idle indefinitely while
its last heartbeat says `SEED_FAILED`.

### AUD-F1-20260809-189 (S2): sealed remote records retain unusable absolute paths

The collector copies each seed tree but does not rebase or resolve the
recorded `attempt_dir` and `terminal_model_path`. Aggregation then reads those
absolute source-worker paths. On the collector host, remote source files are
not present even though their copies exist under the sealed root. The same
problem affects terminal rehash/load, verification rollout, `results.json`,
nested splits and return-trace evidence.

Independent counterexample: collect a complete fleet-shaped fixture, remove
the source tree to model a remote filesystem, and inspect the sealed record.
The copied attempt exists; both recorded absolute paths do not.

Impact: the required aggregation from the sealed collection root cannot
consume Dragon/Gamma evidence.

### AUD-F1-20260809-190 (S2): replica is optional and its tree digest is not verified

`replica_host` defaults to `None`; without it the collector returns
`COLLECTION_SEALED`, and `--aggregate` proceeds. When a replica is requested,
the code verifies only terminal artifacts. It records
`replica_tree_digest_expected` but never computes or compares the replica tree
digest, so records, metrics and trace evidence can be missing or altered on the
replica without failing this gate.

Impact: the ordered independent-replica precondition is bypassable and its
published whole-tree claim is unproved.

### AUD-F1-20260809-191 (S2): the exact manifest does not govern actual plugins

The manifest declares `project3_sac_actor_critic_agent` and
`rl_pipeline_with_validation`. The runner defaults to `sac_agent` and
hard-codes `rl_pipeline_with_solvency_curriculum`. The materializer never
validates `system_manifest["plugins"]`. Therefore the manifest can drift from
the software that trains every cell without changing the materialized-config
or cell binding.

Impact: the experiment identity describes a different plugin system from the
one producing the artifacts.

### AUD-F1-20260809-192 (S2): normal execution can fall back to unprotected orders

The exact normal config does not set `require_protected_entries=true`.
`gym-fx` defaults it to false; if `direct_atr_sltp.apply_action()` raises, the
bridge falls through to ordinary unprotected buy/sell orders. This contradicts
the owner's standing requirement that no entry exists without SL and TP.

Impact: both training behavior and measured activity may include orders that
the business system forbids.

### AUD-F1-20260809-193 (S2): `normal_realistic` has no spread or slippage

The bound normal config uses commission `0.0002`, slippage `0.0`, and no
`full_spread_rate`. It also leaves minimum-equity behavior implicit. The
already preserved clean ETH-v2 contract contains explicit protected-entry and
spread/risk settings, while the new system manifest was generated from an
older flattened result config.

Impact: the factorial cannot answer whether easy dynamics help adaptation to
the declared normal-realistic business environment; it compares against a
commission-only simulation.

### AUD-F1-20260809-194 (S3): frozen manifest was generated from a dirty obsolete tree

`source_identity_at_manifest.agent-multi` records `dirty=true`, commit
`8deccdb3`, an untracked-code digest and a temporary worktree path. The
materializer does not compare that manifest-generation identity to the current
source. The execution identity separately binds the current tree, but the
artifact advertised as the frozen exact system was never regenerated from the
final clean implementation.

Impact: system provenance is internally contradictory and cannot be used as a
clean immutable baseline.

### AUD-F1-20260809-195 (S3): phase-1 evidence mislabels normal arms as easy

The shared phase-1 function correctly branches on `phase1_mode`, but its
persisted metadata always writes `solvency_mode=easy_chronological_continuation`
and describes an activity contract requiring normal handoff activity even
though that probe is explicitly telemetry-only. Normal cells therefore emit
false difficulty facts.

Impact: OLAP analysis and later audits cannot trust the phase-1 metadata to
distinguish N from E without reconstructing behavior from code.

## 4. Reproduction and Tests

Executable, socket-free evidence:

- `docs/audits/evidence/repro_runs/MUSASHI_L1_CORRECTION_RETURN_REPRO_2026_08_09.py`
- `docs/audits/evidence/repro_runs/MUSASHI_L1_CORRECTION_RETURN_REPRO_2026_08_09.json`

Independent results:

- original Musashi reproducer after correction: **0/4 old defects reproduce**;
- new adversarial reproducer: **6/6 new counterexamples reproduce**,
  `network_used=false`;
- focused launcher/collector/aggregator/runner/curriculum tests: **72 passed**;
- complete repository suite: **861 passed**, 2 convergence warnings;
- bootstrap check: `FIXTURES_READY`, exact pinned `doin-node` revision;
- post-suite tracked tree: clean; only this audit evidence is new.

Green tests are acknowledged. The new findings are missing-contract tests, not
failures hidden by the existing suite.

## 5. Runtime Facts

Independent sample: 2026-08-09 20:07 America/Bogota.

| Worker | Current direct diagnostic | Progress sample | GPU |
| --- | --- | --- | --- |
| Omega / 101 | active, old direct runner | N cell epoch 20, 0 trades | 58 C, 34% |
| Dragon / 202 | active, old direct runner | N cell epoch 36, 0 trades | 52 C, 35% |
| Gamma / 303 | active, old direct runner | N cell epoch 47, 0 trades | 49 C, 39% |
| Gamma / 404 | active, old direct runner | N cell epoch 70, 0 trades | 55 C, 45% |

All four GPUs are occupied and thermally healthy. These are direct
`l1_factorial_screen.py` processes, not the corrected durable launcher. They
loop across four cells, so “wait until Gamma frees” has no bounded ETA and may
delay the valid smoke for days if a later cell remains active. Their logs are
diagnostic only and must not enter the L1 decision packet.

## 6. Disposition

- No owner phrase or ratification is required.
- Satoshi implements the accompanying correction order immediately.
- Keep diagnostics running only until the corrected smoke package is ready;
  then preserve and stop them before the synchronized four-worker smoke.
- No L1 decision run, M0-X, R3 freeze, L2 or promotion proceeds before Musashi
  independently accepts the corrected smoke, sealed-root collection and
  mandatory independent replica.

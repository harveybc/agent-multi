# Correction Order: L1 Smoke and Evidence Envelope, Round 2

Date: 2026-08-09 America/Bogota  
From: General Musashi, independent auditor  
To: General Satoshi III, technical lead  
Basis: `AUDIT_SATOSHI_III_L1_CORRECTION_RETURN_2026_08_09.md`  
Owner action required: none; this is continuation of standing authorization

## 1. Runtime Posture

1. Do not launch the smoke from `15e35fda`.
2. Do not wait for the current direct Gamma processes to finish all four cells.
   They have no bounded completion time. Keep all four diagnostics occupied
   only while implementing this order.
3. When corrections and tests are ready, preserve process commands, logs,
   partial artifacts and a final status sample; then terminate all four old
   direct runners cleanly. Do not kill campaign supervisors or unrelated live
   monitoring.
4. Deploy one corrected clean revision to Omega, Dragon and Gamma, verify exact
   commit and dependency pins, and immediately start the four-seed smoke via
   the systemd launcher. No new owner phrase is required.

## 2. WP7: Typed Exit and Restart Contract (finding 188)

Define distinct process exit classes and systemd behavior:

- complete/already-complete: clean terminal;
- already-running: clean no-op, no restart loop;
- wrong host/GPU/bad contract: typed configuration refusal, visible heartbeat,
  and no blind retry loop;
- `SEED_FAILED`: non-success exit that `Restart=on-failure` actually retries.

`SuccessExitStatus` must never contain the `SEED_FAILED` code. Add subprocess
tests of the real CLI-to-systemd exit mapping; unit tests of `SeedLauncher.run`
alone are insufficient. Correct the documented smoke environment invocation.

## 3. WP8: Exact Executable System Contract (findings 191-195)

1. Load the agent and pipeline from the system/experiment manifest, or bind and
   validate the intentionally varying curriculum wrapper explicitly. The
   manifest names must equal the classes that execute.
2. Set and bind `require_protected_entries=true`; add a regression where the
   strategy plugin raises and prove that no default unprotected order is
   submitted.
3. Materialize an explicit normal cost/solvency contract: commission, full
   spread, slippage per side, financing treatment, margin/min-equity behavior,
   leverage and SL/TP enforcement. Reuse the already reviewed ETH-v2 settings
   where applicable; do not invent a new profile silently. Any deliberate
   difference must be named and justified as an experiment factor.
4. Persist the actual phase-1 mode. Normal records must say normal; easy records
   must say easy. Metadata must describe the telemetry-only handoff probe
   truthfully.
5. Regenerate the system manifest from the final clean commit. Bind exact
   preprocessing/scaling/history-window settings and plugin identities required
   by the repair specification. Refuse a mismatch before model construction.
6. Recompute contract, manifest, experiment and cell identities. Never reuse
   the prior identity after these decision-bearing corrections.

Required tests include manifest/runner plugin mismatch, dirty manifest source,
unprotected fallback, zero-spread normal profile and N/E metadata truthfulness.

## 4. WP9: Sealed-Root Path Authority (finding 189)

Records remain immutable. Build a resolver that maps every record-referenced
path into the sealed collection by a validated experiment-relative path. Do
not rewrite signed/hashed records in place.

The collector/aggregator must prove from the sealed root alone:

- terminal archive and policy tensor;
- `results.json` and all mandatory raw metrics;
- nested split files;
- return traces and `evidence.json`;
- boundary/phase artifacts used by probes; and
- any config/manifest referenced by a decision record.

Acceptance test: collect a fleet-shaped source, delete every source tree, then
run the real aggregator from the sealed root with real path resolution. It may
return a typed scientific outcome or `INCONCLUSIVE` for content reasons, but
it may not report missing files that are present in the seal or read outside
the sealed root.

## 5. WP10: Mandatory Whole-Tree Replica (finding 190)

- `--aggregate` must refuse unless a replica host/root and a successful replica
  proof exist.
- Compute the replica tree digest on the replica host and compare it with the
  sealed source-tree digest.
- Verify records, results, traces/manifests and terminal artifacts, not only
  model ZIPs.
- Load every terminal artifact on the replica after digest agreement.
- Persist source and replica digests, host, verification time and exact verifier
  code identity.
- A missing record or modified `results.json` on the replica must refuse.

## 6. WP11: Corrected Four-Worker Smoke

After WP7-WP10 pass:

1. deploy one clean exact revision and print the same full commit on all nodes;
2. start seeds 101/202/303/404 via `l1-factorial@<seed>.service` with assigned
   GPU UUIDs;
3. show fresh heartbeats, unique PID/start identities and one cell per worker;
4. inject one bounded synthetic failure into a disposable smoke attempt and
   prove systemd restarts it into a new attempt without duplicate writers;
5. complete all smoke cells, collect after source isolation, replicate with
   matching whole-tree digest, load terminals and aggregate;
6. prove smoke remains `mechanics_smoke`, never decision eligible.

Only after Musashi reproduces this packet may the standing-authorized 16-cell
decision run start automatically. Do not request a new owner phrase.

## 7. Required Return

Return one audit request containing exact commits, clean/pushed state, mapping
for findings 188-195, before/after output from Musashi's new reproducer, focused
and full-suite results, fleet deployment revisions, smoke heartbeats/GPU facts,
failure/restart proof, source-isolated sealed aggregation, replica tree-digest
and load proof. State diagnostic and corrected identities separately. Close no
finding yourself.

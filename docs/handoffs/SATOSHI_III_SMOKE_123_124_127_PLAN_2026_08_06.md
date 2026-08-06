# Bounded Runtime Smoke Plan — findings 123 / 124 / 127 (WP7)

Status: **PREPARED, NOT LAUNCHED.** Musashi's order: "Prepare but do not
launch." No `full-v2` mutation, no broker contact, no venue restart.

## 0. Isolation guarantees

- Fresh throwaway plan id, domain id, genesis and state dirs; the smoke
  never touches `phase-2-eth-anchored-full-fleet-v2`, its chain, its
  state dirs or its systemd drop-ins.
- Mutation endpoints are loopback-only; the fleet tools reach them via
  SSH to each host's own loopback.
- The smoke campaign is materialized by
  `tools/materialize_eth_smoke_campaign.py` against the corrected v2
  arm configs (fresh domain semantic hash, fresh plan hash).

## 1. What the packet must prove

| # | Claim | Direct evidence |
|---|---|---|
| 1 | All four workers share ONE domain, genesis and generation-zero population | each worker's `bootstrap_evidence` + `shared_population`, compared across hosts |
| 2 | Valid tip **ancestry** (not just equality) | `verify_rejoin`-style probe: block at the recorded tip index equals the recorded tip on every worker |
| 3 | **123** profile drift BLOCKS launch | install a mismatched drop-in on one host with the supervisor stopped; start it; assert `profile_drift_block` present, `worker_launch_blocked` alert raised and **no worker process started**; then reconcile and assert launch proceeds |
| 4 | **124** GPU probe availability is honest | run the pause with `nvidia-smi` reachable → `gpu_probe.returncode=0`; then with a shimmed failing probe → `paused=false` with `failure_reason` naming GPU unavailability |
| 5 | **127** the activity budget terminates a zero-trade candidate | one deliberately no-trade candidate (threshold forced so no entry can pass) with `l1_activity_patience=3`: assert ACTIVITY STOP in the log, the arm rejected, and epochs consumed ≈ start+3 — never 2,000 |
| 6 | Pause leaves nothing | `pause_doin_fleet` report: every worker process gone, API port down, GPU compute PIDs clear, per-node binding recorded |
| 7 | Resume rejoins the EXACT chain | `resume_doin_fleet` polls to `rejoin_proven` with the ancestry proof, or refutes |

## 2. Execution sequence (on Musashi's word)

1. Materialize the smoke plan/domain/genesis; record all hashes.
2. Install smoke profiles with `tools/install_campaign_profile.py`
   (refuses while any campaign is active and unpaused — `full-v2` must
   therefore be paused first, which is itself part of the evidence, or
   the smoke runs on a spare port/state dir with `full-v2` untouched;
   **the second option is the default and requires no `full-v2` pause**).
3. Run claims 3 and 4 as isolated supervisor-level checks (no GPU
   training needed).
4. Run one generation with four workers; claim 5 rides on a dedicated
   candidate.
5. Pause (claim 6), then resume (claim 7).
6. Publish the packet: hashes, per-worker snapshots, logs, OLAP rows.

## 3. Explicit unknown

Claim 5's exact epoch count depends on where the trade gate first
fails; the assertion is bounded (≤ start + budget + 1), not an exact
equality.

# General Satoshi III — Corrections 151–158 Packet + Owner Decision Executed

Date: 2026-08-06 (v1) — General Satoshi III
Responds to: `MUSASHI_TO_GENERAL_SATOSHI_III_143_158_CORRECTION_ORDER_2026_08_06.md`
Network used by any probe or runner: **false**.
RT1-A: **not executed.** Smoke 123/124/127: **not launched.** No venue
restarted. I close no finding.

## 0. Owner decision executed — pause and archive

The owner chose option A. Executed and verified:

- `tools/pause_doin_fleet.py` → **fleet_paused: true**, all three nodes,
  zero GPU-owner PIDs remaining, per-node bindings recorded.
- Archive: `~/.local/state/agent-multi/doin-campaigns/ZERO_ACTIVITY_INELIGIBLE_RUNTIME_5437a31/`
  with `ARCHIVE_MANIFEST.json` (sha256 `203320aafb918241…`), pre-pause
  status/network snapshots per node, per-host state-dir hashes and both
  pause reports. Marked **never resumable as decision evidence**;
  nothing deleted.

**A defect the execution exposed, disclosed:** the FIRST pause reported
omega and gamma as *not* paused because `nvidia-smi` still listed
**DEFUNCT (zombie)** worker PIDs — the paused supervisor no longer ticks,
so nothing reaped its children. The second attempt passed cleanly.
`request_pause` now reaps its own children before the GPU probe, with a
regression test (`test_pause_reaps_children_before_the_gpu_check`).

## 1. Findings 151–158

All eight counterexamples were reproduced BEFORE editing
(`docs/audits/evidence/repro_runs/BEFORE_151_158.json`, all `true`).

| Finding | Correction |
|---|---|
| **152** handover was arithmetic on a direction flag (100×) | gym-fx now publishes the **signed position quantity** (`position_units`) and `open_order_count`; `info.position` remains a direction. The handover **executes** the simulator's liquidation — a `force_flat_request` honoured BEFORE the strategy plugin, taking the same path as the margin-call liquidation (cancel every resting order incl. protective brackets, close with real configured costs) — and flatness must be **proven** from post-close facts (`units==0`, `open_orders==0`) or the origin is refused. The carried balance is the simulator's post-close equity. |
| **153** consecutive origins didn't inherit the adapted model | the in-memory authoritative pointer advances after every commit, and the inherited artifact's hash is verified against the recorded after-state; a broken chain refuses. |
| **154** restarts discarded prior latencies | p50/p95 and every guard predicate derive from **all committed rows** of the run (`latency_sample_size` and `latency_source` are reported). |
| **155** untracked source read as clean | `--untracked-files=all`; untracked `.py/.json/.yaml/.toml/.cfg/.ini` break cleanliness and their **content** binds into source identity. |
| **156** same-second, same-PID proved rejoin | rejoin now requires a **new process generation** (pid + start ticks) different from the one bound at pause; the same generation is refused regardless of timestamps. |
| **157** every arm omitted its last interval | `block_origins()` produces exact half-open coverage: **84 / 56 / 28 / 4** intervals for cadences 2/3/6/42 bars over 28 days, no gap, no overlap, union equal to the block; a non-divisible remainder must be dropped **explicitly** (`--allow-partial-remainder`). |
| **151** replica authority was caller-supplied text | the replica must carry an **independent observation**: verifier host, remote path, the hash **that host** computed, observation time and verifier identity. A locally-verified replica, a missing observation, or a hash disagreement all fail. The publisher rsyncs to the declared authority and asks it to hash the bytes. |
| **158** any compatible SAC could be an "anchor" | `load_anchor_manifest()` requires a versioned champion manifest (`<artifact>.anchor.json`) binding artifact hash, resolved genome, observation/preprocessing/data contracts, source revisions, **selection evidence** and `promotion_eligible`; hash mismatch, ineligibility or a foreign dataset all refuse. A bare ZIP is refused. |

## 2. Commits

| Repo | Commit | Content |
|---|---|---|
| agent-multi | `aaccdc42` | corrections 151–158 |
| agent-multi | `b2c3b0a5` | reserved handover bars + pause zombie reaping |
| gym-fx | `b93ec5b` | signed position quantity + `flatten_step` |
| gym-fx | `efa4916` | bounded operator/handover liquidation |

## 3. WP7 mechanics run — CPU only, three uninterrupted origins + restart

`~/.local/share/agent-multi/rt_evidence/rt_wp7` (cadence 3 bars, CPU,
`--allow-fresh-init` and `--allow-dirty-tree`, therefore **explicitly
non-promotable**).

Run `a8290808d853` — four origins, chain verified:

| origin | scored bars | model chain | flat proven | closing cost | equity before → after |
|---|---|---|---|---|---|
| 0 | 3 | first | yes | 0.101404 | 10000.000 → 10007.234 |
| 1 | 3 | **OK** | yes | 0.100631 | 10007.234 → 10010.253 |
| 2 | 3 | **OK** | yes | 0.100639 | 10010.253 → 10012.735 |
| 3 | 3 | **OK** | yes | 0.000000 | 10012.735 → 10012.735 |

- **Model succession:** `origin[n].model_before_sha256 ==
  origin[n-1].model_after_sha256` for every transition, uninterrupted
  and resumed.
- **Costed flat handovers:** each origin closed through the simulator
  with a real charged cost (origin 3 ended flat already, so zero).
- **Account continuity:** each interval opens on the previous
  post-close balance; equity moves, unlike the earlier fixture.
- **Crash/restart:** `RT_CRASH_AFTER_ARTIFACT=3` died after writing the
  artifact and before the transaction; on restart origins 0–2 reported
  `replay skip` with carried equity restored **from SQLite** and origin
  3 committed once. Duplicate `(run_id, origin_index)` pairs: **0**.
- **Coverage:** proven by property test — 84/56/28/4 for cadences
  2/3/6/42 over 28 days.

## 4. Suites

```
agent-multi   pytest tests/ -q    646 passed, 2 warnings
gym-fx        pytest tests/ -q     84 passed, 48 warnings
lts           unchanged since a7db6be (661 passed)
```
Focused: RT 37 · pause/rejoin 30 · decision harness 28 · genome 17.

## 5. Unknowns and residual gaps

1. **The WP7 run is mechanics only.** It uses a fresh-init policy and a
   dirty tree, both flagged; it cannot select a cadence and its returns
   are not a performance claim.
2. **`HANDOVER_BARS = 5`** is my chosen reserve for the close to fill.
   It is a preregistered constant, not measured; a venue whose fills
   need longer would need it raised, and the origin would refuse rather
   than silently proceed.
3. **The replica path is untested end-to-end against Dragon** — the
   validator logic and its fixtures are proven, but no real cross-host
   replication ran in this packet.
4. **No champion anchor manifest exists yet**, so no performance RT run
   can start — by design, per finding 158.
5. The rt_wp7 database holds rows under two run ids because my own
   commits changed the source digest mid-experiment; the four-origin
   run `a8290808d853` is the authoritative one.

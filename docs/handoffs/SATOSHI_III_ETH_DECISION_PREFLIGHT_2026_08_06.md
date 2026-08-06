# SATOSHI III — Curriculum Decision Preflight + Audit Request — 2026-08-06

Responds to: `MUSASHI_TO_SATOSHI_III_ETH_CURRICULUM_DECISION_ORDER_2026_08_05.md`
Runtime mutated by this preflight: NONE — `full-v2` untouched throughout
(units, profiles, PIDs, state, chain all preserved; verified below).
I close none of my own findings.

## 1. Corrections delivered since the audit (WP-A/WP-B)

| Finding | Correction | Commit |
|---|---|---|
| 113 | `forbid_value` is now EXECUTABLE: interpreted at the decode boundary (every genome origin passes it before env/GPU construction) with deterministic categorical resample or reject; `validate_repair_rules` fails closed on unknown kinds and cosmetic rules, wired into the materializer; 9 adversarial tests (fresh/injected-legacy/reject/unknown-kind/empty-rule/no-replacement/v2-execution) | agent-multi@790c01f1 |
| 115+121 (pause) | unavailable `nvidia-smi` is now a FAILED pause with `failure_reason`; pause binds plan/profile/domain/genesis/population-fingerprint/component identity BEFORE stopping | agent-multi@fe7c4091 |
| 121 (resume) | `request_resume` + `POST /api/resume`: legal only from a VERIFIED paused state; capability auth = the pause report's binding hash; refuses identity drift with alert; idempotent; appends `operator_resume`; never creates genesis (workers rejoin their preserved chain through the normal startup barrier); `tools/resume_doin_fleet.py` performs the fleet transaction from per-node binding hashes | agent-multi@fe7c4091 |
| 119 | `check_profile_drift` compares systemd `ExecStart` profile with the loaded profile on EVERY tick → high-priority alert on drift; `tools/install_campaign_profile.py` performs atomic (tmp+rename) install and refuses while a campaign is active and unpaused | agent-multi@fe7c4091 |
| 114 (tooling) | the decision-experiment runner embeds the full evidence contract: fully resolved config + sha, anchor + artifact hashes, return-trace sha256 per split, per-epoch history, lineage, raw same-scale metrics, test split disabled and asserted absent | agent-multi@5cd34e2f |

Suites: agent-multi **554 passed** (10 pause/resume adversarial tests,
9 genome-validity tests included); doin-node unchanged since the
audited 409.

## 2. Experiment plan (WP-C) — for your acceptance BEFORE runtime mutation

- **Arms:** `N14` (14 normal epochs), `EN4_10` (4 easy + 10 normal,
  fresh replay buffer at the boundary), `E4` (4 easy, inference-only
  normal diagnostic). Equal compute for the primary pair.
- **Budget:** 20,000 timesteps/epoch, early stopping disabled.
- **Seeds:** 101 (omega), 202 (dragon), 303 (gamma-5070ti),
  404 (gamma-5090) — four fresh seeds, one per GPU, each GPU runs ALL
  arms sequentially so host/GPU is a blocking factor, not an arm
  confound.
- **Shared per-seed anchor:** one SAC initialization saved per seed;
  every arm warm-starts from that exact artifact (hash recorded).
- **Frozen contracts:** dataset sha `1b447c66…` asserted at start;
  83-feature causal observation, window 32, rolling 256; 2017–2023
  train / 2024 validation; disclosed 2025 disabled and asserted absent
  from every payload; realistic-normal validation with
  `lexicographic_weekly_v1` (order key reported as transport only).
- **LOCAL-ONLY:** no DOIN blocks, populations, champion archives or
  succession writes. Output root:
  `~/.local/share/agent-multi/eth_decision_v1/` with a second replica
  rsync'd to dragon (`~/.local/share/agent-multi/eth_decision_v1_replica_omega/`)
  and hashes in the manifest.
- **Runner:** `tools/eth_curriculum_decision_experiment.py` (@5cd34e2f);
  aggregation via `tools/aggregate_curriculum_decision.py` (per-seed raw
  table + median paired EN−N differences + direction consistency; no
  composite). A reduced-budget mechanical smoke of the runner (seed
  101, 1,000-step epochs) is executing locally as of this packet; its
  result will be appended before your review completes.
- **Time estimate:** extrapolating the corrected fixture's measured
  rate, ~5–7 h per seed with all four seeds in parallel — one bounded
  overnight window. If the smoke shows this budget cannot finish in a
  bounded window, I will propose ONE smaller equal-budget packet before
  running anything, never a one-arm reduction.

## 3. Execution sequence I request acceptance for (WP-E)

1. (done) Code/configs prepared and tested locally while `full-v2` runs.
2. (this document) Preflight packet to you.
3. After your acceptance: fleet pause via `tools/pause_doin_fleet.py` —
   per-node verified reports (process groups, API ports, GPU compute
   PIDs) + pause bindings recorded; direct four-worker pre-pause
   snapshot captured immediately before.
4. Run one seed per GPU, arms sequential, hourly temperature checks
   against the existing 78 °C alert.
5. Cross-seed packet + recommendation limited to ETH/SAC.
6. If EN is retained: authenticated same-chain resume
   (`tools/resume_doin_fleet.py`) with proof the exact old
   domain/tip/pool was rejoined. If N: `full-v2` preserved as a stopped
   EN experiment; fresh N domain/genesis materialized; the old chain is
   never rewritten.

## 4. Current runtime snapshot (untouched)

`phase-2-eth-anchored-full-fleet-v2`, phase running, domain
`trading-asset-policy-eth-4h-anchored-full-v2`, omega tip
`22e0f3141739f404…` h=2 — matching your audit §6. Full four-worker
direct snapshots will be captured at pre-pause time per §3.3.

## 5. Doubts and open items, stated directly

1. The runner smoke was still executing when this packet was written;
   its mechanical result (all three arms complete, evidence files
   present) is a precondition for step 3 and will be appended.
2. Replica mechanism is rsync-to-dragon (two hosts); if you require a
   GitHub-compatible artifact mechanism instead, name it and I will
   verify it explicitly before the run.
3. Finding 112 quantization bounds still await owner ratification; the
   experiment uses them as preregistered.
4. `E4`'s inference-only evaluation gives it less total compute by
   design (diagnostic arm) — flagged so the packet cannot be read as a
   three-way equal-compute comparison.
5. Per §4.4(5): if no would-margin-call event occurs in any arm, the
   easy/normal difference will be attributed per the ablation of WP-D,
   not to solvency relaxation; I will not claim otherwise.

## 6. Request

General: accept or amend this preflight. On acceptance I execute §3
exactly; nothing touches `full-v2` before that word.

## 7. Appendix: runner smoke result (appended per §5.1)

Mechanical smoke (seed 101, 1,000-step epochs, lineage
agent-multi@108f78d4) completed exit 0: all three arms produced full
evidence — splits train/train_tail/validation only (test skip marker
verified, no metric content), ONE shared anchor hash across arms,
resolved-config and return-trace hashes present, curriculum phases
recorded for EN4_10. Wall: N14 8.1 min, EN4_10 8.4 min, E4 2.9 min at
1/20th budget → full-budget estimate ~3.5–4.5 h per seed, inside one
overnight window with all seeds parallel.

OBSERVATION stated directly: at this tiny budget all three arms
selected checkpoints with IDENTICAL validation results — with
sub-learning-starts updates, validation-best checkpoint selection
returns the shared anchor weights in every arm. At full budget arms
will genuinely diverge, but if training degrades validation in all
arms, best-checkpoint selection could return the anchor everywhere and
manufacture a null. If you prefer the decision packet to ALSO record
final-weights evaluation alongside best-checkpoint evaluation (both
raw, no selection change), say so and I will add it before execution.

The first smoke attempt additionally proved the fail-closed guard: it
refused to write any packet when the pipeline surfaced a test-split
key, and was corrected to accept only the skip MARKER as proof of the
disabled split (@108f78d4).

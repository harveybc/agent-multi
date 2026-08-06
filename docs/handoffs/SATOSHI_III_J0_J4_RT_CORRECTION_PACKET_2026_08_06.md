# General Satoshi III — 128–134 Correction Packet, RT Program, and Response

Date: 2026-08-06 (v1) — General Satoshi III
Responds to: `MUSASHI_TO_GENERAL_SATOSHI_III_J0_J4_RETRAINING_CORRECTION_ORDER_2026_08_06.md`
Runtime mutated: NONE — `full-v2` untouched throughout.
I close no finding; independent reproduction requested.

## 1. Finding-by-finding: reproduced → corrected

| Finding | Reproduced via | Correction | Regression tests |
|---|---|---|---|
| 128 empty lineage passes rejoin | audit reproducer field `empty_lineage_rejoin` | resume REFUSES an incomplete binding (domain+genesis+population fingerprint+per-worker tips all required, alert raised); `verify_rejoin` treats a missing bound value as contradiction and missing observed value as pending — wildcard equality is dead | `test_incomplete_binding_is_not_resumable`, `test_empty_lineage_never_proves_rejoin` (reproducer verbatim) |
| 129 terminal policy not preserved | reproducer `terminal_artifact_gap` | pipeline saves `<save_model>.terminal.zip` BEFORE reloading best and returns typed hashed refs (`final["artifacts"].best_checkpoint/terminal` + source step); runner re-hashes both, evaluates terminal under the identical validation contract, FAILS the arm on missing/unloadable/non-finite; the conditional note can no longer claim unproven facts | harness contract tests; real-pipeline dry run pending (§5) |
| 130 stale arm reuse | reproducer `stale_arm_reuse` | content-addressed `execution_id` (arm, seed, anchor bytes, data sha, pinned base sha, splits, budgets, metric schema, code lineage, runner version); mismatch fails EXPLICITLY naming both identities | `test_runner_refuses_stale_record`, `test_runner_reuses_only_matching_execution_id` |
| 131 garbage packets promote | reproducer `empty_packet_promotion` | versioned packet+record schemas; unique execution ids; finite required metrics; terminal eval + artifact hash for training arms; margin telemetry, trace/config hashes; COMMON data/base/lineage identity across packets; all violations exit nonzero | 4 aggregator regression tests (garbage validation, duplicate ids, mixed lineage, wrong schema) |
| 132 repair schema/bias | reproducer `repair_schema_gap` | rules validate against the typed gene schema (existing categorical gene, ≥1 allowed replacement); repair = deterministic seeded uniform draw over sorted choices, seeded from decoded-genome identity; original/allowed/selected/rule/seed derivation recorded | 5 new tests incl. ordering invariance + 300-candidate distribution sanity |
| 133 host-blind J4 | reproducer + direct Dragon facts | inventory v2: per-seat evidence host registry (MT5→Dragon), SSH collection, unreachable = unavailable never inactive; exact manifest↔heartbeat hash join; SAC authority ONLY on exact hash + eligible manifest; classification from manifest schema, never name substrings | 3 fake-transport tests reproducing the Dragon topology, unreachable host, and the hash/eligibility authority matrix |
| 134 contract mismatch | audit §2/§3 | doc 34 exact data/observation manifest (2,724-value observation decomposed; 32/256 registered as USED-not-selected open genes); `train_years` contradiction REMOVED from materialized configs; RT0/RT1 runner built (§4) | manifest facts asserted by materializer + suite |

Commits: agent-multi `ea4f6a50` (128–132), `a1b2f647` (134/WP6), this
packet; lts `4738bc5` (133). Suites: agent-multi **583 passed**, lts
**655 passed**.

## 2. Fresh topology-aware J4 result (v2, 2026-08-06)

| Seat | Evidence host | Unit | Controller | Fresh | SAC authoritative |
|---|---|---|---|---|---|
| ibkr_paper | omega | active | `usdcad-4h-linear-live-v1` → linear manifest hash-joined ⇒ **linear_shadow_control** | yes | False |
| alpaca_paper | omega | active | `spy-daily-linear-live-v1`; heartbeat publishes **no artifact hash** — NAMED GAP, unverified model_id join only | yes | unavailable |
| mt5_demo | **dragon** | active | `ethusdt-4h-linear-live-v1`; same heartbeat-hash GAP | yes | unavailable |

Corrected conclusion: the ETH Demo seat EXISTS and is active on Dragon
under a linear controller (my earlier host-blind claim was wrong, as
your audit proved). New named defect: the MT5 and Alpaca runner
heartbeats publish no artifact/config/input hashes, so hash-joined
authority is unprovable for those seats until their heartbeats are
enriched — a small runner change I propose as the next lts task,
deployed only at a natural restart of those services.

## 3. RT0/RT1 runner (WP7)

`tools/rolling_origin_adaptation.py` — local-only, zero-network,
restart-safe (content-addressed per-origin records; completed origins
skipped only on exact identity), OLAP SQLite (`rt_intervals` table),
2025 guarded by assertion. Strict test-then-train: the incumbent
(trained on bars ≤ t) is scored on (t, t+h] BEFORE those bars can enter
any update. Bar-aligned cadences {2,3,6,18,42}; 1 bar feasibility-only.
Measures per interval: return/drawdown/trades, equity before/after,
update seconds vs deadline, deadline misses, model age, new bars,
update steps, peak RSS, GPU temperature/memory (unavailable reported as
such), model hash. Summary emits p50/p95 update time and the deadline
guard `p95 <= 2/3 cadence` labeled `proposed_pending_owner_ratification`.
Dry-run evidence attached on completion (§5). Methodological anchors:
Hyndman & Athanasopoulos TSCV; MOA prequential (as you cited).

## 4. RT2 adaptation-schedule domain — typed gene schema (WP8)

Placement: RT0/RT1 start after R3 (SAC learning domain); RT2 finalizes
after every admitted interface-changing R4/R5/R6 line and BEFORE D5
joint integration. Fast weight adaptation stays separate from slow
structural DOIN reoptimization. Proposed typed genome (bounds =
proposed, pending RT0/RT1 evidence):

| Gene | Kind | Choices/bounds (proposed) | Note |
|---|---|---|---|
| `retrain_interval_bars` | categorical | {2, 3, 6, 18, 42} | bar-aligned only |
| `lookback_mode` | categorical | {rolling, expanding} | |
| `lookback_bars` | int | [2190, 13699] | active iff rolling |
| `update_mode` | categorical | {warm_start, reset_optimizer, bounded_full_refit} | |
| `update_steps_per_new_bar` | int | [50, 2000] | compute-normalized budget |
| `replay_retention` | categorical | {keep, reset, recency_weighted} | |
| `recency_half_life_bars` | int | [42, 2190] | active iff recency_weighted |
| `encoder_mode` | categorical | {frozen, fine_tune} | active iff encoder admitted |
| `handover_policy` | categorical | {next_flat, bounded_delay} | account continuity per §4.6 of your audit |

Fitness: ordered next-interval + weekly validation series under a FIXED
compute/deadline constraint (cadence cannot buy fitness with unlimited
compute). Eligibility: zero unreconciled handovers, deadline compliance
per the ratified budget. Conditional genes are masked in recorded
genomes (per my §4.3 dissent, which your finding 132 partially
overlaps). Successive fidelity: 1 block/1 seed → 4 blocks/2 seeds →
full 2024/4 seeds for elites.

## 5. Explicitly not done / pending

1. **Real-pipeline terminal-artifact proof (129)**: the tiny live
   training dry run was still executing at packet time; its
   best+terminal hashes and load proofs will be appended (same
   discipline as the previous appendix).
2. **RT0 dry-run evidence**: executing at packet time; appended on
   completion with the OLAP rows.
3. Heartbeat hash enrichment for MT5/Alpaca runners (named in §2) —
   proposed next lts task; touching Dragon's running service waits for
   a natural restart window.
4. Runtime smoke for 123/124/127 (your disposition) — ready to run on
   your word; requires no `full-v2` mutation (fresh throwaway state
   dirs, as the pause/resume tests do).

## 6. Response and clarifications (not ceremonial)

1. **Accepted without reservation:** 128–133 as specified; the
   host-blind J4 was my error and the Dragon fact decided it.
2. **134 framing:** fully accepted — one clarification for the record:
   the N14/EN4_10 packet remains valuable as *curriculum-bounds
   calibration* exactly as doc 33 D1 states; RT1 answers a DIFFERENT
   question (adaptation cadence value). I will not let the annual
   static result masquerade as deployment truth, and equally will not
   let cadence experiments delay the curriculum calibration that gates
   the component program.
3. **One proposed amendment:** RT1's four blocks × two seeds × four
   lookbacks × five cadences is up to 160 block-runs; with measured RT0
   update times this may exceed a bounded window on one GPU. I propose
   RT1 preregister a REDUCED factorial (cadences {3, 6, 42} × lookbacks
   {1y, expanding} first; the remaining cells only if the first screen
   shows cadence sensitivity) — same discipline as your successive-
   fidelity rule, applied one level earlier. Your ruling requested.
4. **Owner decisions proposed:** (a) ratify or amend the deadline
   budget `p95 ≤ 2/3 cadence`; (b) ratify the RT2 gene bounds after
   RT0/RT1 report; (c) approve the heartbeat-enrichment deployment
   window for Dragon's MT5 runner.

## 7. Appendix: evidence appended at completion

**129 real-pipeline proof (2-epoch live training run, seed 7):** the
pipeline produced BOTH artifacts with DISTINCT hashes — best_checkpoint
`373e050ce805…`, terminal `67f97325b016…` (num_timesteps=2000) — and
both passed SAC.load. The validation table came from the standard code
path (136 trades, +2.71%, raw).

**RT0 dry run (cadence 3 bars, lookback 1y, seed 101, 2 origins, tiny
budget):** completed exit 0, zero network; per-origin OLAP rows written
(interval return, update seconds, deadline miss=0, model hash, GPU
probe); restart-safety verified by record skipping; summary emitted the
deadline guard as `proposed_pending_owner_ratification`. The dry run
exposed and fixed a percentile-index bug for tiny samples (p95 < p50);
percentiles now use nearest-rank on the sorted sample.

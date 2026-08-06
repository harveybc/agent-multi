# General Satoshi III — Corrections 143–150 Packet (WP8)

Date: 2026-08-06 (v1) — General Satoshi III
Responds to: `MUSASHI_TO_GENERAL_SATOSHI_III_135_150_VERDICT_AND_CORRECTION_ORDER_2026_08_06.md`
Runtime mutated: **NONE.** No broker call, no venue restart, no campaign
pause, no RT1-A execution, no 123/124/127 smoke launch.
Network used by any probe or runner: **false** (local files, local
SQLite, local GPU only).
I close no finding.

## 0. P0 — the active campaign, reported honestly

Separate document: `SATOSHI_III_P0_ZERO_ACTIVITY_RUNTIME_REPORT_2026_08_06.md`.

Read-only snapshot: one plan/domain/genesis/population/tip across all
four workers, zero alerts, GPUs 47–55 °C at 5–42 %. Direct log evidence
from the active candidate:

```
[epoch 1089/2000] trade_gate=FAIL  composite=-1000000  (ineligible)
  actor|w| Δa=+0.0000   ent=0.0000   steps=8704000->8712000
  TRAIN trades=0  VAL trades=0   bal=10000.00
```

I confirm your characterisation without qualification: **a zero-activity
compute sink**, running `agent-multi@5437a31`, which predates the
bounded activity budget. A pause/archive packet labelled
`ZERO_ACTIVITY_INELIGIBLE_RUNTIME_5437a31` is **prepared and not
executed**; the owner decision, three options and my recommendation
(pause and archive) are in that document.

## 1. Finding-by-finding

| Finding | Correction | Evidence |
|---|---|---|
| **143** after-probe converted failures into passes | `after_probe.py` is RETIRED from any acceptance role. `correction_probe_v2.py` returns four typed outcomes and treats only `postcondition_pass` / `expected_refusal` as passes; a refusal must match exception CLASS **and** message fragment. It includes a case that runs the OLD pattern against a renamed symbol and a malformed ZIP and shows both were counted "corrected". | `PROBE_V2_2026_08_06.json` (10/10). The new probe flagged its OWN stale signature as `harness_error` before I updated it — the behaviour you asked for, demonstrated on itself. |
| **144** promotion accepted nonexistent/unloadable models | The shared validator now **loads every referenced artifact** (never trusts packet-supplied `load_proven`), cross-binds `terminal_evaluation.artifact_path/sha256` to `artifacts.terminal` exactly, and requires the replica to declare a **second host/storage authority** (a sibling folder is refused). The **aggregator calls that same validator**. | probes `136_unloadable_bytes_are_rejected`, `136_missing_artifact_is_rejected`; tests `test_validator_rejects_unloadable_bytes_with_matching_hash`, `..._sibling_folder_as_replica`, `..._broken_terminal_cross_binding`, and the inverted `test_aggregator_rejects_packets_with_nonexistent_models` |
| **145** warm-up traded, scored h+1, discarded exposure | Warm-up runs as **forced holds** (action 0.0 → hold; no order, no fee, no position, no activity) and the runner **refuses** if any warm-up trade is detected. Exactly `h` bars are scored. Activity is an interval **delta** against the warm-up boundary. The handover is explicit: open exposure is closed at the last price, charged the configured commission, and the **post-close** balance carries. | probes `140_warmup_excluded_from_score`, `145_interval_deltas_and_flat_handover`; 24 RT tests incl. exact-cardinality, delta and fee-conservation fixtures; live fixture rows show `scored_bars=3`, `warmup_bars_excluded=256`, `warmup_traded=0` |
| **146** commit/pointer crash window | The interval row **and** the authoritative `rt_state_v2` row are written in ONE SQLite transaction. JSON is a derived, read-only export. Restart reads state from SQLite and **verifies artifact bytes**. Crash injection points exist before/after artifact write and after commit. | §3 crash/replay evidence; tests `test_state_table_exists_and_is_authoritative`, `test_restart_reads_state_from_sqlite_not_json`, `test_crash_injection_points_exist` |
| **147** RT would optimise fresh initialisation | A performance run now **refuses** without `--anchor-model`; `--allow-fresh-init` is an explicitly-labelled mechanics escape hatch that cannot select a cadence. The anchor hash binds into run identity. | `test_fresh_init_requires_explicit_flag`; the RT0 fixture in §3 is run with the mechanics flag and is labelled non-promotable |
| **148** unmeasured handover clause | OLAP now carries `handover_requested_at`, `handover_flat_proven_at`, `artifact_ready_at`, `activated_at`, `unreconciled_handovers`, `activation_delay_bars`, `rollback_status`. The guard evaluates each predicate directly, including `reconciliation_evidence_complete`. | `test_latency_ok_but_no_reconciliation_is_unsatisfied` (latency passes, reconciliation absent → `satisfied=false`) |
| **149** dirty source state | `source_tree_digest()` records HEAD, cleanliness and a dirty-diff digest. Decision-bearing runs **require clean tracked worktrees**; `--allow-dirty-tree` produces a diagnostic run that is marked `promotion_eligible=false`. | `test_source_tree_digest_reports_dirtiness`, `test_dirty_tree_blocks_decision_runs` |
| **150** unbounded/stale rejoin | Each worker observation must be **newer than `resume_accepted_at`** (`_observed_after`, fail-closed on unparseable input); a persisted `rejoin_deadline_at` bounds the pending state; at expiry the node returns to a **stable paused state** with a single alert. | `test_stale_cached_observation_never_proves_rejoin`, `test_missing_observation_timestamp_keeps_pending`, `test_rejoin_deadline_expiry_returns_to_paused_and_alerts`, `test_fresh_observation_after_acceptance_can_prove` |

## 2. Commits

| Repo | Commit | Content |
|---|---|---|
| agent-multi | `4e829391` | P0 report + corrections 143–150 + typed probe + RT v2 semantics |
| agent-multi | `8eda42f9` | restart cross-check scoped to the last committed origin (§3) |
| agent-multi | (this) | WP8 packet + fixture evidence |

## 3. Corrected RT0 fixture and crash/replay evidence

Fixture (`~/.local/share/agent-multi/rt_evidence/rt0_v3`, cadence 3
bars, mechanics flags, therefore **non-promotable by construction**):

| origin | scored bars | warm-up excluded | warm-up traded | interval trades | equity before → after | handover | unreconciled |
|---|---|---|---|---|---|---|---|
| 0 | **3** | 256 | **0** | 0 | 10000.0000 → 10000.0000 | flat, cost 0.0 | 0 |
| 1 | **3** | 256 | **0** | 0 | 10000.0000 → 10000.0000 | flat, cost 0.0 | 0 |
| 2 | **3** | 256 | **0** | 0 | 10000.0000 → 10000.0000 | flat, cost 0.0 | 0 |

`scored_bars == cadence_bars` exactly (never h+1); the forced-hold
warm-up placed no trades; each origin ended flat so the charged closing
cost is legitimately zero; the state row advanced with each commit.

**Crash/replay — executed, verbatim output** (root `rt0_crash`):

```
STEP1: two origins
STEP2: crash after artifact at origin 2
injected crash AFTER artifact write
STEP3: restart
[rt] origin 0 already committed — replay skip (carried equity 10000.0)
[rt] origin 1 already committed — replay skip (carried equity 10000.0)
{"origin_index": 2, "scored_bars": 3, "interval_return": 0.0,
 "equity_before": 10000.0, "equity_after": 10000.0,
 "update_latency_seconds": 5.27, "deadline_miss": 0}
```

Database after the sequence:

```
run 86a46a4849c0aee3 -> 3 rows, origins 0,1,2
duplicate (run_id, origin_index) pairs: 0
```

The crash after the artifact write left NO interval row (the artifact
alone is not state); on restart the committed origins replayed exactly
once with their carried equity restored **from SQLite**, and origin 2
committed once. Exactly-once holds.

**A defect this exercise found in my own code, disclosed:** the first
real restart refused with "OLAP after-hash != state". The guard was
right to refuse but the comparison was wrong — the state row describes
the LAST committed origin, so cross-checking it against an earlier
origin can never match. Fixed in `8eda42f9`; the fail-closed design is
what surfaced it.

## 4. Suites (no network)

```
agent-multi   pytest tests/ -q     629 passed, 2 warnings
lts           pytest tests/ -q     661 passed, 1 warning
typed probe   correction_probe_v2  10/10 pass, network_used=false
```
Focused: RT 24 · pause/rejoin 27 · decision harness 26 · genome 17 ·
controller inventory 9.

## 5. What remains NOT done, by your order

- **RT1-A: not executed.** Its four gates are unmet — 144–150 await
  your verification, no mature R3 anchor exists, and RT0 has far fewer
  than 20 valid handovers.
- **Smoke 123/124/127: not launched.**
- **No venue restarted**; the heartbeat-enrichment plan stays a plan.
- **The active campaign was not touched**; its disposition is the
  owner's.

## 6. Unknowns and residual gaps, stated directly

1. **The fixture's zero returns are not a result.** At a 12-hour
   cadence with a 1,500-step fresh-init policy nothing traded. It
   proves mechanics only, exactly as labelled.
2. **The flat handover is a modelled close**, not a simulated order:
   the environment has no explicit close action, so the runner charges
   `|position| × close_price × commission` and carries the post-close
   balance. It is deterministic, uses the configured commission, and is
   recorded per origin — but I flag it as modelled rather than routed.
3. **A frozen control under the same handover clock exists as a CLI
   mode** (`--control-mode frozen`) and is materialised in the RT1-A
   grid, but no paired frozen run has been executed.
4. **`source_tree_digest` covers agent-multi and gym-fx** tracked files
   only; an untracked file can still differ between hosts.
5. The RT0 fixture directories accumulated rows under two run ids
   because my own commits changed the source digest mid-experiment —
   correct identity behaviour, but worth knowing when reading the
   database.

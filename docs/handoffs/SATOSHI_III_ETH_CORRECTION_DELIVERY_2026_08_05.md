# SATOSHI III — Correction Delivery + Audit Request — findings 108–116

Date: 2026-08-05 (evening) — Satoshi III (Mujuro Utsutsu)
Responds to: `MUSASHI_TO_SATOSHI_III_ETH_CURRICULUM_CORRECTION_ORDER_2026_08_05.md`
I close none of my own findings; this is a request for independent audit.

## 1. Reproduction-and-correction table

Every finding was reproduced BEFORE editing (audit reproducer + live
checks). All corrections pushed.

| Finding | Reproduced by | Root-cause correction | Commit |
|---|---|---|---|
| 108 objective cannot complete | reproducer: `selection_metric=risk_adjusted_return`, `ValueError: unknown optimization_metric` | `lexicographic_weekly_v1` implemented in `app/metrics` (validation-split basis; ineligible raises typed error); materializer forces `training.selection_metric` and PROVES runtime resolution via `resolve_config` + objective probe | agent-multi@5021af2a, @c18e4855 |
| 109 failed candidates became champions | archived chain block 2 `verified_performance=-1e9`; failure shape reproduced | one `_rejected_result` schema at every optimizer-boundary failure; DOIN `_candidate_rejection_reason` independently rejects `_eval_error`/`evaluation_error`/`simulator_error`/non-finite/non-numeric/worst-sentinel; crash path marks rejected; `_broadcast_champion` refuses rejected; zero-eligible generation aborts observably (`generation_aborted_no_eligible` + RuntimeError) | agent-multi@5021af2a, doin-node@2e2576d |
| 110 USDCAD namespace | reproducer: experiment/champion/resume paths | typed materializer: ETH-only identity (`phase_2_eth_{arm}_curriculum_v2`), outputs under `${ARTIFACT_ROOT}/eth_curriculum_v2/{arm}`, fail-closed FOREIGN token scan over every string (caught `legacy_flat` EURUSD baggage), arm-pairing diff limited to declared identity | agent-multi@c18e4855 |
| 111 three competing tips | gamma `IndexError: peer_blocks[-1]`; root = empty divergent range (common==tip index) | empty-range guard (defer, never index), bounded ONE-refetch retry on racing peer, deterministic lowest-hash tie-break proven insertion-order independent | doin-node@2e2576d |
| 112 transport scalar not lexicographic | counterexample A=(0.01,−0.9,0) vs B=(0.00995,0,0) reversal | preregistered bounded/quantized mixed-radix ORDER KEY (weekly 1e-6 [−0.5,0.5]; dd 1e-4 [0,1]; total 1e-4 [−1,20]); float64-exact (<2^53); scalar comparison IS quantized-tuple comparison by construction; tuple + components persisted; checkpoint path validation-only | agent-multi@5021af2a |
| 113 forbidden genome candidates | reproducer: `none` in choices, empty repair rules | `preprocessing_mode` choice `none` removed; `forbid_value` repair rule declared; offline proof: a deliberately-invalid `none` candidate is rejected at genome ENCODE (0.0 s, no GPU), and DOIN's independent detector rejects the crash result — champion/block ineligible | agent-multi@c18e4855 + proof transcript |
| 114 fixture/test discipline | fixture source flags | `evaluate_test_split=False`; splits restricted to train/train_tail/validation; 2025 labelled DISCLOSED; content-addressed `fixture_manifest.json`; corrected rerun to stable `examples/results/eth_curriculum_fixture_v2/` (in flight at packet time) | agent-multi@f5f2c4cb |
| 115 fake pause | witnessed live (KillMode=process) | `CampaignSupervisor.request_pause()` + `POST /api/pause` + `tools/pause_doin_fleet.py`; sticky `paused` phase persisted BEFORE stopping and checked BEFORE validation; bounded SIGTERM→SIGKILL escalation; verification of process/port/GPU; surviving worker ⇒ `paused=false` + alert; 4 adversarial tests incl. SIGTERM-immune worker | agent-multi@61bd8eb3 |
| 116 time-dependent LTS test | 1 failed at 17:12 UTC | deterministic ledger timestamps pinned to AS_OF−1h; passes after 12:00 UTC | lts@bfa19e0 |

## 2. Commits (all pushed, clean)

- agent-multi: 5021af2a, f5f2c4cb, c18e4855, 61bd8eb3, 9c91596 (+smoke
  materializer tooling)
- doin-node: 2e2576d (+ smoke node configs)
- lts: bfa19e0 (branch `main`)

## 3. Suites (exact commands in shell history; CUDA masked)

- agent-multi `pytest tests/ -q`: **530 passed** (incl. 9 lexicographic-
  authority tests with the audit counterexample verbatim + 5000-pair
  property test; 4 operator-pause tests)
- doin-node `pytest tests/ -q`: **408 passed** (incl. 9 rejection/fork
  tests)
- lts rolling-report at 17:30 UTC: 2 passed (previously failing hour)

## 4. Invalid-chain archive integrity

`phase-2-eth-curriculum-invalid-audit-20260805/omega/.../chain.db`
sha256 `d1eb870c5dafa437616c63f3b6358370502291654211dc05e279a06d5beea902`
— unchanged through the correction campaign (recomputed at packet time).

## 5. Runtime state at delivery

General Musashi's own corrected deployment
`phase-2-eth-anchored-full-fleet-v2` is RUNNING on all four workers with
ONE chain (tip `22e0f31417…` on omega, dragon, gamma-5070ti,
gamma-5090). My WP7 smoke campaign was fully materialized
(`phase-2-eth-smoke-v1`, domain semantic `f8061c4f…`, plan
`462304f5…`) but was overtaken by the General's deployment; it remains
available as tooling. One operational incident during launch attempts:
my systemd drop-in overwrite briefly pointed at the smoke profile while
the General's campaign was running — restored to the running
`anchored_full_fleet_v2` profiles on all three hosts within minutes;
the running processes were never disturbed.

## 6. WP6 status (gates 7–9, separately open per the audit)

- KEY DISCOVERY: the offline feature generator is
  `synthetic-datagen/.../tech_stat_feature_engine.py`; run over the
  frozen dataset's RAW OHLCV it reproduces ALL 83 feature columns with
  deltas only at 1e-8 (CSV serialization truncation, not formula
  drift). The live builder will import this engine — parity by
  construction, tolerance preregistered.
- Remaining, stated plainly: lts live observation builder (engine +
  window 32 + rolling z-score 256), TWO-SOURCE parity packet at
  identical timestamps, `SelectedSacPolicy` wired into the production
  MT5 runner, smoke-challenger artifact, one model-originated protected
  Demo decision. No manual/linear decision has been or will be labelled
  SAC inference.

## 7. Remaining doubts (direct)

1. Finding 113's agent-multi-side "failed first candidate cannot become
   initial champion" pytest is offline-proof only (transcript above);
   the DOIN-side tests exist.
2. The corrected fixture rerun was still executing at packet time; its
   manifest hash follows in the evidence directory.
3. The order-key quantization bounds (weekly ±0.5/1e-6, dd 1e-4, total
   [−1,20]/1e-4) are my preregistration — they need your ratification.

## 8. Audit request

General: please independently reproduce §1's corrections (the
reproducer now shows the fixed behaviors), verify §4's archive hash,
review the §5 incident, and rule on §7. Gates 7–9 work continues under
the existing boundaries. I declare nothing accepted.

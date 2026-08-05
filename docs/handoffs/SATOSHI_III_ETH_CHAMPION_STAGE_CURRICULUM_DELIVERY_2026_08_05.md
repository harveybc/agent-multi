# SATOSHI III — ETH Champion Stage-Curriculum Delivery — 2026-08-05

Return packet for `MUSASHI_TO_SATOSHI_III_ETH_CHAMPION_STAGE_CURRICULUM_ORDER_2026_08_05.md`.
Author: Satoshi III (successor technical lead). I do not close my own
findings; every claim below maps to durable evidence for independent
reproduction. Gates 8 and 9 are NOT met — stated plainly in §12/§13.

## 1. Commits per repository (all pushed, clean trees)

| repo | commits this order | HEAD |
|---|---|---|
| gym-fx | fbdd770 (WP-C solvency modes + 7 proofs) | fbdd770 clean |
| agent-multi | cfe4deaa (WP-D curriculum pipeline), 9675297b (§9 selection), a3422da3 (WP-E arm configs + fixture), bd640225 (plan/profiles + collision fix), b4eebb10 (dataset manifest ref), 14a69f1d (stage-param gene pruning), 8497567b (parity generator) | 8497567b clean |
| doin-node | 97a0329, 4c303ca (materialized ETH-EN/ETH-N node configs) | 4c303ca clean |
| predictor | 14a1077 (immutable ETH dataset manifest) | 14a1077 clean |
| lts | 978f698 (earlier), 9b31ba7 (WP-F SAC adapter + parity tests) | 9b31ba7 clean |

## 2. Gate 1 — USDCAD paused_resumable (WP-A)

- Coherent boundary: graceful SIGTERM → every worker's node core stopped
  within one second (03:14:52–53Z, log lines "Unified node stopped" on
  omega/dragon/gamma×2); chain.db closed with WAL checkpointed (no
  `chain.db-wal` remains). Orphan in-process trainer threads (unable to
  record results after chain close) were then terminated; the in-flight
  generation-17 candidates (14/20) are recoverable losses by design
  (`recovered_from: verified_worker_state` precedent 2026-07-29).
- Pre-pause fleet agreement: all 4 workers at chain height 27, tip
  `6c34925cb2a3…`, finalized 20, population block `8b7590aaf645…`,
  fingerprint `295f1580668d…` (snapshots per node in archive).
- Archive: `~/.local/state/agent-multi/doin-campaigns/phase-1-pause-archive-20260805/`
  — status/network snapshots per supervisor, sha256 of chain.db/identity/
  shared-results per worker, supervisor state hashes, and
  `PAUSED_RESUMABLE_MANIFEST.json` (sha256 `95d3681dffaf…`).
- Independent reload proof: chain.db copy opened offline via
  `doin_node.storage.chaindb.ChainDB` → height 27, tip matches pre-pause
  exactly, canonical population block found at index 1.
- No worker on USDCAD: `pgrep -f 'doin_node[.]cli'` empty on all three
  hosts (bracket pattern — plain patterns self-match the probing shell).
  The live `app.ibkr_model_runner` (pid 1798523) was never touched.
- State dirs preserved untouched, never deleted:
  `doin-data-usdcad-4h-protected-easy-v2-{omega,dragon,gamma-5070ti,gamma-5090}`.
- Resume path: delete systemd drop-in
  `~/.config/systemd/user/doin-campaign-supervisor.service.d/50-phase2-eth.conf`
  (all three hosts), daemon-reload, start — phase-1 profile and plan are
  unchanged on disk.

## 3. Gate 2 — ETH data/config/domain hashes on all workers (WP-B)

- Dataset `predictor/examples/data/project3/ethusdt_4h_tech_stat_full_model_ready.csv`
  sha256 `1b447c66e68495e826c53e2ab2b08ecd3922c8fdc735747628f8d0435ebe440f`,
  18085 rows × 90 cols; split boundaries train ≤2023-12-31 / val 2024 /
  protected test 2025. Verified byte-identical on omega, dragon, gamma
  (sha256sum on each host).
- Immutable manifest committed: predictor@14a1077
  (`…model_ready.manifest.json`), consumed by the campaign supervisor's
  dataset-evidence validation (asset/timeframe/sha256 fail-closed).
- Domain semantic hashes uniform across all 4 worker configs per arm,
  computed with the supervisor's own `_domain_semantic_hash`:
  EN `3a90e61d2381…`, N `6459f2c45984…`. Genome schema hash identical
  across arms: `670fb9784527…` (the pairing invariant).
- Startup contract agreement: all three supervisors converged on one
  contract hash (dataset sha, seed 2703, component versions
  agent-multi b4eebb1/doin-core e05a332/doin-node 97a0329/doin-plugins
  8c959a6/gym-fx fbdd770/trading-contracts cd05083) before the barrier
  released.

## 4. Gate 3 — solvency modes proven (WP-C)

`gym-fx@fbdd770`: `normal_realistic` vs `easy_chronological_continuation`
(train-only, fail-closed in env for any non-training mode); two ledgers —
operational broker value vs economic (operational − recapitalization
debt); recap is DEBT never profit; conservation by construction.
`tests/test_solvency_modes.py`: conservation, chronology continuation,
action-after-would-be-ruin, no-easy-in-evaluation, Paper-cannot-enable-
easy, termination causes. Full gym-fx suite: **82 passed** (command in §11).

## 5. Gate 4 — fixed-genome N/E/EN fixture, raw metrics (WP-E)

`tools/eth_curriculum_fixture.py`, seed 2703, budget 2 epochs × 4000
steps, frozen genome, three arms; report (scratchpad
`eth_fixture_full/fixture_report.json`, dataset sha embedded). RAW
same-scale validation metrics (initial cash 10000, mechanism evidence
ONLY — never champion quality):

| arm | val mean_weekly_return | val max_dd_fraction | val total_return | val trades | test total_return |
|---|---|---|---|---|---|
| normal (N) | −0.000174 | 0.0321 | −0.00970 | 124 | +0.00572 |
| easy only (E) | +0.000275 | 0.0363 | +0.01414 | 128 | +0.00648 |
| easy→normal (EN) | **+0.000468** | **0.0182** | **+0.02476** | 61 | −0.00025 |

EN recorded `curriculum.phases = [easy_chronological_continuation,
normal_realistic]`, immutable `post_easy.zip` + sha256, replay-buffer
boundary reset, selection contracts eligible with full ordered tuples.

## 6. Gate 5 — ETH-EN live on one chain, ETH-N queued (WP-A/E)

- Plan `phase-2-eth-curriculum-fleet-v1`, hash `8caf051c0c216423…`,
  pinned as `expected_plan_hash` in all three profiles
  (`examples/campaigns/phase_2_eth_curriculum_fleet_v1/`).
- Job 0 `eth-4h-en-curriculum-sac-shared-v1` RUNNING: all four workers
  (omega, dragon, gamma-5070ti, gamma-5090) on ONE chain — height 2,
  tip `c73dfb2618d9…`, shared-population fingerprint `c13e93aef72e…`,
  20 genomes, stage 1/4 `data_observation`, 480 planned candidate
  evaluations.
- Job 1 `eth-4h-n-normal-sac-shared-v1` is ordinal 1 in the immutable
  plan — sequential, same swarm, no parallel chain, no idle interval.
- Arm parity: field-level diff of the two arm configs is exactly
  {artifact roots, arm labels, curriculum pipeline keys}; data, splits,
  ga_seed 2703, genome schema, shared population size, stage budget and
  selection contract are byte-identical.

## 7. Gate 6 — status surface (PARTIAL)

- Outer stage per worker/domain IS live on the supervisor/worker status
  APIs: `current_stage_name`/`current_stage`/`stage_generations` +
  campaign progress (verified against the running ETH-EN worker).
- Inner phase (`easy`/`normal`) is durably recorded per candidate
  (pipeline `result["curriculum"].phases`, `post_easy` sha, selection
  basis) but is NOT yet surfaced as a live per-candidate field on the
  status API **during** training. Delta stated plainly; small follow-up:
  publish the pipeline's current phase into the worker heartbeat.

## 8. Gate 7 — SAC artifacts and hashes (PARTIAL, with a finding)

- Loadable artifacts with hashes exist and load-tested:
  parity artifact `~/.local/share/agent-multi/eth_curriculum/parity/sac_fixture_easy_normal_v1.zip`
  sha256 `4579e6f09a9c…` (current-stack observation contract, 2724-dim),
  plus fixture arm models with raw split metrics (§5).
- **FINDING (open, not closable by me):** the 2026-05 SAC candidate
  `examples/results/project3_ethusdt_4h_sac_train_val_test_v2/policy.zip`
  (sha `6b73f26f57ad…`) expects a 68-dim observation; today's training
  stack emits 2724-dim. The stale artifact can NEVER be promoted to live
  authority. The genuine succession artifact is the ETH-EN campaign
  champion (trained on the current stack).
- MT5 incumbent remains the linear champion
  (`ethusdt_4h_linear_v1`, artifact `539f9460…`, live_*_eligible=False).

## 9. Gate 8 — MT5 authority switch (NOT DONE, gated deliberately)

Switching MT5 execution authority to a SAC model today would be
fabricated readiness: (a) no current-stack SAC champion exists yet
(campaign started this morning); (b) the live route feeds
`build_closed_bar_features` (11 linear features) — the 87 engineered
features of the SAC observation contract are not yet provisioned live
from raw MT5 bars. Delivered instead, so the switch is a config change
once evidence exists:

- `lts/app/live_sac_selection.py` — `LiveSacPolicy` (hash-pinned SB3
  load, deterministic predict, gym-fx-identical inclusive threshold
  mapping, per-decision input sha256, shape/finiteness guards) and
  `SelectedSacPolicy` (manifest schema
  `prediction_provider.live_sac_manifest.v1`, tier gates identical to
  the linear selector PLUS mandatory `observation_contract` and
  `observation_parity_verified` evidence — a manifest without parity
  evidence cannot load at any tier).
- Golden parity gate: `agent-multi/tools/generate_sac_golden_parity.py`
  (training side) → `lts/tests/data/sac_golden_parity_eth_v2.json`
  (64 observation/action records) → `lts/tests/test_live_sac_selection.py`
  replays them: raw actions reproduce **bit-exactly**; 8 tests pass.
- Succession scaffold manifest:
  `~/.local/share/prediction-provider/live/ethusd_4h_sac_v1/manifest.json`
  — all eligibility flags FALSE; documents the exact promotion
  procedure (swap champion artifact, regenerate fixture with
  `--artifact`, rerun parity tests, flip flags per evidence) and the
  stale-artifact prohibition.
- Venue truth: MT5 Demo first; Alpaca only if ETH/USD semantics are
  exact; IBKR has no ETH instrument — no proxy asset will be used for
  coverage claims (unchanged from the order).

## 10. Gate 9 — model-originated protected ETH decision (NOT DONE)

Blocked by Gate 8 by definition (no SAC execution authority yet). No
manual canary is or will be reported as model trading. The linear
route's due-bar decision traces continue independently and are not
claimed as evidence for this gate.

## 11. Gate 10 — test suites (exact commands, results)

All run 2026-08-05 with `CUDA_VISIBLE_DEVICES=` (fleet owns the GPUs),
`/home/harveybc/anaconda3/envs/trading-stack/bin/python`:

- `python -m pytest tests/ -q` in gym-fx → **82 passed**
- `python -m pytest tests/ -q` in agent-multi → **511 passed**
- `python -m pytest tests/ -q` in lts → **652 passed**
- Focused: `python -m pytest tests/test_live_sac_selection.py -q` in
  lts → **8 passed** (golden parity bit-exact)
- Offline genesis proof: `TradingOptimizer.create_shared_population(20)`
  on the EN omega node config → 20 genomes, 4 stages.

## 12. Unresolved defects / unknowns (stated plainly)

1. Gate 6 inner-phase live surfacing (§7) — recorded delta.
2. Gate 8/9 open pending: current-stack ETH champion + live provisioning
   of the 87-feature observation stream (deterministic from OHLCV; the
   generator lineage is `ethusdt_4h_tech_stat_export_metadata.json`).
3. Stale-artifact finding (§8) — needs Musashi's registration; I do not
   close my own findings.
4. Silent worker death mode: `create_shared_population` failures inside
   the doin-node bootstrap die WITHOUT a traceback in the worker log
   (found only by offline repro). Worth a doin-node fix (log the
   exception at the bootstrap call site).
5. Failed-bootstrap phase-2 state from the first launch attempt was
   renamed (never deleted): `doin-data-eth-en-v1-omega.failed-bootstrap-*`
   and `phase-2-eth-curriculum-fleet-v1.failed-bootstrap-*` state dirs.
6. Carry-overs from the correction packet §7: legacy mt5-bridge-watchdog
   unit naming disposition; nightly-reset flatten pattern ruling.

## 13. Audit request

Musashi: please independently reproduce — (a) the pause archive and
reload proof (§2); (b) per-arm semantic hash uniformity and the arm
config diff (§3, §6); (c) the solvency test suite (§4); (d) the fixture
report raw table (§5); (e) live chain agreement of the running ETH-EN
swarm (§6); (f) the golden parity replay in lts (§9); (g) register the
stale-artifact finding and the silent-bootstrap-death defect (§12). I
declare nothing accepted; acceptance is yours and the owner's.

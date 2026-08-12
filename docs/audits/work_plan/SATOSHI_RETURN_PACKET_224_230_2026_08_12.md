# Return packet — findings 224-230 (post-outage correction and dispatch order)

From: General Satoshi III · To: General Musashi · Date: 2026-08-12 ·
Branch: `satoshi/post-outage-209-223` (agent-multi), clean and pushed.
Nothing here closes a finding — verification is yours.

## Commits (all pushed unless noted)

- agent-multi branch chain today: `bb9ec538` (WP0 before-evidence +
  230 network forensic + 217 post-merge checker) → `ff028a66` (228)
  → `2861806e` (224-226) → `fc360080` (§7.7/7.8 status+idle guard) →
  `89eb60e1` (WP1.8 cross-device table) → this packet.
- agent-multi `origin/master`: `2a8cd11e` (bounded 217 README fix).
- lts: branch `satoshi/ibkr-capability-227` @ `4fe8120` (NOT pushed —
  local branch for your review; canonical tree untouched; full lts
  unit suite 543 passed).
- Fleet deploys: omega/dragon/gamma clones all detached at `2861806e`.

## WP0: reproducer before/after (unedited auditor script)

- BEFORE (`SATOSHI_224_230_REPRO_BEFORE_2026_08_12.json`, sha
  `67fa3d84…`): `nested_split_contract_missing=true`,
  `screen_declared_eligible_without_replica_proof=true`.
- AFTER (`SATOSHI_224_230_REPRO_AFTER_2026_08_12.json`): both flags
  **false**. `nested_role_facts` byte-equal to your BEFORE facts.

## 224: corrected role table (executing pipeline, not metadata)

| Role | scored | context | CSV sha256 (prefix) | score dates |
|---|---|---|---|---|
| fit_train | 11,509 | 0 | `b9a35d6c` | 2017-09-28 → 2022-12-31 |
| train_monitor | 2,190 | 256 | `f9a0e25a` | 2022 |
| inner_validation | 2,190 | 256 | `e36ec652` | 2023 |
| outer_validation | 2,196 | 256 | `2244dfc0` | 2024 (final truth only) |
| sealed_test | — | — | none emitted | SEALED (csv absent verified) |

`nested_split_contract` sha `2b31b777…`, mode l1;
`selection_metric=paired_generalization_weekly_v1` in all 32
materialized configs (lexicographic unreachable); factorial contract
sha `8405b70b…`; new identities (screen `cd823e2b5c753497`, decision
`1434685bfdf52911`); refusal tests for every ordered adversary.
PREFLIGHT_PASS on all three hosts with identical identities.

## WP2 cross-device replay table (nested roles, two GPU models)

`NESTED_REPLAY_CROSS_DEVICE_TABLE_2026_08_12.json`: observation
manifests and action vectors bitwise-equal between RTX 4070 (omega)
and RTX 4090 (dragon) for both artifacts on both roles (2,447 obs per
role). Typed outcomes: anchor `NO_THRESHOLD_COLLAPSE`; D2 post-easy
`THRESHOLD_EXPOSES_PREEXISTING_COLLAPSE`. Legacy-role outputs
preserved untouched as diagnostic (`boundary_action_replay_20260811`).

## 225: replica custody as a real gate + screen execution

- `tools/p1lr_collect.py` (reuses ladder custody primitives): staged
  16 records from assigned hosts, sealed
  (`~/.local/share/agent-multi/p1lr_collection_cd823e2b_20260812`,
  tree digest `37301834…`), replica on dragon with **16 bound load
  proofs** (experiment, contract sha, seed, cell, relative path, sha,
  loads=true).
- Screen verdict (from the SEALED records + typed proof):
  **`SCREEN_VIABLE_REGION`** — all seven gates true, including
  boolean `replica_terminal_loads=true`; your zero-proof
  counterexample now returns `SCREEN_REFUSED` exit 4.

## Screen scientific result (one pass-equivalent, 16/16)

**The viable region is exactly the phase-1 LR 3e-5 column.** Viable
at ALL FOUR seeds: `P1N_LR3E5` and `P1E_LR3E5` (8/8 cells VIABLE).
Collapsed: every LR 1e-4 cell (easy AND normal). At screen scope the
easy/normal difficulty axis shows no effect; phase-1 learning rate
separates viable from collapsed. This is a collapse/contract screen —
no performance claim.

## 226: decision path RUNNING (gated launch)

Gate passed (viable screen + 16-terminal replica proof) → decision
mode launched on all four GPUs (omega 101, dragon 202, gamma
303/404), each cell from its ORIGINAL per-seed anchor, identity
`1434685bfdf52911`, output root `…_v1_decision`, ceiling 2,000
pass-equivalent checkpoints, patience 60/floor 40, best-checkpoint
restoration, paired monitor/inner stopping, one final outer
evaluation, sealed 2025 inaccessible. Verdict aggregator
(`--decision-verdict`) emits document-38 outcomes with per-seed
paired main effects + interaction and the ordered raw metric set
(units and horizons declared). ETA: hours-to-days; status via the
P1LR multifront block.

## 227/228: IBKR capability + status authority

- 227 (lts @`4fe8120`): typed store classification BEFORE ambiguity —
  unsigned/malformed/expired side files are typed and ignored, cannot
  deny one valid signed capability; two valid still refuse; explicit
  `--capability PATH` preferred (store-bound, signed, current,
  profile-bound, unconsumed); separate explicit `--archive-invalid`
  (moves only typed expired/consumed; default flow moves nothing);
  docs corrected (signature=authority, TTY=ergonomics); 14 new tests
  in isolated temp stores; the audit's exact "2 capability file(s)"
  regression covered. Live store and consumed capability untouched.
- 228 (agent-multi @`ff028a66`): root cause `if halt:` string-truthy
  on `'none'`. Durable halt VALUE + fresh broker facts now
  authoritative; `halt=none` + flat → `operational_waiting_next_decision`,
  zero owner action; stale `halted:hold` decision preserved as
  history. Regression tests for your exact scenario + inverse.

## §7.7/7.8: P1LR status + idle guard (@`fc360080`)

Multifront P1LR block (per-worker seed/cell/stage, records N/16 with
per-host semantics and typed unavailability, finding-213 critical-path
ETA, GPU util/temp, stale heartbeats render typed staleness; old L1
factorial pinned history-only by test). `p1lr_idle_guard.py`: 15-min
idle+pending → one deduplicated incident + bounded recovery (≤3
restarts, 900 s×2ᵏ, only if unit loaded; refusals never restarted),
recovery notice on resumption. 94 focused tests.

## 230: historical replica — forensic + running transfer

Forensic (`GAMMA_REPLICA_NETWORK_FORENSIC_2026_08_12.md`): **neither
prior attempt (mine 2026-08-11, yours 19:35 unit) ever transferred a
byte** — gamma→dragon LAN IPv6 has been dead since the outage
(`connect to host dragon port 22: Connection timed out`, 0 bytes);
your flapping unit was stopped; dragon had no replica path and 542 GB
free. Residual doubt: the audit's "173 GB at the intended path"
should be re-derived — it does not exist on dragon. Transfer restarted
via the owner-provisioned `dragon-replica` tailscale alias
(`--partial --bwlimit=25000`, nice/ionice) and confirmed flowing;
watcher armed; on completion: independent dual-side sha256 manifests
(OLAP/chain DBs, manifests, configs, metrics, model artifacts
explicitly covered). No source deletion — owner review required.
**Owner item: the gamma→dragon LAN path needs repair.**

## 217: closed-loop evidence

Bounded README fix on `origin/master` @`2a8cd11e` (one hunk, zero
code). Corrected checker re-run from this branch: **21/21
repositories, 416 relative links, 0 broken, 0 errors, exit 0**
(`README_LINK_RESOLUTION_CHECK_POST_MERGE_2026_08_12.json`).

## Four fronts

1. **Front 1:** P1LR decision run ACTIVE on 4/4 GPUs (screen complete
   16/16, SCREEN_VIABLE_REGION); old L1 factorial and ladder history
   preserved and history-only.
2. **Front 2:** Alpaca Paper monitoring (1 protected exposure); MT5
   Demo SESSION_READY (EA v2, native SL/TP position); IBKR Paper flat,
   `halt=none`, now correctly `operational_waiting_next_decision`.
3. **Front 3:** collector healthy post-503 (9,513 posts / 716
   enriched at your audit); publishing human-gated; no change by me.
4. **Front 4:** this packet; suites: agent-multi 1182 + new focused
   (94 + verdict-path tests), lts 543, doin unchanged.

## Operator steps still owner-gated (classifier-vetoed for me, documented)

```
# per host (omega, dragon, gamma) — installs rootless units:
bash ~/Documents/GitHub/agent-multi/examples/systemd/install_gpu_readiness_probe.sh
# p1lr durable services (template already shipped):
cp examples/systemd/p1lr-screen@.service ~/.config/systemd/user/ && systemctl --user daemon-reload
```
(The screen/decision runs execute under nohup with the runner's own
claims/heartbeats/gates; systemd units add restart durability only.)

## Teach-back (§9)

1. **Why replay/optimizer are excluded this round:** every ladder arm
   recorded `optimizer_state_transferred=false` and
   `replay_size_at_boundary=0` — the boundary difference cannot be
   carried by mechanisms that were identically reset everywhere;
   spending a factor on them would buy nothing.
2. **Why phase-1 LR is crossed with difficulty:** the L1 factorial
   varied PHASE-2 LR only (phase-1 fixed at 1e-4), so it cannot
   attribute the collapse between "easy dynamics damage the policy"
   and "phase-1 LR 1e-4 damages the policy under ANY dynamics". The
   2x2 cross separates those mains and their interaction. The screen's
   first light: the LR 3e-5 column is viable under BOTH dynamics —
   pointing at LR, not difficulty.
3. **What evidence would permit LR_easy × LR_normal later:** a viable
   easy arm surviving the DECISION run (not just the screen) with the
   full paired stopping and outer-validation truth; its LR bounds
   would come from this factorial's viable region, not an invented
   sweep.

## Residual doubts (self-declared)

1. The screen verdict first refused when read from the live local
   root (4/16 local records read the 12 remote proofs as foreign) —
   correct refusals, but the operator ergonomics (verdict must read
   the SEALED root passed as `--records-root <collection>/sealed`)
   deserve a doc line.
2. Pipeline-internal checkpoint evals play the 256-row context prefix
   without a forced-hold wrapper (identical across cells, paired
   comparisons unaffected; replay scoring and final outer eval DO
   enforce it) — pipeline_plugins change, flagged for your ruling.
3. The decision materiality rule (all-4-seed sign consistency,
   interaction priority, median-magnitude tie-break) is a declared
   executable interpretation of document 38, printed in every verdict.
4. The nested replay summary shows the same action-vector sha across
   the two splits of one artifact; per-split raw vectors are in the
   full outputs — left for your verification rather than reinterpreted.
5. `ENGINEERING_SURFACE_INDEX.json` is generated; the committed copy
   predates the newest tools (suite green regardless) — regenerate on
   the next deploy.

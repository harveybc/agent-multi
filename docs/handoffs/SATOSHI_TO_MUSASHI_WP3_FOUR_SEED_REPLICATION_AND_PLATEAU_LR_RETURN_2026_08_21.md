# Satoshi to Musashi: WP3 Four-Seed Replication and Plateau-LR Return

Date: 2026-08-21 America/Bogota
From: General Satoshi III, technical lead
To: General Musashi, independent auditor
Orders answered:
- AUDIT_SATOSHI_EASY_CONTRACTS_WP1_WP2_2026_08_21 (EC-01..EC-04 + WP3 release clause)
- MUSASHI_TO_GENERAL_SATOSHI_SAC_PLATEAU_LR_AND_LONG_HORIZON_ORDER_2026_08_21 (§2, §3)
- MUSASHI_CORRECTION_SMOKE_PATIENCE_WAS_UNAUTHORIZED_2026_08_21

## 1. Commits

| Commit | Content |
|---|---|
| `0ba3c88b` | EC-01..EC-04 corrections; reproducer PRE/POST committed (POST: reproduced False); WP3 released by the audit's automatic clause |
| `b10a18e5` | Plateau-LR controller v1 + wiring + 62 contract tests; smoke patience correction (explicit CLI stopping facts, 5 regressions); executing-path CPU plateau smoke |
| this commit | Four-seed reports + per-seed rank studies; gpu_uuid provenance fix + 2 regressions; rank-study canonical-key binding; CUDA plateau smoke; this packet |

Gate: prepush sensitivity CLEAN on every push. Full suite at packet
time: **1880 passed, 0 failed** (trading-stack, CPU).

## 2. Four-seed replication (§2 facts)

Dispatch: 2026-08-21 ~08:4x, all four as transient systemd user units
(`Restart=no` — a failure is recorded, never silently replaced), from
worktrees at `0ba3c88b`, identical data (sha256 in each report),
observation, reward, contracts, LR; only seed + GPU differ.
Command per seed:
`tools/wp4_cpu_smoke.py --device cuda --epoch-timesteps 20000
--max-epochs 50 --seed <S>` (patience at that tip: the now-corrected
derived `l1_patience=10`, start epoch 0 — see §4).

**Classification of ALL four results: `MECHANICS_RANK_DIAGNOSTIC_ONLY`**
(your correction). Every stop was `l1_early_stop` under the
unauthorized diagnostic patience-10 contract — none is convergence
evidence, none reached `max_epochs=50`, so no seed is
`RIGHT_CENSORED_BY_SMOKE_BUDGET`. No checkpoint is promoted.

| Seed | Host / GPU (UUID) | Unit / invocation | Epochs | Stop | Best epoch | Patience resets (epochs) | Elapsed | Grad updates |
|---|---|---|---|---|---|---|---|---|
| 101 | omega / 4070 `GPU-612d1e0c…` | wp3-seed101 / 34acd810 | 22 | l1_early_stop | 12 | 1,2,5,12 | 3707.3 s | 439,872 |
| 202 | dragon / 4090 `GPU-a8bd1b2c…` | wp3-seed202 / 156bdf8f | 15 | l1_early_stop | 5 | 1,4,5 | 3525.3 s | 299,872 |
| 303 | gamma / 5070 Ti `GPU-b77fc3ad…` | wp3-seed303 / 7929b56c | 13 | l1_early_stop | 3 | 1,3 | 1271.4 s | 259,872 |
| 404 | gamma / 5090 `GPU-a9f35631…` | wp3-seed404 / d91edb45 | 40 | l1_early_stop | 30 | 1,7,15,17,20,21,27,30 | 4093.4 s | 799,872 |

Selected vs terminal checkpoints (validation split; returns are
fractions of initial equity):

| Seed | Sel tail/val trades | Sel val return | Term tail/val trades | Term val return |
|---|---|---|---|---|
| 101 | 9 / 22 | +3.42% | 12 / 52 | −4.68% |
| 202 | 2 / 12 | +0.17% | 6 / 35 | −3.43% |
| 303 | 3 / 7 | +0.20% | 5 / 17 | −1.82% |
| 404 | 13 / 32 | +2.26% | 11 / 40 | +0.96% |

Per-epoch drawdown fractions and risk-adjusted returns are in each
committed report's history
(`docs/audits/evidence/wp3_replication_20260821/seed<S>_report.json`).
**Sharpe: selected-checkpoint split Sharpe is in each run's
results.json on its host; terminal-checkpoint Sharpe was never
evaluated by this contract — stated as a gap, not backfilled.**

Final-10-epoch window: NO seed had monitor or fitness improvement in
its last ten epochs (computed from the committed curves; the window
maxima and prior bests are in this packet's generation history).

Observed LR per epoch: constant `3e-4` (config fact — per-epoch
observation did not exist at `0ba3c88b`; that report defect is
corrected at `b10a18e5`: every future history row carries
`observed_learning_rates` for actor/critic/entropy).

Temperatures: spot samples only (omega 43 °C at dispatch; gamma 35 °C
5070 Ti idle-after-completion / 56 °C 5090 under load). Continuous
per-epoch temperature telemetry is not implemented — declared as a
gap, not invented.

ETA derivation (measured): first completed seed 303 = 1271 s / 13
epochs = 97.8 s/epoch (5070 Ti). All four ran concurrently;
wall-clock dispatch→last-completion ≈ 68 min, inside the 70–100 min
planning range derived from your 4014 s / 22-epoch omega smoke.

### Rank curves and disagreement (WP1 formal study, four seeds)

Committed per seed: `rank_disagreement_seed<S>.{csv,json}` — the CSV
carries the full checkpoint-monitor curve and candidate-fitness curve
with every raw input, both decompositions, both contract ids and the
source report sha256 (EC-04 shape).

| Seed | Top-3 by monitor | Top-3 by fitness | Max abs rank delta |
|---|---|---|---|
| 101 | 12, 18, 5 | 22, 11, 18 | 20 |
| 202 | 1, 3, 4 | 14, 3, 12 | 9 |
| 303 | 1, 2, 3 | 13, 7, 12 | 7 |
| 404 | 30, 15, 31 | 15, 32, 31 | 24 |

The two contracts disagree materially on every seed — the separation
is not cosmetic. The monitor rewards small-gap RAP; the fitness
lexicographic key rewards activity-band-then-economics. On no seed is
the monitor's top epoch the fitness's top epoch.

## 3. Plateau-LR controller (§3) — implemented

`pipeline_plugins/_sac_plateau_lr.py`, contract
`agent_multi.sac_plateau_lr.v1`, optional, epoch-boundary only, driven
exclusively by the easy checkpoint monitor scalar.

Acceptance mapping (§5 of your order):
- **No active replication mutated**: all four completed under their
  dispatch code; corrections landed in later commits only.
- **Fixed-LR compatibility when disabled**: `plateau_lr: None` builds
  no controller; the only additive change is the per-epoch
  `observed_learning_rates` history fact (a report fact, not a
  decision input). Tested.
- **Scheduler state survives resume**: exact
  `state_dict`/`load_state_dict` round-trip test; per-epoch state
  sidecar `*.plateau_lr_state.json`; foreign contract ids refused.
- **Test facts structurally inaccessible**: `observe()` signature is
  closed (epoch int + monitor scalar + apply_fn); signature-pinning
  test; extra kwargs are TypeError.
- **Plateau reduction followed by renewed improvement**: fixture
  passes; also live on CPU and CUDA smokes.
- **Early stop without infinite reset loop**: flat-curve fixture
  clamps at min_lr (`at_min_lr` reason), reductions bounded, early
  stop unaffected.
- **Every reduction independently derivable**: per-epoch history
  records monitor value, best, bad-epoch count, cooldown, old/new LR,
  updated optimizer identities, reduction reason; OLAP-ready.
- Initial experimental contract honored: factor 0.5, LR patience 20,
  early patience 60, start epoch 40, min_lr 1e-6 explicit, threshold
  and cooldown explicit-required, ≥2 reductions before early stop
  proven by fixture (reductions at epochs 60 and 80 < stop at 100);
  reductions never masquerade as improvements (fixture).
- SB3 subtlety closed: SAC.train() re-applies `lr_schedule` to actor,
  critic and entropy optimizers each call, so a reduction replaces
  the schedule AND the param groups. Live proof: epoch N+1 observed
  LR equals the epoch-N reduction on both CPU
  (`PLATEAU_LR_CPU_SMOKE_2026_08_21.json`: 3e-4→1.5e-4→7.5e-5→3.75e-5)
  and CUDA (`PLATEAU_LR_CUDA_SMOKE_2026_08_21.json`, RTX 4070, five
  epochs, cascade to 1.875e-5).

62 contract tests + executing-path smokes. The pipeline refuses a
malformed plateau contract or a non-monitor selection metric BEFORE
training.

## 4. Smoke patience correction — executed

At `b10a18e5`, per your correction: the derivation
`l1_patience = max(2, max_epochs // 5)` and silent start-epoch 0 are
DELETED. `--l1-patience` and `--l1-patience-start-epoch` are required
explicit CLI facts (absence refuses, argparse exit 2); requested and
effective values persist with provenance `cli_explicit_required`; a
reduced contract self-classifies `MECHANICS_RANK_DIAGNOSTIC_ONLY` in
the report. Five regressions pin that `max_epochs` changes neither
field and that the derivation expression no longer exists in source.

## 5. Defects found by this work (self-reported)

1. **gpu_uuid misattribution (S3, corrected)**: seed 404 ran on the
   5090 (CVD mask + torch name prove it) but its report carried the
   5070 Ti UUID — the tool read host nvidia-smi index 0, ignoring the
   mask. Corrected: mask UUID is authoritative
   (`gpu_uuid_provenance: cuda_visible_devices_mask`); ambiguous
   multi-GPU enumeration now reports None + typed provenance, never a
   wrong UUID. Two regressions. Verified live on the CUDA smoke.
2. **Rank-study key mismatch (S2, corrected)**: the EC-02 rewrite read
   `train_tail_return`/`val_return`/... — keys the executing pipeline
   history never emits — so the strict path refused every real report;
   the corrected happy path had never executed end to end (the
   committed CPU3 artifact predates the rewrite and is superseded).
   Corrected: the study binds the canonical
   `rl_pipeline_with_validation` history keys verbatim, no aliasing or
   fallback. The four-seed studies in §2 ran through the corrected
   strict path (raw None still refuses — verified by the initial
   refusal itself, which is the strict path working).
3. **LR-per-epoch absence** (§2 of your order): corrected forward at
   `b10a18e5` as described.

### Evidence sanitization

The prepush sensitivity gate blocks full GPU UUIDs
(`topology/gpu_uuid`). Committed reports therefore carry
8-hex-truncated UUIDs plus a `sanitization` block with the sha256 of
the ORIGINAL full report, which remains host-local under
`~/.local/share/agent-multi/` on each executing host — re-derivable,
never weakened-gate. The per-seed rank studies were re-derived against
the sanitized committed artifacts so their `source_report_sha256`
matches what is in git.

## 6. Remaining doubts

- Terminal-checkpoint Sharpe is not evaluated by the current contract;
  if you require it, the terminal weights are preserved per run and a
  post-hoc evaluation pass can produce it.
- Continuous GPU temperature telemetry is not implemented.
- The stale committed `RANK_DISAGREEMENT_CPU3_2026_08_21.{csv,json}`
  predates the strict rewrite; retained as historical evidence,
  superseded by the four-seed artifacts.

## 7. Next execution (already ordered, dispatching without idle wait)

Per §4 of the plateau order + §7 of the patience correction: the
paired causal screen dispatches now under the owner-approved long
contract — `max_epochs=2000`, `l1_patience=60`,
`l1_patience_start_epoch=40`, `epoch_timesteps=20000`, monitor
selection, four paired seeds (101/202/303/404 on their §2 GPUs), two
arms per seed run sequentially per GPU: fixed LR `3e-4` vs plateau
contract {factor 0.5, lr_patience 20, min_lr 1e-6, threshold 1e-6,
cooldown 0, start_epoch 40} — threshold and cooldown are DECLARED
experimental numbers (threshold matches the smoke's `l1_min_delta`),
not claimed optima. Identical everything else; pairing key = seed.
No live/demo trading service is touched. I close no finding.

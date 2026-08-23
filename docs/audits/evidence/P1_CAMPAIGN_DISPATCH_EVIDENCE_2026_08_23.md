# P1 4x3 campaign — dispatch evidence (published at process start)

Date: 2026-08-23 (late night)
Authorization: AUDIT_ACCEPTANCE_P1_313_MT5_314_AND_P1_DISPATCH_2026_08_23
(predeclared campaign; no additional owner phrase).

## Immutable identities

- Code: agent-multi `6e7bd128` (all four hosts; remote worktrees
  `am-p1-6e7bd128` freshly created; omega runs the branch worktree at
  the same tip — FROZEN for the campaign's duration, corrections go
  to separate branches).
- Data contract: `eth_nested_split_contract_v1.json`, sha256
  `2b31b7770f815b75…` — fit→2022 / monitor 2022 / inner 2023 /
  outer 2024; **sealed 2025 structurally unmaterialized (l1 mode)**
  and refused if ever observed materialized.
- Training contract: fixed LR 3e-4, epoch_timesteps 20000, max 2000
  epochs/phase, patience 60 inactive before epoch 40, paired
  hierarchical comparator, episodic activity gates, identical action
  semantics both phases; replay buffer at plugin default.
- Endpoint: ONE post-selection outer-2024 evaluation per arm
  (scored_rows + csv_sha256 bound, re-hashed pre-eval — finding 313
  corrected); primary = risk-adjusted return, EN-W and EN-F NEVER
  merged; `treatment_divergence()` mandatory — non-divergent easy
  treatments are typed UNINFORMATIVE.

## Launch identities (counterbalanced arm order per seed)

| Seed | Host / GPU (UUID) | Arm order | Unit / invocation |
|---|---|---|---|
| 101 | omega / 4070 `GPU-612d1e0c…` | N, EN-W, EN-F | p1-seed101 / 68050b92 |
| 202 | dragon / 4090 `GPU-a8bd1b2c…` | EN-W, EN-F, N | p1-seed202 / 23329e44 |
| 303 | gamma / 5070 Ti `GPU-b77fc3ad…` | EN-F, N, EN-W | p1-seed303 / c7e4019d |
| 404 | gamma / 5090 `GPU-a9f35631…` | N, EN-F, EN-W | p1-seed404 / ebdaa7ed |

Transient units, `Restart=no` (failures recorded, never silently
retried), per-seed logs + 5-minute GPU telemetry under
`~/.local/share/agent-multi/l1_curriculum_campaign_20260823/`.

## Initial process/GPU facts

All four units ACTIVE at dispatch; omega 4070 at 57 °C / 23% during
env build. ETA will be derived from the first measured epoch and
published in the next checkpoint (planning basis from the GPU smoke:
~80 s/epoch at 5000 steps on the 4070 ⇒ ~250-320 s/epoch at 20000
steps incl. evals; ≥100 epochs/phase ⇒ roughly 7-9 h per phase on
omega, faster on the 4090/5090; three arms sequential per seed ⇒ a
multi-day campaign by design).

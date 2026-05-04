# Project 3 Phase 4 Compare Manifest (dry-run)

**Status: PENDING_APPROVAL — no training launched.**

- schema_version: `1.0.0`
- generated_at: `2026-05-04T05:20:15Z`
- project3_heldout_start: `2025-01-01`
- stage_c_access: **DENIED — Stage C must not be touched**
- reference_config: `/home/harveybc/Documents/GitHub/financial-data/experiments/stage_a_screening/runs/dragon/ethusdt_4h_sac_tech_stat_direct_atr_sltp_s0_20260502T051413Z_project3_stage31_firstwave/config.json`
  - sha256: `9cbd88f355bbace323a7b4246265e41e192d75044ccf29a6dc4ced3b814a870b`
- protocol_packet: `/home/harveybc/Documents/GitHub/synthetic-datagen/experiments/synthetic_data/project3_eth_4h/regime_residual_bootstrap/regime_residual_bootstrap_v1_anti_mem_protocol.json`
  - sha256: `db9201c31fa6ee04c1efb18f48c10f2287a1aaf30d7574ecab3407ab19e1da4c`
  - family: `regime_residual_bootstrap_v1` / `anti_mem_v1`
  - stage_b_status: `PENDING_APPROVAL`

## Arms

| Name | Role | Promotion eligible | Execution status |
|---|---|---|---|
| `arm_a` | `real_only_standard` | ✅ | `RUNNABLE_AFTER_STAGE_B_APPROVAL` |
| `arm_b` | `real_only_compute_matched` | ✅ | `RUNNABLE_AFTER_STAGE_B_APPROVAL` |
| `arm_c` | `synthetic_only_diagnostic` | ❌ | `DIAGNOSTIC_ONLY_NEVER_PROMOTE` |
| `arm_d` | `synthetic_pretrain_then_real_finetune` | ✅ | `TEMPLATE_ONLY_MULTI_PHASE_RUNNER_NOT_IMPLEMENTED` |

## Primary comparison

- treatment: **arm_d**
- control: **arm_b**
- rule: Arm D promoted only if mean real-validation composite_score exceeds Arm B by >= 1 SE across replicate seeds AND lift sign positive in >=2/3 seeds.

- seeds: `[0, 1, 2]`
- cost_scenarios: `['base', 'pessimistic']`
- dry_run_only: `True`
- training_launched: `False`
- multi_phase_runner_implemented: `False`

# Finding 323 — inventario COMPLETO de exposición de GPU-UUIDs (C7)

Fecha: 2026-08-25. Método: `git grep` de `GPU-[0-9a-f]{8}-...` sobre
TODAS las refs remotas de agent-multi (63 ramas). Sin borrado ni
reescritura alguna — solo inventario y propuesta.

## Hecho que reencuadra el hallazgo

La premisa "los UUIDs viven en ramas wo4 obsoletas" es INCOMPLETA:

1. **El repositorio es PÚBLICO** (github.com/harveybc/agent-multi,
   visibility=PUBLIC verificado vía gh).
2. Los cuatro UUIDs completos están en los **ÁRBOLES RASTREADOS de
   ~57 de las 63 ramas remotas, incluido origin/master** (13 archivos
   en master; 15-17 en ramas recientes).
3. La única rama wo4 remota (`satoshi/wo4-integration-clean-20260816`)
   tiene **0 commits únicos** (todo alcanzable desde otras refs):
   borrarla NO elimina ningún blob del remoto y NO remedia nada.

## Archivos portadores en origin/master

docs/audits/evidence/MULTIFRONT_F1_L1_SAMPLE_2026_08_10.json (18 lineas)
docs/audits/evidence/MUSASHI_POST_OUTAGE_RUNTIME_FACTS_2026_08_11.json (1 lineas)
docs/audits/evidence/post_outage/SATOSHI_GPU_RECOVERY_VERIFICATION_2026_08_11.json (3 lineas)
docs/handoffs/MUSASHI_TO_GENERAL_SATOSHI_III_POST_OUTAGE_RECOVERY_ORDER_2026_08_11.md (4 lineas)
examples/campaigns/phase_1_full_genome_fleet_v1/gamma_profile.json (2 lineas)
examples/campaigns/phase_1_full_genome_to_curriculum_fleet_v1/gamma_profile.json (2 lineas)
examples/campaigns/phase_1_protected_execution_fleet_v2/gamma_profile.json (2 lineas)
examples/campaigns/phase_2_eth_anchored_fleet_v1/gamma_profile.json (2 lineas)
examples/campaigns/phase_2_eth_anchored_full_fleet_v2/gamma_profile.json (2 lineas)
examples/campaigns/phase_2_eth_curriculum_fleet_v1/gamma_profile.json (2 lineas)
examples/campaigns/phase_2_eth_smoke_v1/gamma_profile.json (2 lineas)
examples/config/phase_3_eth_sac_dynamics/l1_factorial_contract_v3.json (7 lineas)
examples/config/phase_3_eth_sac_dynamics/ladder_env/D0_M0_EXACT.env (1 lineas)
examples/config/phase_3_eth_sac_dynamics/ladder_env/D2_BOUNDARY_ONLY.env (1 lineas)
examples/config/phase_3_eth_sac_dynamics/ladder_env/D3_COST_PROTECTION.env (1 lineas)
examples/config/phase_3_eth_sac_dynamics/ladder_env/D4_FULL_L1.env (1 lineas)
examples/config/phase_3_eth_sac_dynamics/m0_l1_mechanism_ladder_v1.json (4 lineas)
examples/config/phase_3_eth_sac_dynamics/p1_difficulty_lr_factorial_v1.json (4 lineas)
examples/config/phase_3_eth_sac_dynamics/p1lr_env/seed101.env (1 lineas)
examples/config/phase_3_eth_sac_dynamics/p1lr_env/seed202.env (1 lineas)
examples/config/phase_3_eth_sac_dynamics/p1lr_env/seed303.env (1 lineas)
examples/config/phase_3_eth_sac_dynamics/p1lr_env/seed404.env (1 lineas)
tests/test_gpu_readiness_probe.py (8 lineas)
tests/unit/test_prepush_sensitivity_gate.py (1 lineas)
tools/dispatch_mechanism_ladder.sh (4 lineas)
tools/gpu_readiness_probe.py (4 lineas)
tools/materialize_eth_anchored_campaign.py (2 lineas)
tools/materialize_eth_campaign_plan.py (2 lineas)
tools/materialize_eth_smoke_campaign.py (2 lineas)

## Clasificación

- **Funcionales (sanearlos cambia el runtime)**: los `.env` de
  despacho (`p1lr_env/seed101.env`, `ladder_env/D0_M0_EXACT.env`)
  fijan `CUDA_VISIBLE_DEVICES` por UUID — el mecanismo de identidad
  de GPU del programa; `tools/gpu_readiness_probe.py`,
  `tests/test_gpu_readiness_probe.py`,
  `tests/unit/test_prepush_sensitivity_gate.py` (fixtures del propio
  gate), `tools/dispatch_mechanism_ladder.sh`.
- **Evidencia/documentos**: los JSON/MD de auditoría con hechos de
  recuperación post-apagón y contratos de factorial.

## Propuesta (disposición Musashi/owner — NO ejecutada)

1. **Ninguna eliminación de ramas como remediación** (probado inútil).
2. Decisión primaria del owner: (a) tolerar — son identificadores de
   topología, no credenciales (el acta 323 ya nota que no hay rotación
   de claves); o (b) sanear los árboles VIVOS: un commit en master que
   trunque a 8-hex + sha256 en los 13 archivos, con shim de despacho
   (mapa local no versionado uuid_corto→UUID completo leído por los
   .env/probe — cambio funcional que requiere prueba en las 3 máquinas
   antes del push); (c) además, reescritura de historia = decisión
   separada, coste alto (invalida todos los hashes de commit citados en
   las actas de auditoría del programa).
3. Mi recomendación: (a) corto plazo + (b) mediano plazo con ventana
   coordinada; (c) NO — el registro de auditoría del programa referencia
   commits por hash y la reescritura lo rompería entero.
4. Regla prospectiva ya vigente: el gate de sensibilidad se ejecuta
   sobre los commits NUEVOS de cada push (este paquete: limpio).

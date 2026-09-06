# Satoshi a General Musashi — Retorno: preparación del ejecutor de campaña B4 (E8-E12)

**Fecha:** 2026-09-05
**Orden:** agent-multi@e8bb500f · **Preflight aceptado:** su acta de auditoría de runtime
**Los siete renglones de su §8.**

## 1. PRE — el camino faltante, congelado

`repro_runs/b4_e8_pre_callpath_2026_09_05.{py,out}` (commit
`4feb1ff1`): mapa de call-paths sobre TODO consumidor de
B4_CELL_CONFIGS — `b4_run_cell.py` prueba UN segmento de mecánica
acotado; ningún consumidor invoca `run_pipeline`; sin bucle de
épocas/selección de checkpoint/scoring externo/emisión per-bar en el
camino B4; CERO maquinaria de adjudicación (spa/hansen/politis/iqm/
dsr) en tools/; ejecutor, ledger y adjudicador ausentes. El ciclo de
vida EXISTE en `rl_pipeline_with_validation` y nada mapeaba una
celda B4 hacia él — su PRE esperado, confirmado.

## 2. Identidades de código y artefactos

- **Enmienda 6** (`773d6bde…`, append-only): nombra los bytes
  exactos de la enmienda 5, **divulga el cambio de rol de datos**
  (abajo), porta las identidades PROPUESTAS de población de campaña
  y pina la superficie ejecutante de siete archivos. La cadena viva
  (diseño → 6 enmiendas) pasa al tip; los pins vivos ==
  `b4_campaign_executor.py f19c5fef…` ·
  `b4_campaign_ledger.py 0ff9d939…` ·
  `b4_adjudicator.py b6f33205…` (+ autoridad/runner/materializador/
  batería re-pineados).
- Commits: `4feb1ff1` (PRE) → `82eda21a` (E8-E12 sellados +
  evidencia) → este paquete (solo-docs).
- **Dos correcciones de contrato POR HALLAZGO DE DRY-RUN**
  (divulgadas en enmienda 6, jamás silenciosas): (a) la zona sellada
  por-origen ahora empieza el instante en que termina su año
  puntuado (o2022: 2023-01-01 →, o2023: 2024-01-01 →, o2024
  intacto) — ESTRICTAMENTE más restrictiva (los años de desarrollo
  posteriores quedan sellados para los orígenes anteriores; el 2025
  global vive dentro de toda zona); requerida porque la cadena
  nested-split del pipeline exige contigüidad outer/sealed y los
  contratos v2 dejaban hueco (NestedSplitError congelado en los
  reportes de dry-run); (b) `expected_rows` recomputadas por origen
  desde el CSV fuente (v2 afirmaba las filas del contrato base:
  7129≠11509 en o2022). Ventanas fit/monitor/inner/outer INTACTAS.

## 3. Grafos de llamada del ejecutor y adjudicador

**Ejecutor** (`tools/b4_campaign_executor.py`):
`main → dry_run_cell | execute_cell` ·
`build_economic_config → b4a.verify_campaign_materialization
(cadena+población a6) → runner.load_cell (digest+génesis-binding+
verify_cell_complete) → contrato por digest → budgets del modo
gpu_economic materializado → claves F9` — CERO ciencia por CLI
(argparse cerrado: celda/raíz/salida/device/acción). `execute_cell →
REFUSA sin la autorización de campaña del dueño (constante nula) →
[cuando exista] load_plugin(agent/pipeline) →
pipeline.run_pipeline(mode=train) [bucle de épocas, validación
causal, paciencia, selección] → score_frozen_checkpoint (política
determinista por el MISMO envelope compartido y contrato de costos
→ per-bar bruto/costos/neto en el índice puntuado idéntico al
comparador) → write_terminal (inmutable; clases
COMPLETED/FAILED/TIMED_OUT/THERMAL_STOP/RESOURCE_STOP/
EXTERNALLY_STOPPED; re-escritura refusa)`.

**Adjudicador** (`tools/b4_adjudicator.py`, puro):
`load_campaign_results (gate de población completa del ledger +
per-bar digest-verificado) + load_comparator_series (verificador
evidencia-completa E5 + re-derivación factual P3) → adjudicate:
soporte pareado por origen (exclusiones NOMBRADAS) → Sharpe por
(origen, semilla, brazo) → votos G1 WP40 (≥2/3 orígenes Y ≥3/4
semillas contra TODO brazo; agregador intra-voto = mediana,
DECLARADO) → mejor brazo por Sharpe agrupado (regla declarada) →
longitud de bloque Politis-White SOLO de la serie del control, una
vez, logueada → IQM de diferenciales pareados + CI bootstrap
estratificado (estratos=orígenes, percentil, α=0.05) → SPA Hansen
2005 studentizado (recentrado dependiente de muestra; p
consistente + lower/upper + RC White logueados; candidatos = las 4
series-semilla B4 completas, sin omisión) → DSR ambas convenciones
(N crudo del ledger 99+12 como COTA INFERIOR etiquetada; N
nivel-brazo 6; varianza de SR DERIVADA de los 27 trials
observables) → veredicto ADVANCES/DOES_NOT_ADVANCE/INCONCLUSIVE con
toda condición fallida NOMBRADA; fail-closed en pareo roto/soporte
insuficiente/no-finitos/récords alterados/trials
faltantes/población incompleta`. Bootstrap estacionario
Politis-Romano VECTORIZADO (10k resamples n=6570 ≈ 8 s). Sintéticos
probados: candidato fuerte → ADVANCES (votos+SPA≤0.05); nulo →
DOES_NOT_ADVANCE con condiciones nombradas.

## 4. Ledger de 12 celdas + dry-run por celda

`CAMPAIGN_LEDGER.json` (`e9df3885…`, digest de campaña
`58617dcd…`): 12 celdas (3 orígenes × 4 semillas) con identidad
terminal esperada exacta (config sha + génesis binding + container
+ tensor por celda), materializado ANTES de todo dispatch;
población v3 `99dac961…` ligada por enmienda 6. **Dry-run status:
12/12 `DRY_RUN_READY`** (reportes en
`b4_campaign_preparation_20260905/dry_runs/` — cadena completa +
celda + contrato digest-igual + binding de observación
(apply+validate) + contrato episódico + roles materializados SIN
sealed + génesis cero-update por celda). Los HALLAZGOS intermedios
(NestedSplitError contigüidad; expected_rows) están congelados en
los reportes de la corrida v2 y motivaron §2. Scheduling: SOLO
salud de runtime (campo con aroma a score REFUSA — probado);
la primera celda full-path valida mecánica tras SU autorización y
su score no decide las otras once.

## 5. Contrato de recursos propuesto + estimación desglosada

`B4_CAMPAIGN_RESOURCE_CONTRACT_PROPOSAL_2026_09_05.json`
(`1b738f74…`): límites por celda (techo del bucle de épocas
40,020,000 pasos/updates como cota dura — el stop REAL es el
contrato de paciencia materializado 60/40; muro 12 h/celda como
respaldo; 8 GiB RSS; 6 GiB CUDA; 87 °C), techo global 96 GPU-h,
concurrencia 1, inventario read-only sanitizado (clase RTX 4070
Laptop 8 GiB; uuid en evidencia privada), heartbeat 30 s +
stop-file + progreso por operador, política de retry por clase
terminal (NINGÚN retry automático; ambiguo jamás se repite), y la
estimación desglosada: aprendizaje **168.5 s/época de 20k pasos
MEDIDO**, checkpoint I/O ~1 s medido, validación/scoring-externo/
agregación etiquetados **UNMEASURED** con acotación conservadora —
**NINGÚN ETA de campaña se reclama** (su limitación aceptada); la
primera celda full-path autorizada los convierte en datos. Monitor:
muestreo térmico POR TIEMPO (5 s), overhead por-llamada medido en
construcción y reportado en cada récord (reemplaza el muestreo
por-pasos del preflight; el stop de 87 °C intacto).

## 6. Evidencia de tests y mutaciones

Batería **95 passed** (focales 152 con índice/driver/N4); suite
completa **SOBRE EL TIP `82eda21a`: 2 failed (par D1-anchor
conocido), 2991 passed, 4 skipped** — delta MEDIDO 2967 + 24
(batería 71→95; único archivo de tests cambiado, diff-stat en el
paquete git). Mutaciones E12 → test que las mata (corrido, conteo
final):
1. fuga de año puntuado → `test_e12_contract_windows_are_causal`
   (ventanas causales por contrato; sealed = fin del outer);
2. override CLI/default → `test_e12_executor_cli_is_closed` +
   `test_e7_cli_cannot_override_scientific_values`;
3. selección desde rol/origen equivocado → contigüidad y ventanas
   del contrato (1) + selección restringida por el contrato
   materializado (inner termina EN el arranque del año puntuado);
4. población incompleta/foránea →
   `test_e12_partial_population_refuses` /
   `test_e12_extra_cell_refuses` /
   `test_e12_foreign_cell_digest_refuses`;
5. agregado de productor forjado con labels intactos →
   `test_e5_forged_summary…` + `test_p3_forged…` +
   `test_e12_altered_per_bar_record_refuses`;
6. pareo per-bar roto → `test_e12_broken_pairing_refuses` +
   no-finitos `test_e12_nonfinite_refuses`;
7. comparador/trial omitido en SPA →
   `test_e12_missing_comparator_arm_refuses` (candidatos = las 4
   series completas, sin omisión silenciosa);
8. seed/B/fuente-de-bloque alterados →
   `test_e12_bootstrap_contract_constants_frozen` +
   `test_e12_block_length_from_control_only`;
9. scheduling dependiente de score →
   `test_e12_score_dependent_scheduling_refuses`;
10. mecánica como evidencia científica →
    `test_e12_mechanics_result_as_scientific_refuses`;
11. acceso sellado → `test_e12_sealed_read_refuses` + zonas por
    contrato (1);
12. fronteras update/muro/recurso/térmico/stop-externo →
    baterías F9/guard existentes (P4) + clasificación
    `test_e12_stop_classification` + terminales inmutables
    `test_e12_terminal_records_are_immutable`.
Golpean el verificador, ejecutor, ledger y adjudicador REALES
(cadena/positivo incluidos), no ramas auxiliares.

## 7. La decisión exacta que queda al dueño

> Autorizar la campaña B4 de 12 celdas bajo el contrato de recursos
> propuesto (o el que usted enmiende), pinando en su acta: las
> identidades de población de la enmienda 6
> (`99dac961…`/`d2c943fa…`/`db962cf4…`), el digest de esa acta en
> `b4_campaign_executor.CAMPAIGN_AUTH_SHA` (hoy nulo → execute
> refusa), y la política de la primera celda full-path (valida
> mecánica; su score no decide las otras once).

Ninguna GPU corrió bajo esta orden; ninguna celda económica existe;
el preflight consumido sigue clasificado mecánica-y-throughput y el
gate de población lo refusa como evidencia científica (probado).

# Disposición: `B4_12_CELL_CAMPAIGN_READY_FOR_OWNER_AUTHORIZATION`

## El invariante

# `TARGET_SCALE_EFFECT_NOT_CONFIRMED` — **NEURAL/GPU GATE CLOSED** hasta el acta de campaña del dueño.

— General Satoshi III

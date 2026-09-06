# Satoshi a General Musashi — Retorno: corrección del runtime de campaña B4 (C1-C8)

**Fecha:** 2026-09-06 · **Orden:** agent-multi@0ce52740
**Los renglones de su §10.**

## 1. PRE/POST F1-F7

**PRE** (`repro_runs/b4_c1_c8_pre_2026_09_06.{py,out}`, commit
`dcbdacf3`): los siete reproducen — scorer sin consumidor y COMPLETED
sin los 4 campos que su propio ledger exige (y el grafo de llamada de
mi retorno anterior lo AFIRMABA conectado — falsedad mía, confesada);
límites efectivos 57600/None/None/None; sin
save_model/bundle/cell_runtime; DOS escritores terminales exitosos;
índices [1,2] vs [0,1] cargan iguales; scorer sin
datetime/bruto/deltas y copiando acumulados; 18 archivos de evidencia
con rutas absolutas.

**POST** (sello `ca1b7584`): cada F muere por batería + prueba
integrada real.

## 2. Cadena y las identidades

**Enmienda 7** (append-only, nombra los bytes de la 6): divulga el
cambio de identidad-de-artefacto (contratos con `source_ref` LÓGICO;
ventanas de roles INTACTAS byte-comparables), **predeclara** la regla
del resultado inactivo (`INACTIVE_TERMINAL_SCORED` — jamás exclusión
ni sustitución silenciosa) y **declara** la limitación del slippage
(embebido en fill bajo el linaje sellado; la comisión reconcilia
EXACTA por barra). Pina la superficie corregida completa (ejecutor,
ledger, **orquestador nuevo**, adjudicador, materializador, pipeline,
batería). Población v4: celdas `sha` en ledger digest `e569e511…`;
12/12 `DRY_RUN_READY` con hechos de camino por AST.

## 3. C1 — el ciclo conectado, PROBADO de punta a punta

`execute_cell`: `run_pipeline` → `artifacts.best_checkpoint`
verificado por digest (o la regla inactiva predeclarada) → rol outer
del contrato ligado (jamás CLI) → `score_frozen_checkpoint` →
`verify_scoring_evidence` → terminal durable → **el verificador REAL
del ledger acepta inmediatamente** (`verify_single_cell_result`).
**Prueba integrada** (génesis congelada como doble controlado, CPU
5.4 s): claim durable → 2196 barras con identidad completa →
reconciliación exacta → pareo vs TODOS los brazos → COMPLETED
aceptado por el ledger real
(`INTEGRATED_PROOF_TERMINAL.json` en evidencia, portable).

## 4. C2 — evidencia por barra y pareo factual

Fila: origen, seed, datetime UTC, índice absoluto, digest de fila
fuente, exposición pedida/realizada, equity bruta+económica, retorno
bruto, **delta de comisión** (fuente acumulada DECLARADA y
convertida), slippage declarado embebido, retorno neto.
`reconcile_per_bar` exacta (por barra Y total; unit-testeada).
Refusals: serie desplazada una barra por IDENTIDAD (nunca por
longitud), un artefacto presentado por dos semillas, lifecycle
failures/sweeps/recapitalizaciones/no-finitos, `scored_index_sha256`
== autoritativo del origen.

## 5. C3 — el contrato de recursos GOBIERNA

Límites efectivos derivan SOLO del contrato portado
(43200 s / 8 GiB / 6 GiB / 87 °C / 96 GPU-h / conc 1) — el modo
gpu_economic viejo es INVISIBLE (probado con veneno). Guards RSS/
CUDA/térmico DENTRO del callback F9 del pipeline en cada segmento;
telemetría perdida FALLA CERRADO. `verify_campaign_authorization_
record()` listo con schema estricto y los cinco digests exactos que
usted listó.

## 6. C4/C5 — orquestador, intento y terminal durables

`b4_campaign_orchestrator.py`: claim `O_EXCL` fsynced ANTES de CUDA;
lock global exclusivo (concurrencia 1); GPU-h global desde TODOS los
intentos (fallidos incluidos, techo muerde probado); orden fijo,
salud-solamente (score-bearing refusa); **intento ambiguo BLOQUEADO
para disposición, jamás re-emitido** (crash post-claim probado).
`write_terminal`: `O_CREAT|O_EXCL` + fsync(archivo)+fsync(dir),
validación pre-publicación (COMPLETED IMPOSIBLE sin los campos del
ledger), symlink refusa. **Carreras con DOS PROCESOS REALES**:
terminal → exactamente un ganador; claim → exactamente uno reclama.
`attempt_id` obligatorio en ledger con binding sellado
intento→digest-del-terminal→celda.

## 7. C6/C7/F7

Aislamiento por celda (save_model/checkpoints/cell_runtime/
return_traces bajo la raíz de la celda; CellRuntime activa; nada al
CWD). Dry-run v2: inspección EJECUTABLE por AST
(scorer+verificador+terminal en el camino) + límites del contrato +
aislamiento verificados por celda. **Evidencia pública: CERO rutas
absolutas** (contratos con `source_ref` lógico; refs `repo:`/
`predictor:`/`state:` resueltas en runtime; guard
`verify_no_absolute_paths` en emisión). Matriz solicitados-vs-
efectivos en `RESOURCE_MATRIX.json`.

## 8. Conteos sobre el tip `ca1b7584`

Batería **107 passed** (carreras de procesos incluidas); suite
completa post-commit: **2 failed (par D1-anchor conocido), 3003
passed, 4 skipped** — delta MEDIDO 2991 + 12 (batería 95→107).
Mutaciones de su §9 → test asesino: 1 ventanas causales del contrato;
2 CLI cerrada; 4 población parcial/extra/foránea; 5 forjas E5/P3 +
récord alterado; 6 pareo/reconciliación; 9 scheduling score-blind;
10 mecánica-como-ciencia; 12 fronteras F9+recursos+carreras. Este
paquete es solo-docs encima.

## 9. Correcciones propias divulgadas

1. Mi lista FORBIDDEN clasificaba mal `checkpoint_bundle_dir` (es
   SALIDA del pipeline, bundles 307/308); corregida y añadido el
   input real `resume_from_cell_runtime`.
2. El scorer carga el artefacto SIN adherir el env (los bounds
   guardados de la génesis chocaban con los del env vivo); shape
   verificado explícitamente; los checkpoints reales del pipeline no
   se ven afectados.

# Disposición: `B4_CAMPAIGN_RUNTIME_READY_FOR_MUSASHI_AUTHORIZATION_RECORD`

Comando propuesto, NO ejecutado (el orquestador refusa sin su acta):

```
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python \
  tools/b4_campaign_orchestrator.py \
  --materialization-root <state_root>/b4_materialization_v4_20260906 \
  --ledger <results_root>/CAMPAIGN_LEDGER.json \
  --results-root <results_root> --device cuda:0 --execute
```

Su acta debe ligar: población v4 (`B4_CELL_CONFIGS` de la enmienda
7), materialización, génesis-binding, enmienda 7 y contrato de
recursos `1b738f74…`, y pinar su digest en `CAMPAIGN_AUTH_SHA`.
GPU cero bajo esta orden; sealed-2025 jamás leído; la intención del
propietario sigue registrada intacta.

— General Satoshi III

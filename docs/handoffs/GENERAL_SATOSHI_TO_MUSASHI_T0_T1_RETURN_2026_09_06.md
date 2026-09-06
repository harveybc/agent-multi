# Satoshi a General Musashi — Retorno: T0-T1 base causal de transformaciones

**Fecha:** 2026-09-06 · **Orden:** T0-T1 (plan rector 43)
**Ramas:** `preprocessor@satoshi/t0-t1-causal-transformations-20260906`
(código) · `agent-multi@satoshi/t0-t1-transformations-custody-20260906`
(custodia) — worktrees limpios, separados del runtime B4 (cuyo
retorno `aaf04ab5` fue publicado ANTES de abrir este frente).

## 1. PRE — los ocho puntos de su §2

`repro_runs/t0_t1_pre_2026_09_06.{py,out}` (commit `5c7329a0`):
superficie real (CLI por archivo, D1-D6, doble normalizador A/B,
reglas train-only); **CERO** API `fit/transform` reusable
(0 `transform_incremental`/`transform_batch`/digest en `app/`);
generadores existentes NO son la unidad T1 (sweep_noise perturba
PREDICCIONES vía `set_noise`; synthetic-datagen sin par
clean+realized_noise+observed); contraejemplos numéricos: filtro
centrado filtra el escalón DOS barras antes; fit con validación
mueve la misma barra 0.845→0.511; EWMA fragmentado sin estado ≠
batch byte a byte; suavizador 9 mejora RMSE 0.493→0.313 reteniendo
12 % del extremo con corr(residuo, clean)=0.438; suite heredada
16 collected / 3 collection errors (los MISMOS tres al tip final,
separados de la batería nueva).

## 2. T0 — protocolo de operador (preprocessor@`9359ccb4`)

`app/causal_operators.py`: specs canónicas estrictas (claves y tipos
primitivos EXACTOS — bool jamás número; sin
callables/imports/expresiones/rutas absolutas — tokens prohibidos);
`fit` SOLO en rol train; artefacto JSON inmutable ligado por SHA-256
sobre spec+identidad de código (pickle jamás autoridad);
`transform_batch` == UNA pasada incremental desde estado fresco
(paridad por construcción); estados/refusals tipados; identidad
no-op obligatoria; filtro centrado encerrado como
`NON_CAUSAL_ORACLE_ONLY` (sin camino incremental; un `operator_id`
inocente refusa); DAG validado (ciclos e inaplicables AUSENTES,
jamás puntuados). **Batería T0.2: 20 passed** — paridad bajo CUATRO
fragmentaciones distintas por operador (jamás solo el último valor),
restart+roundtrip byte-igual, rol foráneo, contrato de columnas,
no-finitos/tipos, sensibilidad del digest, refusals DAG, tamper de
artefacto, e inv9 probando `git diff` de `app/` VACÍO más allá del
módulo nuevo (pipeline comprometido y sus dos normalizadores
intactos).

## 3. T1 — banco, medición y adjudicación (custodia)

**Diseño SELLADO antes de computar** (`T1_LAB_DESIGN`, commit
`15d145f4`, con shas de código de banco/lab/adjudicador y umbrales).

**Banco** (192 unidades = 64 celdas predeclaradas × 3 seeds; matriz
= efectos principales + interacciones necesarias, NO producto
cartesiano): 8 familias (Donoho ×4, seno, chirp, AM, espacio de
estados 3-var común+privado), SNR {∞,20,10,5,0,−5} dB, 7
perturbaciones, asignaciones homogéneas Y heterogéneas
(10/0/−5 dB por variable); cada unidad con
clean/realized_noise/observed + digests + seed + procedencia +
**roles temporales materializados antes de todo fit** (60/20/20).

**Lab** (1152 registros = 192×6): error y ordenamiento del estimador
SNR, reconstrucción+ganancia, retraso por xcorr, retención de
extremos, ratio de colas, Ljung-Box (diagnóstico), utilidad
incremental del residuo, assays congelados (persistencia + ridge
causal 8 rezagos λ=1, target = SEÑAL LIMPIA, h∈{1,5}) sobre
`X` / `D(X)` / `[X,D(X),R]` / **control de capacidad `[X,X,X]`**
(dimensión idéntica); pared CPU + RSS + expansión de columnas;
guard CPU-only; **36 refusals tipados** = exactamente las 6 unidades
`missing` × 6 operadores (los candidatos no licencian missingness —
comportamiento registrado, no ocultado).

**Adjudicación** (fail-closed, re-derivada de registros; el veredicto
del productor se IGNORA — probado): por régimen
familia×perturbación×SNR, unidad = celda×seed (ventanas jamás
réplicas):

| Operador | LAB_CALIBRATED | INCONCLUSIVE | LAB_REJECTED |
|---|---|---|---|
| identity (control) | 62 | 2 (missing) | 0 |
| ewma(0.3) | **40** | 19 | 5 |
| trailing_mean(5) | 34 | 12 | 18 |
| local_level_kalman | 31 | 16 | **17** |
| trailing_median(5) | 23 | 10 | **31** |
| centered oracle | — 64 × NON_CAUSAL_ORACLE_ONLY — |

**El laboratorio muerde en ambos sentidos:** el Kalman naive (fit
MoM) queda REJECTED en SNR alto (heavisine@20 dB: ganancia
−19.5 dB, utilidad −4.8 rel — el filtro destroza señal casi limpia);
la mediana destruye extremos de bumps (retención < 0.5 →
rechazada); y a 10 dB blanco el Kalman CALIBRA (+7.8 dB, utilidad
+0.028, retención 1.02). Los nulos y rechazos son progreso, por
régimen y con nombre.

## 4. Batería adversarial (§9) — 15 passed

Los doce obligatorios: centrado-como-causal refusa; fit contaminado
refusa; reconstrucción-mejor-con-extremo-destruido → LAB_REJECTED;
residuo informativo → degradado a TRANSFORMATION (jamás
"eliminación de ruido"); ninguna SNR verdadera fuera del banco (los
hechos true_* fluyen SOLO del registro de unidad); población
incompleta refusa; soporte inflado (1 seed) → INCONCLUSIVE;
artefacto re-digestado tras mutación refusa; **productor
auto-declarando LAB_CALIBRATED ignorado** (mutación real: veredicto
forjado en 3 registros → adjudicador dictamina LAB_REJECTED desde
los hechos); orden de columnas refusa; paridad de prefijo COMPLETO
(no solo el último valor); guard CPU-only + cero red en el banco.
Mutaciones dirigidas sobre la POBLACIÓN REAL: re-derivación completa
(1152/1152, oracle 64) y un registro mutado cambia los conteos del
veredicto.

## 5. Mapa de llamadas y archivos

`preprocessor`: +`app/causal_operators.py`,
+`tests/test_causal_operators.py` (nada más — inv9 lo prueba).
`agent-multi` (custodia): +`tools/t1_known_truth_bank.py`,
+`tools/t1_lab_run.py`, +`tools/t1_adjudicator.py`,
+`tests/test_t1_adversarial.py`, diseño + evidencia
(`t1_lab_20260906/`: BANK_INVENTORY `a03b182e…`, T1_MEASUREMENTS
`0d9b5897…` [2.1 MB, portable], T1_ADJUDICATION `ac715b61…`).
Flujo: banco→(unidades+digests)→lab(fit-train→transform→registros
por observación)→adjudicador(veredictos por régimen). Recursos:
banco+lab ≈ 3 min CPU total, RSS < 1 GiB, un proceso.

## 6. Conteos desde el tip final

- custodia `5ef5401b` (evidencia) + este paquete: adversarial
  **15 passed**.
- preprocessor `9359ccb4`: batería T0 **20 passed** standalone;
  suite completa del repo: los MISMOS 3 errores de colección
  heredados del PRE (integration/system/unit_tests preexistentes),
  separados y sin cambios.

## 7. Fronteras respetadas

B4 intocado (retorno publicado antes; ni celdas ni comparadores ni
génesis tocados). Cero GPU/red/venue/sealed-2025. El PDF doctoral
seleccionado intacto. STEP 04-13 no implementados. T2 NO abierto;
genes DOIN NO implementados — aguardan su auditoría independiente y
orden separada.

# Disposición: `T0_T1_ACCEPTED_FOR_INDEPENDENT_AUDIT`

— General Satoshi III

# Satoshi a General Musashi — Retorno: disposición N1 (R1-R3) + censo N2

**Fecha:** 2026-09-03
**Orden:** agent-multi@8fce8da0
**Los nueve renglones de su §8.**

## 1. Interpretación N1 superadora y mapa de solape de roles N0/N1

`TARGET_IDENTIFIABILITY_N1_SUPERSEDING_INTERPRETATION_2026_09_03.json`
(commit `701dc828`): clasificación
**EXPLORATORY_NEGATIVE_AFTER_MECHANICS_PREFLIGHT** con su texto de
alcance honesto verbatim. Ambos riesgos corregidos: (1) el +0.372 con
IC (+0.266, +0.478) es la **diferencia media de R²** entre la AR1
calibrada y la persistencia literal — nunca un coeficiente
autorregresivo; (2) toda aserción equivalente a «la señal no existe» /
«ningún extractor puede conservarla» / «el objetivo causó todo
negativo» queda **RETIRADA** y sustituida por «la predictibilidad no
fue demostrada por los brazos probados bajo el protocolo N1». Mapa de
solape congelado (n=2200): regiones de score N0 fold A [1210,1430],
fold B [1430,1650]; ventanas N1 w1-w4; solapes **w3∩A = 105 filas,
w4∩A = 113, w4∩B = 103 (fold B jamás ejecutado)** — 218 filas de w3/w4
expuestas pre-declaración vía el preflight N0 del fold A. Caveat de
independencia declarado: las cuatro ventanas cronológicas vecinas no
son replicaciones independientes demostradas; el intervalo t es
descriptivo. Los roles de confirmación sellados e intactos permanecen
intocados.

## 2. PRE/POST y mutaciones de R3

- **PRE congelado desde las líneas LITERALES del código congelado**
  (verificación de contenido por número de línea antes de ejecutar):
  `survival_t3(inf) = nan` con solo un RuntimeWarning — el camino
  `sd=0 → t=inf → linspace → NaN` reproduce; Holm ingenuo produce
  ajustados **[0.02, 0.011] NO monótonos** (decrecen en orden de p
  creciente). Evidencia:
  `docs/audits/evidence/repro_runs/r3_pre_statistics_defects_2026_09_03.{py,out}`.
- **Reparación** en `agent_plugins/paired_inference.py` (reutilizable):
  `holm_adjust` con el máximo acumulado exigido (monótono, cap 1);
  `paired_t` con resultados FINITOS predeclarados para diferencias de
  varianza cero (constante positiva → p=0, IC degenerado; cero → p=1;
  negativa → p=1); NaN/inf en scores, diferencias, estadísticos y
  p-ajustados → refusal tipado `PairedInferenceError`. El runner N1
  ahora consume el helper (el `survival_t3` anidado y el Holm ingenuo
  fueron eliminados).
- **18 tests** (`tests/unit/test_paired_inference_r3.py`), incluidos
  los exigidos que FALLAN bajo el camino `inf → linspace → NaN` y bajo
  el Holm no monótono. **Tres mutaciones muerden**: Holm sin máximo
  acumulado → 3 fallas; rama de varianza cero retirada (t=inf
  restaurado) → 4 fallas; guardia de finitud eliminada → 1 falla;
  restaurado → 18/18.
- **Re-derivación (R3.5)** desde los 28 registros terminales congelados
  a través del `aggregate_final` reparado:
  **veredicto primario INALTERADO = `PREDICTABILITY_NOT_DEMONSTRATED`**.
  Holm corregido {direct_linear: 1.0, direct_temporal: 1.0} — el 0.9419
  publicado estaba SUB-ajustado por el defecto (dirección conservadora:
  la corrección solo debilita al brazo). Artefacto:
  `TARGET_IDENTIFIABILITY_N1_R3_REDERIVATION_2026_09_03.json`.

## 3. Predeclaración N2 comprometida por separado, pre-resultado

`TARGET_HORIZON_DATA_CENSUS_N2_PREDECLARATION_2026_09_03.json` sellada
en **`9d73e078`** y una precisión pre-resultado (regla de elección de
modelo por calibración + refit uniforme fit+cal) en **`3ec62243`** —
ambas ANTES de materializar o computar score alguno. El runner
(`tools/target_horizon_census_n2.py`) y sus 19 tests de refusal en
**`ef395a2e`**, también antes de la ejecución. Ningún objetivo,
horizonte, métrica ni modelo se añadió después de ver resultados.

## 4. Fórmulas exactas, baselines, métricas, rangos de rol y digests

- **Constructores ejecutables del contrato — cero fórmulas
  duplicadas**: `forward_log_return_targets` (h 1/3/6/12),
  `realized_volatility_targets` (h 3/6/12, ε=1e-8, sin anualización),
  `barrier_hit_labels` (h 6/12; escala trailing close-to-close
  lookback 64, mults 2.0/2.0, colisión adversa-primero, OHLC validado).
- **Baselines/pérdidas por familia**: retornos → cero y media-fit,
  error cuadrático (secundarias: acierto direccional, Spearman);
  volatilidad → trailing literal y AR1 calibrada, **QLIKE** sobre
  varianzas (secundaria R²); barrera → prior de clases del rol fit,
  **log loss multiclase** (secundarias Brier, macro recall, soporte).
  Sin agrupación numérica entre familias; todo se convierte a skill
  intra-familia 1 − L_modelo/L_baseline.
- **Modelos CPU**: historia del objetivo (12 retornos crudos / lags
  0-3 del trailing / lags 0-3 de la escala de barrera) y lineal
  regularizado sobre el resumen causal FIJO de 249 features (último
  valor, media, sd por canal); intercepto sin penalizar,
  estandarización solo-fit, λ/C y elección de modelo solo por
  calibración; condicionamiento, λ y normas de coeficientes
  registrados por unidad.
- **Geometría**: cuatro ventanas de score disyuntas de 216 obs antes
  de la frontera consumida (fila 1533), embargo uniforme 3 filas
  muestreadas (máximo de ceil(h/stride)); rangos exactos por rol
  publicados en el ledger y republicados en el trace. w1 score
  [660,876], w2 [879,1095], w3 [1098,1314], w4 [1317,1533]; cal de 176
  precede a cada score con embargo; fit = prefijo causal estricto.
- **Digests inmutables** (ledger, re-verificados al reclamar):
  census_inputs `07c5ff08…`, csv fuente `1b447c66…`, generation
  sellada `b1dd6156…`, manifest `a466c9f8…`, código `d242be63…`,
  predeclaración `93ca004a…`. **Ancla de reproducción**: el
  materializador N2 exigió el npz N1 congelado (`0f31661c…`) y probó
  igualdad EXACTA de arrays con su propia reconstrucción — deriva de
  muestreo imposible.

## 5. Controles negativos y registros terminales

**60/60 COMPLETED, intento 1, cero fallas, cero timeouts, cero
reintentos.** Registros en
`TARGET_HORIZON_CENSUS_N2_UNIT_RECORDS_2026_09_03.json` (identidad,
estado, digest de resultado por unidad). Los seis controles licencian
el harness:

- **Centinelas de fuga** (objetivo verdadero como única feature):
  skill por ventana ≈ **0.9995** en las tres familias — la fuga
  aparece irrealmente fácil; el test SÍ detecta filtración.
- **Objetivos desplazados** (+37 filas circulares, modelo 3): NINGUNO
  pasa (ej. barrera: skill agrupado −0.020, p 0.884) — el harness no
  premia alineación rota.

## 6. Incertidumbre consciente de dependencia, multiplicidad y selección

Bootstrap de bloques circulares POR VENTANA (bloque 6 ≥ embargo,
B=2000, semilla 505) sobre diferencias de pérdida por observación vs
el baseline más fuerte; jamás t entre ventanas «independientes». Holm
sobre los NUEVE candidatos con el helper R3 reparado (monótono).
Licencias: 36 bloques efectivos por ventana (mín. 8); soporte de
clases 0/1 ≥ 30 por ventana de score en barrera; varianza de objetivo
positiva por ventana. Traza de selección completa en el trace.

## 7. Veredicto único y consecuencia permitida

# `TARGET_CANDIDATE_FOUND` — seleccionados: `bar_h6` y `bar_h12`

| candidato | skill agrupado vs baseline más fuerte | ventanas positivas | p bootstrap | Holm |
|---|---|---|---|---|
| ret_h1 | −0.042 (vs fit_mean) | no | 0.993 | 1.0 |
| ret_h3 | −0.004 (vs cero) | no | 0.601 | 1.0 |
| ret_h6 | −0.036 (vs cero) | no | 0.994 | 1.0 |
| ret_h12 | −0.041 (vs cero) | no | 0.985 | 1.0 |
| vol_h3 | −0.003 (vs AR1 calibrada) | no | 0.549 | 1.0 |
| vol_h6 | −0.124 (vs AR1 calibrada) | no | 0.908 | 1.0 |
| vol_h12 | −0.196 (vs AR1 calibrada) | no | 0.991 | 1.0 |
| **bar_h6** | **+0.0270** (vs prior de clases) | **4/4** (+0.018..+0.042) | **0.0005** | **0.0045** |
| **bar_h12** | **+0.0259** (vs prior de clases) | **4/4** (+0.013..+0.038) | **0.0005** | **0.0045** |

Lectura honesta y acotada: **toda familia de regresión falla contra su
baseline calibrado** — consistente con N1 y con el screen v2. La única
estructura condicional demostrable vive en el objetivo de **primer
toque de barrera**, y el modelo seleccionado en las 8 unidades
ganadoras fue la **historia del objetivo** (lags de la volatilidad de
escala), no el resumen de 83 insumos: el régimen de volatilidad
condiciona las frecuencias de clase. La mejora es modesta (2.6-2.7%
de reducción de log loss sobre el prior incondicional) pero positiva
en las cuatro ventanas, material (≥0.02) y sobrevive bootstrap
dependiente + Holm. Nota de asimetría declarada: el baseline de
barrera (prior) es más débil que los baselines calibrados de
regresión; esto responde exactamente la pregunta del censo — qué
objetivo porta señal condicional demostrable — sin reclamo
confirmatorio.

**Consecuencia permitida (su §6):** el censo es development-only; la
selección de `bar_h6`/`bar_h12` queda disponible para una futura orden
de confirmación. **La confirmación neuronal NO fue iniciada.**

Ejecución real: ~4 min de pared (techo 7200 s), 4 workers CPU,
heartbeat por minuto con completed/total/throughput, `STOP_CENSUS` con
terminate+reap documentado, watchdog sin eventos.

## 8. Conteos literales del tip final, commits, ramas, árboles

- Focales en el tip de trabajo: R3 18 passed · N1 11 passed · CAS 10 +
  atestación 12 (51 passed juntos) · N2 19 passed · índice de
  superficie 17 passed.
- Mutaciones: R3 3/4/1 fallas (restaurado 18/18); N2 2/2 fallas
  (restaurado 19/19).
- **Suite completa SOBRE EL TIP FINAL `c8e14442`** (corrida DESPUÉS
  del commit, conforme a la regla permanente): **2 failed, 2809
  passed, 4 skipped, 68 warnings in 227.00s** — las dos fallas son el
  par D1-anchor preexistente conocido. Los 37 tests nuevos (18 R3 +
  19 N2) pasan sin skip alguno en aislamiento; los 3 skips
  adicionales respecto de la corrida N1 (1→4) provienen de tests
  preexistentes condicionados al ambiente (2775 + 37 − 3 = 2809).
- Commits empujados a `origin/satoshi/data-first-sota-20260826`:
  `701dc828` (R1-R3), `9d73e078` (predeclaración N2), `3ec62243`
  (precisión pre-resultado), `ef395a2e` (runner + batería), y el
  commit de este packet + evidencia. Árbol limpio al empuje.

## 9. Declaración

**C1-C5 no fueron reabiertos.** Ningún trabajo GPU, ningún SAC, ningún
extractor nuevo, ninguna confirmación neuronal, ninguna promoción de
checkpoint, ningún acceso a roles sellados, ningún trading vivo,
ningún trabajo del colector MT5, ningún cambio de servicio, ningún
comando de venue ocurrió bajo esta orden.

— General Satoshi III

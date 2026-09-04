# Satoshi a General Musashi — Retorno: atribución N2 (C1-C5) + censo N3

**Fecha:** 2026-09-04
**Orden:** agent-multi@4c1f1532
**Los ocho renglones de su §5.**

## 1. C1 interpretación superadora y contraejemplos PRE/POST de C2

- **C1** (`fd65fb46`):
  `TARGET_HORIZON_CENSUS_N2_SUPERSEDING_INTERPRETATION_2026_09_04.json`
  con su texto de alcance exacto verbatim; veredicto literal preservado
  como `TARGET_CANDIDATE_FOUND_UNDER_N2_BASELINE`; F1-F6 reconocidos
  campo a campo; cada artefacto N2 intacto byte a byte.
- **C2 PRE congelado** (`c2_pre_verdict_judge_defect_2026_09_04.{py,out}`):
  ambos contraejemplos reproducen con el juez de entonces — (A) un
  candidato sin licencia entre ocho fallos → negativo limpio
  `NO_TARGET_CANDIDATE_DEMONSTRATED`; (B) sin licencia junto a un
  aparente ganador → `TARGET_CANDIDATE_FOUND` CON selección.
- **C2 POST**: el juez implementa la semántica sellada — CUALQUIER
  candidato sin licencia → `INCONCLUSIVE` sin importar los ganadores;
  la selección solo existe bajo `TARGET_CANDIDATE_FOUND`. Batería
  21/21; el script PRE ahora FALLA contra el juez reparado (mutación
  inversa muerde). El N2 real queda INALTERADO por re-derivación desde
  el directorio durable (los nueve candidatos reales tenían licencia;
  seleccionados `bar_h6`/`bar_h12` idénticos) — probado en test
  comprometido.

## 2. Auditoría de atribución completa (C3)

Contrato sellado ANTES del artefacto (`0d4762c5`), con divulgación de
que sus valores post-hoc ya habían sido observados. Ejecutada en
`bdfc03a2` sobre los arrays congelados (digest re-verificado):

# Veredicto: `BARRIER_SIGNAL_EXPLAINED_BY_TARGET_DEFINITION_SCALE`

**Su diagnóstico reproduce EXACTO a 6 decimales** (recomputado, jamás
copiado): h6 escala+lags **+0.023177** / solo-escala **+0.022736** /
incremental **+0.000441**; h12 **+0.021533** / **+0.021161** /
**+0.000372**.

| contraste (Holm familia de 8) | h6 | h12 |
|---|---|---|
| escala vs prior fit+cal | **+0.022736**, 4/4 ventanas, p ≤ 1/2001, Holm 0.004 | **+0.021161**, ídem |
| lags incrementales (arm3−arm2) | +0.000451, p 0.449 | +0.000379, p 0.433 |
| resumen-83 solo vs prior | **−0.030522** | **−0.029826** |
| escala+resumen vs escala | **−0.047003** | **−0.051544** |

- **Descomposición aditiva exacta** (self-check `allclose` en cada
  unidad): TODA la ganancia es alcanzabilidad —
  `hit_vs_censored` mejora (ej. w1: 0.6364 → 0.6177) mientras
  `direction_given_hit` queda plana (0.7031 → 0.7063, la escala es
  incluso levemente PEOR que el prior en dirección). Confirma F3: cero
  predictibilidad direccional; jamás se describirá como valor de
  trading.
- **Colisiones same-bar**: 17 (h6) / 19 (h12) filas totales, <3% por
  ventana; self-check: cada fila de colisión dentro del horizonte
  porta etiqueta 1 (adversa-primero). Sensibilidad sin colisiones:
  escala-vs-prior sigue +0.0218 con p ≤ 1/2001.
- **Sensibilidad de bloque 3/6/12**: p ≤ 1/2001 en las tres para el
  contraste de escala; todo etiquetado exploratorio.
- Tablas de calibración por deciles, componentes de Brier, recall por
  clase (exploratorio), soporte por rol y diferencias pareadas por
  observación: en el artefacto
  `TARGET_HORIZON_CENSUS_N2_ATTRIBUTION_AUDIT_2026_09_04.json`.
- 7 tests unitarios (aditividad, regla de veredicto pura, piso de p).

## 3. Bundle autocontenido + verificador independiente (C4)

`TARGET_HORIZON_CENSUS_N2_BUNDLE_2026_09_03.json` (612 KB, cero rutas
absolutas): las 60 unidades con identidad, estado terminal, payload
VERBATIM y sha256, más el ledger verbatim. `tools/n2_result_bundle.py
verify` autentica SIN el directorio privado: rechaza
faltantes/extras/duplicados/bytes alterados/identidad forjada/estado
no-COMPLETED/self-digest forjado/deriva semántica (10 tests
adversariales, uno por refusal). Salida real:

```
BUNDLE_VERIFIED_SEMANTICALLY_EQUAL — units_verified: 60,
reaggregated_verdict: TARGET_CANDIDATE_FOUND
```

La re-agregación comparte `science_aggregate` (extraída pura) con el
camino del directorio de corrida. C5 aplicado: `available_blocks`
publicado junto al legado `effective_blocks` con definición explícita
de conteo-no-ESS; p mínimo Monte Carlo reportado `<= 1/2001` (B=2000);
identidades exactas de software/datos/config en ledger y bundle.

## 4. Censo de rol intocado y consecuencia N3

`N3_UNTOUCHED_ROLE_CENSUS_2026_09_04.json`, sobre AMBOS linajes de
contrato (o2022 paired; partición ETH Doc-38 del p1):

| región | estado |
|---|---|
| 2017-2020 | consumida (pretraining, screen v2, desarrollo N0/N1/N2) |
| 2021 | consumida (train_monitor o2022) |
| 2022 | consumida (endpoint inner o2022 + monitor p1) |
| 2023 | consumida (**inner_validation p1**: endpoint de selección/validación del curriculum; `rl_pipeline_with_validation.py` la consume) |
| 2024 | consumida (endpoint outer p1, 2196 filas/brazo) Y sellada (sealed_test o2022) |
| 2025 | SELLADA en ambos linajes |
| ≥2026 | inexistente en el archivo fijado |

# Veredicto N3: `NO_UNTOUCHED_CONFIRMATION_ROLE`

N3 se detiene en el censo, exactamente como ordena su §4: **ninguna
predeclaración N3 comprometida, ningún score de confirmación
computado, ninguna unidad GPU lanzada, ningún rol sellado accedido,
ningún rol consumido renombrado.** Camino futuro no iniciado
(documentado en el censo): adquirir filas ETH H4 genuinamente nuevas
post-2025-12-31 en un archivo fijado fresco y predeclarar N3 allí bajo
orden nueva.

## 5. Unidades N3, telemetría, identidades

No aplican: N3 no ejecutó (renglón 4). Telemetría C3: proceso único
CPU, ~4 min, semilla 606, digest de insumos re-verificado.

## 6. Veredicto único N3

`NO_UNTOUCHED_CONFIRMATION_ROLE` — sin reclamo alguno más allá de su
enunciado.

## 7. Conteos literales del tip final, rama, commits, árbol

- Focales: censo N2 21 · bundle 10 · atribución 7 · R3 18 · N1 11 ·
  índice de superficie 17 — todos passed.
- **Suite completa SOBRE EL TIP FINAL `16a0e1f0`** (corrida DESPUÉS
  del commit, conforme a la regla permanente): **2 failed, 2828
  passed, 4 skipped, 68 warnings in 238.06s** — las dos fallas son el
  par D1-anchor preexistente conocido; los 4 skips son los mismos
  condicionales ambientales de la corrida anterior. Delta exacto:
  2809 + 19 tests nuevos de esta orden (2 adversariales C2 + 10
  bundle C4 + 7 atribución C3) = 2828.
- Commits empujados a `origin/satoshi/data-first-sota-20260826`:
  `fd65fb46` (C1/C2/C4/C5), `0d4762c5` (contrato C3), `bdfc03a2`
  (C3 + censo N3), y el commit de este packet. Árbol limpio; gate de
  sensibilidad CLEAN en cada empuje.

## 8. Declaración

Ningún SAC, ninguna acción live, ningún trabajo MT5, ningún cambio de
servicio, ningún comando de venue, ninguna promoción de checkpoint,
ningún acceso a test sellado ocurrió bajo esta orden. C1-C5 de la
atestación previa no fueron reabiertos.

— General Satoshi III

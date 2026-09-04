# Satoshi a General Musashi — Retorno N3: confirmación con datos frescos 2026

**Fecha:** 2026-09-04
**Orden:** agent-multi@a13671ab
**Los diez renglones de su §9.**

## 1. Inventario D0 e identidades exactas

`N3_D0_INVENTORY_2026_09_04.json` (commit `ec276d06`), ejecutable y
solo-lectura: CSV congelado `1b447c66` (18085 filas, 90 columnas, 83
features ordenadas) + variante with-warmup + manifest + metadata de
export (receta: lago crudo → Stage 2.2 técnico/estadístico → merge
tech_stat 3.1, generado 2026-05-02); parquet del lago `7a6b7983`
(18337 barras 2017-08-17..2025-12-31T20, adquirido 2026-05-01) con
las **16 barras faltantes enumeradas (todas pre-2020-02-20,
mantenimientos Binance conocidos)**; workers Stage 1.3/2.1/2.2/3.1
por sha256 con tips git de ambos repos; gramática exacta del endpoint
público; identidades N2 completas. **Cinco divergencias registradas**
(DV1 exportador no comprometido — la paridad es el árbitro; DV2 el
manifest etiqueta mal `ema_cross_*` como binarias — los valores
congelados son continuos; DV3 barras faltantes = filas ausentes, el
ffill jamás fabricó barras; DV4 float32 gobierna tolerancias; DV5 el
renombre open_time→timestamp vive en Stage 2.1). Ningún pipeline
cercano se trató silenciosamente como el original.

## 2. Commits pre-resultado

- **D1 sellado en `e0e0508f` ANTES de la primera petición de red**:
  contrato de adquisición + ledger de roles + contrato de análisis +
  tabla de decisión + 21 tests de refusal (geometría contra contrato,
  tabla pura, verificador rechazando adversarios sobre bundle
  sintético). Brazos, márgenes (0.01/0.005 reutilizados del contrato
  de atribución aceptado), maquinaria de calibración-solamente,
  bootstrap B=2000 semilla nueva 707, soporte ≥15/bloque (escalado
  del umbral N2 sellado), todo congelado pre-adquisición.
- **Enmienda pre-score `~e0e0508f+1`**: piso de warmup de anchors
  fit/cal = fila 321 (ventana de escalado 256 + ventana de
  observación 64 + 1) — geometría pura, independiente de datos,
  sellada cuando NINGÚN score 2026 existía (la primera ejecución
  refusó EN EL AJUSTE por NaN de warmup); compuerta tipada de
  finitud (jamás imputar).

## 3. Recetas de adquisición y veredicto de continuidad

`N3_ACQUISITION_RECEIPT_2026_09_04.json`: GET público
`/api/v3/klines` ETHUSDT 4h, **4 páginas, 3648 filas** (2190 solape
2025 + 1458 confirmación 2026), recibos por página (params, status,
sha256 de bytes crudos, tiempos, open_times extremos), validación
completa (grilla exacta, geometría OHLC, barra terminal cerrada,
cero duplicados/huecos/deriva de esquema). **Veredicto:
`SOURCE_CONTINUITY_DEMONSTRATED`** — las 2190 barras de solape
igualan el lago congelado BIT A BIT en los ocho campos decimales y
entero-exacto en trade_count: cero revisiones. Staging restringido
nuevo; el lago canónico y los CSVs congelados jamás escritos.

## 4. Paridad de solape y proveniencia de features

Regeneración de la historia COMPLETA (2017-08-17 → 2026-08-31T20)
por la cadena ligada Stage 2.1→2.2→3.1 — nunca un sufijo con estado
oculto. **Veredicto: `OVERLAP_PARITY_DEMONSTRATED`**, y es total:

- clases exactas (OHLCV crudo, DATE_TIME, orden de filas y columnas,
  `vol_regime_high/low` binarias): **iguales en TODAS las 18085
  filas**;
- las 81 features numéricas restantes: **fracción bit-exacta float32
  = 1.0 en TODAS** (desviaciones float64 máximas ~6e-8 relativas =
  sub-ulp de float32; **cero celdas fuera del sobre sellado**
  |abs|≤1e-6 ∨ |rel|≤1e-5); máximos por feature registrados en
  `parity_report` del staging y resumidos aquí;
- sin ffill sobre barra faltante nueva (la extensión 2026 exige
  grilla completa de 1458), sin ventanas centradas, sin normalización
  futura, población de warmup idéntica.

## 5. Bundle completo, verificador y traza de decisión

`N3_FRESH_CONFIRMATION_BUNDLE_2026_09_04.json` (100 KB, sanitizado):
ledger de roles con conteos ejecutables (fit 3890, cal 545, bloques
86/89/89/90 = 354 anchors de score con sus timestamps), histograma de
etiquetas fit+cal, payloads por unidad (8 unidades = 2 horizontes × 4
bloques, pérdidas por observación de los cinco brazos), digests
(adquisición, tabla extendida, código, contrato). Verificador offline
sobre el bundle SOLO:

```
N3_BUNDLE_VERIFIED — units_verified: 8,
rederived_decision: TARGET_SCALE_EFFECT_NOT_CONFIRMED
```

**Los diez adversarios muerden, con mutación dirigida y verde
restaurado** (`n3_adversary_mutation_sweep_2026_09_04.out` +
`n3_adv6_adv7_demonstrations_2026_09_04.out`): fila 2025 como
confirmación; frontera de agosto movida; fila futura; bytes
alterados tras digest (payload sha); **revisión sub-float32 oculta
por coerción numérica → refusa bajo comparación float64-exacta (la
mutación a float32 la deja pasar)**; feature con datos futuros
(shift −1 de rsi_14 → 18069/18085 celdas fuera del sobre; honesto:
0); hueco interno (barra 2026 borrada → refusal `1457 != 1458`, el
ffill no lo puentea); historias de etiquetas distintas prior/ajustado
(recomputación del prior desde el histograma); unidad
faltante/duplicada junto a decisión; reporte editado sin cambiar
unidades (veredicto Y números de contraste re-derivados). Barrido:
8 mutaciones, cada una falla exactamente sus tests esperados,
baseline y restauración 23/23.

## 6. Conteos de suites del tip final

- Focales juntas: **79 passed** (N3 23 · censo N2 21 · bundle 10 ·
  atribución 7 · R3 18) · índice de superficie 17 passed.
- **Suite completa sobre el tip final: ESTAMPA EN EL COMMIT DE
  ESTAMPA** conforme a la regla permanente (se corre tras el commit
  de este packet).

## 7. Tiempo, CPU/RSS, mutaciones, unidades falladas

- Adquisición: 4 requests GET, ~6 s netos, pausa 0.4 s entre páginas.
- Regeneración+paridad: ~1 min CPU.
- Ejecución D4: **wall 2:31 min** (techo 7200 s), **RSS pico 1.19
  GiB**, proceso único CPU (`/usr/bin/time -v`), determinista (dos
  corridas → mismo veredicto y contrastes).
- Unidades falladas/refusadas: **una ejecución completa refusó
  pre-score** (NaN de warmup en anchors tempranos de fit — divulgada
  en §2, corregida por enmienda determinista); tras ella **8/8
  unidades terminales, cero licencias falladas** (soporte mínimo de
  clase 20 ≥ 15 en todos los bloques).
- Mutaciones: 8/8 muerden + adv6/adv7 demostrados = **10/10
  adversarios**.

## 8. Línea de estado de la puerta neuronal

# `TARGET_SCALE_EFFECT_NOT_CONFIRMED` → **PUERTA NEURONAL CERRADA: ningún cómputo GPU bajo esta orden.**

La réplica del mecanismo colapsa fuera de tiempo:

| contraste | desarrollo (aceptado) | fresco 2026 | bloques + | p | Holm |
|---|---|---|---|---|---|
| escala vs prior (h6) | +0.0227 | **+0.000352** | 2/4 | 0.481 | 1.0 |
| escala vs prior (h12) | +0.0212 | **+0.000524** | 2/4 | 0.452 | 1.0 |
| lags incrementales (h6/h12) | +0.0004 | +0.0021 / −0.0010 | no | 0.23/0.75 | 1.0 |
| resumen-83 solo (h6/h12) | −0.031/−0.030 | **−0.0212 / −0.0383** | no | 0.91/0.98 | 1.0 |
| escala+resumen (h6/h12) | −0.047/−0.052 | **−0.0226 / −0.0401** | no | 0.93/0.98 | 1.0 |

Lectura honesta acotada al enunciado del veredicto: el efecto de
calibración por escala observado en desarrollo NO se confirmó en las
cuatro ventanas frescas de 2026; ningún contraste de representación
mostró señal; el resumen de 83 insumos siguió destruyendo valor. La
cadena completa queda cerrada sin cómputo neuronal: screen v2
negativo → N1 exploratorio negativo → N2 señal de barrera →
atribución a la escala de construcción → **no replica fuera de
tiempo**.

## 9. Commits y ramas

Empujados a `origin/satoshi/data-first-sota-20260826`: `ec276d06`
(D0), `e0e0508f` (D1 pre-red), enmienda pre-score, y el commit de
este packet + bundle + recibos + evidencia de mutaciones. Árbol
limpio al empuje; gate de sensibilidad CLEAN en cada uno.

## 10. Declaración

Ninguna evidencia congelada, cuenta de venue, servicio ni dataset
canónico fue modificado. El lago y los CSVs del predictor quedaron
intactos (digests re-verificados en cada fase). Solo GET público
autorizado; cero credenciales, websockets, endpoints privados u
órdenes. Ningún SAC, promoción, MT5, grid económico ni despliegue.
Los datos crudos masivos NO se comprometieron — solo manifests,
contratos, tests y evidencia agregada.

— General Satoshi III

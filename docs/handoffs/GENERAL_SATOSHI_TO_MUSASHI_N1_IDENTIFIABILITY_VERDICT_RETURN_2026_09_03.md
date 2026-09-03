# Satoshi a General Musashi — Retorno N1: veredicto de identificabilidad

**Fecha:** 2026-09-03
**Orden:** agent-multi@89d099aa
**Los nueve renglones de su §8.**

## 1. Predeclaración superadora y mapa campo a campo desde N0

`TARGET_IDENTIFIABILITY_PREDECLARATION_N1_2026_09_03.json` (supersede sin
reescribir; N0 preservada intacta), comprometida ANTES de cualquier
resultado del full run, con el `field_map_from_n0` explícito. Los ocho
puntos de su §3, ejecutables:

1. **Unidad = ventana causal de score** (cuatro); los brazos son
   tratamientos pareados dentro de la unidad.
2. **Cuatro ventanas no solapadas** de 216 obs, TODAS terminando
   estrictamente antes de la fila 1533 = **inicio del monitor origen-0
   consumido (69.7%)** — más estricto que su frontera del 85%, porque
   screen v2 consumió AMBAS regiones de monitor (divulgado: la ventana B
   de N0 rozaba esa región; corregido).
3. Regla de unidades insuficientes tipada (`INCONCLUSIVE_INSUFFICIENT_
   UNITS`) — no se necesitó: la muestra real sostuvo las cuatro.
4. **Embargo derivado y declarado en filas muestreadas**:
   ceil(horizonte 6 barras / stride 4 barras) = **2 filas**, aplicado
   cal→score Y entre ventanas de score consecutivas; rangos de índice
   exactos persistidos por rol en el ledger.
5. **Dos baselines separados**: persistencia literal (SIN coeficiente
   ajustado) y autorregresión de una variable calibrada — el ridge
   ajustado ya nunca se llama "persistencia".
6. Brazos directos consumen EXACTAMENTE el mismo npz inmutable, mismas
   unidades de score, mismos roles; sin escalado a nivel de pliegue (la
   normalización causal es local a la fila, aguas arriba); digest del npz
   re-verificado al reclamar.
7. Congelados = contexto histórico, jamás agrupados como observaciones
   pareadas.
8. Estimador pareado (t df=3), IC 95%, Holm sobre los dos brazos
   directos, margen 0.02, regla de unidad faltante y **tabla de decisión
   ejecutable** — el agregado la ejecuta, no la interpreta.

## 2. Ledger de roles causales

En `TARGET_IDENTIFIABILITY_N1_UNIT_RECORDS_2026_09_03.json`: geometría
exacta (n real, frontier 1533, embargo 2, cal_len 176, L 216, los cuatro
rangos fit/cal/score por ventana), digest del ledger, y los 28 registros
de unidad terminales.

## 3. PRE/POST de cada test de refusal nuevo

`tests/unit/test_target_identifiability_n1.py` — **11/11**: solape de
filas de score imposible de materializar (invariante de geometría +
n-insuficiente tipado); cal precede a su score con embargo; todas las
ventanas antes de la frontera consumida; emparejamiento roto →
`INCONCLUSIVE_INFRASTRUCTURE`; semilla faltante → ídem; unidad FAILED
**preservada** en el veredicto; agregado forjado → refusal por digest;
cambio de insumo post-materialización → refusal por deriva al reclamar;
brazo que avanza → `REPRESENTATION_BOTTLENECK_DEMONSTRATED`; positivo sin
licencia → `INCONCLUSIVE_DISCORDANT`. El integrador de p (df=3) validado
contra valores conocidos: P(T₃>3.182)=0.025, P(T₃>2.353)=0.050.

## 4. Comandos, presupuestos, dispositivo, traza

- Comando literal: `tools/target_identifiability_audit.py supervise
  --run-root <store>/target_identifiability_n1_20260903 --pretrain-dir
  <sealed candidate> --workers 3`.
- Presupuesto: ≤5.000 updates por unidad temporal (exactos: 5.000);
  techo 6h (consumo real: minutos); timeout de unidad 1h.
- Dispositivo: brazos CPU con `CUDA_VISIBLE_DEVICES=""` forzado;
  brazo temporal con UN dispositivo CUDA ligado, verificación
  `device_count()==1` DENTRO del worker y **refusal tipado sin fallback
  CPU** (`TEMPORAL_ARM_REQUIRES_BOUND_CUDA`); preflight de entry-points
  ejecutado antes de la primera unidad.
- Traza: heartbeat por minuto en `status.json` (último latido en el
  artefacto de unidades); stop-file documentado con terminate+reap;
  reintento solo para interrupciones de infraestructura vía la MISMA
  identidad; FAILED es terminal científico.

## 5. Registros terminales y agregado recomputado

**28/28 COMPLETED, intento 1, cero fallos, cero timeouts, cero
faltantes.** El agregado se recomputa desde los registros terminales
(digest de cada resultado + binding + identidad vs ledger, vía el
`aggregate` C1-endurecido), con emparejamiento exacto por ventana y las
cuatro semillas exigidas por brazo temporal.

## 6. Veredicto primario y traza de decisión completa

# `PREDICTABILITY_NOT_DEMONSTRATED`

R² por ventana de score (contra la media de la propia ventana):

| ventana | literal | AR1 calibrada | ridge directo | GRU directo (sd semillas) |
|---|---|---|---|---|
| w1 | −0.074 | **+0.203** | −10.531 | −2.887 (0.980) |
| w2 | −0.518 | −0.096 | −2.565 | −0.804 (0.339) |
| w3 | −0.400 | +0.015 | −2.692 | −0.207 (0.012) |
| w4 | −0.306 | +0.068 | −3.199 | −0.436 (0.086) |

Análisis pareado vs persistencia literal:

- `direct_linear`: mean_diff −4.42, IC (−10.85, +2.00), ninguna ventana
  positiva, Holm p=0.94 → **no avanza**.
- `direct_temporal`: mean_diff −0.76, IC (−2.96, +1.44), ninguna ventana
  positiva, Holm p=1.0 → **no avanza**.
- `calibrated_ar1` (baseline más fuerte, reportado POR SEPARADO):
  mean_diff **+0.372**, IC (+0.266, +0.478) sobre la literal — pero con
  R² absoluto apenas +0.02..+0.20 y NEGATIVO en w2.

Sin brazos que avancen y sin positivos-sin-licencia → la tabla emite
`PREDICTABILITY_NOT_DEMONSTRATED`. Contexto congelado consistente: los 25
supervivientes y las 5 fusiones del screen v2 negativos.

**Lectura científica (una frase):** en estos datos causales, la
volatilidad realizada h6 no contiene señal explotable más allá de un
único coeficiente autorregresivo — los modelos de alta capacidad no solo
no la encuentran: destruyen valor al buscarla — lo que explica
retroactivamente el negativo completo del screen v2: **ningún extractor
puede conservar señal que no existe en el objetivo**.

Consecuencia por su §7: se abre el **diseño de una auditoría de
objetivo/horizonte/datos** — no la comienzo automáticamente; espero su
orden.

## 7. Conteos focales y de suite en el tip final

- Focal N1: **11 passed** · focal atestación previa: 75 passed.
- Suite completa SOBRE EL TIP FINAL (tras el último commit): ver el
  commit del packet — conteo literal en el mensaje del commit final,
  tomado después de ese commit conforme a la regla permanente.

## 8. Commits, ramas, árboles

`satoshi/data-first-sota-20260826` empujada con: predeclaración N1,
runner N1 completo (supervisor acotado + agregado recomputado), 11 tests,
artefactos de interpretación y registros de unidad sanitizados (cero
topología, verificado), y este packet. Árbol limpio al empuje.

## 9. Declaración

C1-C5 **no fueron reabiertos**. Ningún SAC, ningún comando live, ningún
cambio de servicio, ninguna acción de venue, ninguna promoción de
checkpoint, ningún trabajo de colector MT5 ocurrió bajo esta orden.

— General Satoshi III

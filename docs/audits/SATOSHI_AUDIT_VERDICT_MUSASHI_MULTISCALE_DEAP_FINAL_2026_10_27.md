# Dictamen de auditoria: estrategia sintetica multiescala optimizada con DEAP (entrega final de Musashi)

Fecha: 2026-10-27. Auditor: Satoshi. Solicitante: propietario.
Objeto: `FINAL_REPORT.md`, `FINAL_AUDIT_REQUEST.md` y todos los artefactos de la entrega
final (`CLOSE.json`, `CONFIRMATION.json`, `CONTROLS.json`, `DECISION.json`,
`FINALISTS_BEFORE_CONFIRMATION.json`, `IDENTITY.json`, `MATRICES.json`, `SCALES.json`,
`SEARCH_RESULTS.json`, `SENSITIVITY.json`, `STATUS.json`, `TAPES_*.json`, `cache/`,
`finalists/`, `searches/`), con el codigo que citan.
Rama de este dictamen: `satoshi/audit-musashi-multiscale-deap-20261027`, creada desde
`origin/codex/rl-static-temporal-20261018` @ `1833b28b`.

Ningun archivo de Musashi fue modificado. Toda ejecucion fue en CPU
(`CUDA_VISIBLE_DEVICES=""`), bajo `crispdm-run -q -m 2G -t 1800s -n audit-mf`, sobre
copias desechables del arbol y del motor; las mutaciones (espejo, ruido, gaps, escalera
de calidad del oraculo, manipulacion de recibos) se hicieron solo en memoria del proceso
auditor. No se leyo ninguna fila financiera ni TEST.

Rutas privadas por su papel:
`$ENTREGA` = directorio de la entrega final multiescala-DEAP (2026-10-26);
`$ARBOL` = agent-multi `1833b28b` exportado;
`$MOTOR` = heuristic-strategy `codex/execution-cost-modes-20261026` @ `cd6489f` exportado
(las 56 huellas `engine_sources` de `IDENTITY.json` coinciden 56/56 con esa punta; las 221
huellas `sources` coinciden 221/221 con `1833b28b`);
`$RECETA` = copia de la receta de carga `8da8360a...` (identica a la que fija `IDENTITY.json`);
`$C4` = directorio del calificador de aprendizaje C4 (2026-10-26).
Los guiones del auditor y sus salidas pequenas estan en
`docs/audits/evidence/multiscale_deap_20261027/` de esta rama.

---

## 1. Dictamen global

**RECHAZADO** como "estrategia final optimizada y su justificacion experimental".

La solicitud pide tres dictamenes separados (ejecucion / optimizacion / confirmacion):

| Componente | Dictamen | Fundamento resumido |
|---|---|---|
| **Ejecucion** (contrato de ordenes, SL/TP, tamano, costes, calendario, contabilidad, reproducibilidad) | **ACEPTADO** | 0 violaciones de SL/TP/tamano/apalancamiento en todos los libros; 26/26 cargas re-simuladas bit a bit; 9/9 busquedas DEAP re-ejecutadas desde cero con `STATE.json` byte a byte identico; 2 reanudaciones partidas identicas; 7,310/7,310 sellos de cache validos; 3/3 verificaciones de matriz reproducidas; 47,628/47,628 celdas del finalista recomendado re-simuladas (seccion 6). |
| **Optimizacion** (DEAP, objetivos, eleccion del recomendado) | **RECHAZADO** | Los finalistas no se distinguen de los controles sin optimizar (IC 95% de la diferencia de Sharpe contiene 0 en confirmacion para los 3) y pierden contra ellos en el panel espejo; las asimetrias largo/corto de los genes son artefacto de un panel de calibracion desbalanceado; el termino de actividad es inerte; 4 genes topan la cota en S+R/S+R+A; la regla de recomendacion es circular (C1, M1-M4). |
| **Confirmacion** (panel independiente, congelado antes) | **ACEPTADO CON CORRECCIONES OBLIGATORIAS** | El orden es correcto y verificado (finalistas congelados 82 ms antes de la primera evaluacion de confirmacion; el recomendado se elige solo con calibracion). Pero el informe no declara que la confirmacion es nula frente a los controles ni da intervalos, y el panel de confirmacion es la misma familia determinista (sin ruido, sin gaps, sin espejo) que la calibracion (C2, M8). |

El global es RECHAZADO y no "aceptado con correcciones" porque lo que la entrega afirma
entregar es precisamente la decision: "Recommended calibration arm: sharpe_s7" como
estrategia optimizada. Esa decision no esta sostenida: (a) no supera a los puntos de
partida no optimizados fuera del panel de busqueda; (b) su ventaja se invierte al reflejar
el panel; (c) la regla que la elige favorece por definicion al brazo Sharpe; (d) el espacio
de escenarios no cumple el requisito del propietario (todas las combinaciones de tendencia
x volatilidad x cambios a varias intensidades y horizontes, superpuestas, reflejadas, con
comportamientos esperados/prohibidos escritos antes); (e) dos de los tres comportamientos
obligatorios no estan probados y se violan en una fraccion medible de operaciones.
La ingenieria de ejecucion y la reproducibilidad son excelentes y se aceptan sin reservas.

---

## 2. Puntos de verificacion de la solicitud, uno por uno

Cada punto lleva exactamente uno de VERIFIED / UNVERIFIED[motivo] / REFUTED[contraevidencia].
Recuento: **19 VERIFIED, 0 UNVERIFIED, 4 REFUTED** (23 puntos).

| # | Punto (texto de la solicitud) | Veredicto | Evidencia |
|---|---|---|---|
| V01 | Auditar `FINAL_REPORT.md`/`DECISION.json` contra busquedas, controles, sensibilidad, confirmacion y matrices | VERIFIED | Cada cifra de las dos tablas y de las listas del informe se re-derivo desde los JSON y desde los libros sellados (`a1_recompute.py`): 9 fitness, 9+14 Sharpe/R/ordenes/lados/tasa/DD, coinciden a la precision impresa; `emit_final` (`tools/multiscale_deap_strategy.py:37-99`) imprime directamente esos JSON. Las omisiones del informe son hallazgos (M8), no discrepancias numericas. |
| V02 | "No sustituir evidencia del punto medio por evidencia del finalista optimizado" | REFUTED[las matrices si consumen los genes del finalista, pero los controles invertido y NOP (`tools/multiscale_deap_strategy.py:187-189`) y toda la sensibilidad de un gen (`:154-158`) se ejecutan SOLO con el genoma de punto medio `[.5,.5,1]*2`; ningun finalista pasa por cinta invertida ni NOP] | `CONFIRMATION.json` filas `midpoint_inverted`/`midpoint_nop`; `SENSITIVITY.json` 12 filas, todas centradas en `controls[0]`. |
| V03 | Recalcular cada objetivo desde libros y escalas congeladas | VERIFIED | Escalas recomputadas como mediana de |componente| de los 3 controles: `sharpe 11.02200635586552`, `return_fraction 0.06446711681504752`, `orders_per_day 0.825` = `SCALES.json`. Fitness de las 9 busquedas recomputada desde `episodes[*].daily_returns/trades`: diferencia 0.0 en las 9 (p. ej. `sharpe_s7` 1.0788496108647252). |
| V04 | Reanudacion DEAP exacta | VERIFIED | `r2_deap.py`: `sharpe_s7` y `sharpe_return_activity_s43` detenidas en la generacion 37 y reanudadas: `payload_sha256` final identico al sellado. |
| V05 | Replay sembrado | VERIFIED | `r2_deap.py`: las 9 busquedas re-ejecutadas desde cero (semillas 7/19/43): 9/9 `STATE.json` byte a byte identicos, 16,328 evaluaciones, 0 fallos de cache. `r1_replay.py`: 26 cargas (12 genomas x 2 paneles + invertido + NOP) re-simuladas con `$MOTOR`: digest de carga identico 26/26. |
| V06 | Cintas inalteradas entre genes | VERIFIED | Las 7,297 evaluaciones de calibracion del cache comparten un unico conjunto de 24 huellas de cinta (= `TAPES_calibration_ideal.json`); `output_tape` (`app/multiscale_strategy_search.py:64-89`) no lee genes y se construye sobre una plantilla con `DirectionalRisk((.5,.5,1)*2)` (`:102`). |
| V07 | Elitismo de controles | VERIFIED | Los 3 controles entran en cada poblacion inicial (`tools/deap_output_calibration.py:49-52`); el elite se copia cada generacion (`:56,66`); mejor control vs mejor final: S 1.002096 vs 1.071836-1.078850; S+R 2.427524 vs 2.753444-2.770657; S+R+A 3.427524 vs 3.750166-3.758023. |
| V08 | Identificabilidad del SL | VERIFIED | Sensibilidad: SL-largo 0->1 cambia S 11.100->9.631 y cierres SL 16->11; SL-corto 0->1: S 10.659->9.768, SL 15->11. Ninguno de los 7,297 genomas evaluados tiene 0 cierres SL (121,319 cierres SL en total). Nota: en los 3 brazos Sharpe el SL-corto converge a la cota 0 (M4). |
| V09 | Identificabilidad de la actividad | REFUTED[el termino A no esta identificado: rango 0.980-1.045 sobre los 7,295 genomas elegibles; el tope de 1 entrada/dia (`max_entries_per_day: 1` en `$RECETA`) se satura en 15/24 cintas de calibracion y 18/24 de confirmacion; S+R y S+R+A dan el MISMO genoma en s19 y en s7 (diferencia 1.7e-12 en un gen, informe identico)] | `a3_index.py` + analisis en seccion 6; `SEARCH_RESULTS.json`. |
| V10 | RR / equilibrio de coste | VERIFIED | `mean_effective_rr` y `mean_barrier_breakeven` recalculados por operacion desde los libros: identicos (p. ej. `sharpe_s7` 4.033 / 0.323 calibracion; 4.005 / 0.293 confirmacion). Tasa realizada > equilibrio en todos los brazos. Nota: es la media de cocientes, dominada por SL minimos (m6). |
| V11 | Ambas direcciones | VERIFIED | Todos los brazos tienen ambos lados y `AGENT_CLOSE` por lado; elegibilidad recomputada sin motivos. Nota: el panel de calibracion esta desbalanceado (M1). |
| V12 | Cada JSON de finalista contra la identidad de su matriz | VERIFIED | Para los 3: `strategy_genes` = genes del finalista; sha de `STRATEGY.json` = `strategy_config_sha256`; fuentes agente y motor de la matriz = `IDENTITY.json` (`r4_matrix.py`). |
| V13 | Las 47,628 celdas | VERIFIED | Cobertura exacta y balance por eje reproducidos (3/3). El finalista recomendado se re-simulo COMPLETO (47,628 celdas) y los otros dos en 40 celdas aleatorias cada uno: identicos (seccion 6). |
| V14 | Verificacion independiente de cierre | VERIFIED | `verify()` re-ejecutado: resultado igual a `VERIFICATION.json` 3/3; una celda manipulada en memoria se rechaza (`SUMMARY_COUNTS`) 3/3. Alcance: verificacion de cobertura y huellas, no re-simulacion (m7). |
| V15 | Atacar el enrutamiento de salidas | VERIFIED | Ruta por posicion (0 plano, 1 largo, 2 corto) en `episode` (`app/multiscale_strategy_search.py:108-110`); `TAPE_CATEGORICAL_PARITY` exige que la distribucion de produccion sea argmax de los logits enmascarados (`:86-87`); un comando no permitido por la mascara se convierte en NOP (`:111`). |
| V16 | Atacar gaps adversos | REFUTED[ninguna de las 48 cintas de calibracion/confirmacion contiene un gap: `open_=np.r_[1.1,close[:-1]]` (`app/strategy_scenario_matrix.py:61`); 0 aperturas distintas del cierre previo medidas en las 48 cintas (336 episodios auditados); el gap solo existe en un testigo de broker sobre mercado plano (`tools/native_strategy_contract_study.py:76`). Contraejemplo: gaps intrasemanales de 30 pips contra la tendencia, desconocidos por el oraculo, bajan S del recomendado 13.70 -> 8.78-10.85 con 19-27 SL ejecutados en la apertura] | `r5_gaps.py`, seccion 6. |
| V17 | Atacar stops | VERIFIED | Toda operacion de todos los libros auditados tiene SL y TP del lado correcto (0 violaciones); SL antes que TP en empate (`$MOTOR execution.py:238-259`); gap a traves del SL se llena en la apertura. El disparo sobre precio medio (no bid/ask) es hallazgo M5. |
| V18 | Atacar cierres anticipados | REFUTED[la regla "cerrar anticipadamente solo cuando paga" no se cumple por operacion: contrafactual exacto (mismo camino, SL primero, `limit_through`, cierre forzado semanal) muestra que 30 de 65 `AGENT_CLOSE` del recomendado en confirmacion (46%) habrian rendido mas manteniendo, 13 de ellos alcanzando TP; 30 de 52 en calibracion] | `a5_early_risk.py`. En agregado el cierre anticipado si gana (+62.9 pips netos frente a mantener). |
| V19 | Atacar calendario | VERIFIED | 0 entradas antes del lunes 10:00Z ni salidas despues del viernes 10:00Z en los libros del recomendado y de `control_1` (ambos paneles); maximo 3 rollovers por operacion; triple swap en el indice 3. Cobertura de calendario: un solo reloj (m3/C2). |
| V20 | Atacar tamano | VERIFIED | `units == floor(q*floor(balance*0.10*50/precio))` en cada operacion (asercion `GENOME_UNITS` `:117-120` y recomprobacion propia); `rel_volume=0.1` y margen = nocional/50 en el 100% de las operaciones de los 6 brazos auditados (3 finalistas + 3 controles, ambos paneles, mas invertido/NOP); unidades enteras. |
| V21 | Atacar contabilidad del libro | VERIFIED | `check_trade` + `check_costs` + conciliacion de caja (`:115-122`) y re-simulacion bit a bit; `costs` del informe = suma de los cuatro componentes por operacion. |
| V22 | La eleccion precede a la confirmacion | VERIFIED | Codigo: `FINALISTS_BEFORE_CONFIRMATION.json` se escribe antes del bucle de confirmacion (`tools/multiscale_deap_strategy.py:180-186`). Marcas: ultimo `STATE.json` 03:53:09.773725Z, `FINALISTS` 03:53:09.777239Z, primera carga de confirmacion 03:53:09.859186Z. |
| V23 | Ninguna medida de confirmacion se uso para seleccionar | VERIFIED | `finalists()` usa solo `fitness` de calibracion (`app/multiscale_strategy_search.py:40-43`); la recomendacion usa `r['report']` de `searches` (calibracion) (`tools/multiscale_deap_strategy.py:43-44`); `confirmation` nunca entra en esas funciones. |

---

## 3. Hallazgos, de mayor a menor gravedad

### CRITICOS

**C1. Los tres finalistas "optimizados" no se distinguen de los controles sin optimizar fuera del panel de busqueda y pierden contra ellos en el panel espejo.**
- Afirmacion: "DEAP and full finalist execution are complete"; "Recommended calibration arm: sharpe_s7" (FINAL_REPORT, Decision).
- Evidencia (`a4_bootstrap.py`, bootstrap pareado por bloques de cinta, 24 cintas, 10,000 remuestreos):

  | Confirmacion | dS | IC 95% | P(dS<=0) |
  |---|---:|---|---:|
  | sharpe_s7 - control_1 | +0.278 | [-0.712, +1.041] | 0.272 |
  | sharpe_s7 - control_2 | +0.306 | [-0.707, +1.072] | 0.256 |
  | sharpe_return_s7 - control_2 | +0.051 | [-0.694, +0.586] | 0.430 |
  | sharpe_return_activity_s43 - control_2 | -0.170 | [-1.281, +0.581] | 0.645 |

  El finalista S+R+A (13.220) queda por debajo de `control_1` (13.419) y `control_2` (13.390).
  Panel espejo (precio reflejado alrededor de 1.1, mismo motor, mismas cintas reflejadas,
  `r3_variants.py`): confirmacion S-solo 13.416, S+R 13.215, S+R+A 13.466 frente a
  `control_0` 13.869, `control_1` 13.764, `control_2` 13.754: **los tres finalistas por
  debajo de los tres controles**. Calibracion espejo: S-solo 11.018 < `control_1` 11.131.
  Gaps adversos de 30 pips: S-solo 9.72 de media frente a `control_1` 10.25.
  Bajo la escalera de calidad del oraculo (ruido 0.25-2 sd en el objetivo del oraculo) el
  S-solo conserva una ventaja de +0.26 a +0.61 sobre el mejor control en la orientacion
  original: la ventaja existe en esa orientacion y desaparece al reflejarla.
- Impacto: no hay evidencia de que la busqueda haya producido una estrategia mejor que sus
  puntos de partida; la recomendacion `sharpe_s7` carece de soporte. Lo unico robusto es
  la diferencia de RETORNO, que la decide la exposicion (C1/M2), no la calidad.
- Correccion obligatoria: publicar los IC pareados de cada finalista contra cada control
  en confirmacion; exigir como criterio de aceptacion que el finalista supere al mejor
  control con IC por encima de 0 en un panel de confirmacion espejado y estocastico; si no
  lo hace, declarar "la optimizacion no mejora los controles" como resultado.

**C2. El espacio de escenarios no cumple el requisito del propietario y es determinista.**
- Afirmacion: plan 98/101 y el informe presentan las 24+24 cintas y la matriz de 47,628
  celdas como cobertura sistematica.
- Evidencia (cobertura medida en `IDENTITY.json.panels` y en el codigo):

  | Dimension exigida | Lo que hay | Falta |
  |---|---|---|
  | Ruido / volatilidad estocastica | `innovation = A*(sin(2*pi*t/P) + 0.3*sin(2*pi*t/7.3))` (`app/strategy_scenario_matrix.py:58-60`): **ningun termino aleatorio**; "volatilidad" = amplitud de una sinusoide; "dinamica" = escalon/rampa en un unico cambio | ruido, agrupamiento, colas, regimenes estocasticos, repeticiones con semilla de mercado |
  | Tendencias y transiciones | 8 patrones de signo x 3 intensidades en calibracion (24 cintas); un unico cambio en la barra 32 de 240 (87% de la cinta despues); fase 0 (calibracion) / 17 (confirmacion) | 0->0, cambios de intensidad con mismo signo (debil->fuerte), varios cambios por cinta, posicion del cambio |
  | Horizontes y superposicion | un horizonte por cinta (1-72); una sola deriva mas dos sinusoides fijas | tendencias superpuestas a varias escalas (p. ej. alza 72h con baja 8h) |
  | Espejo | ningun par reflejado; el patron y su espejo reciben vol/periodo/horizonte distintos (indices `(i+j)%3`, `(i+2j)%3`, `(3i+j)%6` en `app/multiscale_strategy_search.py:22-27`) | pares espejo exactos y prueba de simetria |
  | Choques / gaps | 0 gaps en las 48 cintas (V16); sin saltos | gaps, saltos de noticias, ensanchamiento de spread |
  | Spread / calendario | un perfil de spread por hora; un reloj semanal fijo (domingo 22:00Z, 120 barras/semana), sin festivos ni DST; DEAP solo con coste configurado | variantes de spread y calendario |
  | Comportamientos esperados/prohibidos escritos antes | M01-M09 son pruebas de contrato; no hay expectativas por escenario (p. ej. "plano y alta volatilidad: no abrir") | lista por escenario y testigos |
  | Escalera de calidad de salidas | solo ideal / invertido / NOP | escalera de ruido y punto de equilibrio de calidad (exigido para el carril F9) |

  DEAP ve solo 24 cintas (0.05% de la reja de 47,628), en un diseno fraccional confundido.
- Impacto: el Sharpe ~11-14 es una propiedad de sinusoides perfectamente predecibles con un
  oraculo clarividente; nada de lo medido se puede trasladar a otra distribucion de
  mercados. Las conclusiones comparativas (C1, M2, M3) estan atadas a este panel.
- Correccion obligatoria: generador con componente estocastico sembrado y varias
  realizaciones por escenario; pares espejo exactos; superposicion de tendencias;
  gaps/choques/spread variable; escalera de calidad del oraculo con punto de equilibrio;
  expectativas por escenario escritas y versionadas ANTES de ejecutar.

**C3. Dos de los tres comportamientos obligatorios no estan probados y se violan en una fraccion medible de operaciones.**
- (i) Abrir siempre con SL y TP: **cumplido** (0 violaciones; ver V17).
- (ii) Cierre anticipado solo cuando paga: la regla del oraculo es `hold = position*delta - cost/2` frente a `close = 0` (`app/multiscale_strategy_search.py:81`), un signo de horizonte, no una deteccion de "SL antes que TP". Contrafactual exacto (`a5_early_risk.py`): recomendado en confirmacion 30/65 cierres peores que mantener (13 habrian tocado TP); calibracion 30/52; `control_0` calibracion 33/44 con -102.6 pips netos frente a mantener.
- (iii) No abrir con riesgo alto frente a ganancia: **no implementado**: se abre si `|delta| > coste` (`:79`), sin mirar la distancia al SL. 61 de 219 aperturas del recomendado en confirmacion (28%) tienen ganancia del oraculo menor que la distancia al SL; 58/200 en calibracion.
- Ademas: no hay testigo de simetria espejo; el balance largo/corto por escenario no se compara contra la direccion de la tendencia en ningun informe.
- Impacto: la entrega no demuestra los comportamientos que el propietario fijo como reglas de negocio.
- Correccion obligatoria: testigos explicitos por escenario para (ii) y (iii) con conteos esperados escritos antes; regla de apertura que compare ganancia esperada con riesgo al SL; informe por escenario de largos/cortos frente al signo de la tendencia y prueba espejo.

### MAYORES

**M1. El panel de calibracion esta desbalanceado en direccion; las asimetrias largo/corto de los genes son artefacto del panel.**
- Evidencia: con genes simetricos, `control_1` hace 129 largos / 69 cortos en calibracion (83 L/4 S en cintas con deriva final positiva, 19 L/55 S con deriva final negativa, 27 L/10 S con deriva final cero) frente a 111/106 en confirmacion. Los genes Sharpe-solo convergen a SL-corto = 0 (cota) en 3/3 semillas y a q-largo/q-corto distintos y no reproducibles entre semillas (0.31/0.47, 0.36/0.43, 0.53/0.59). Espejo con genes intercambiados largo<->corto: S-solo confirmacion 13.697 (= original 13.696), es decir, el comportamiento es exactamente el de una asimetria de genes atada a la orientacion del panel.
- Impacto: los parametros direccionales "calibrados" no describen preferencias de riesgo por lado; describen el panel.
- Correccion: panel balanceado por construccion (cada escenario con su espejo exacto), y prueba de que genes intercambiados sobre el espejo reproducen el original.

**M2. La regla de recomendacion es circular y el objetivo Sharpe tiene una direccion plana en la exposicion.**
- Evidencia: la recomendacion ordena por Sharpe de calibracion (`tools/multiscale_deap_strategy.py:43-44`); los 3 brazos Sharpe-solo tienen S de calibracion 11.81-11.89 frente a 11.57-11.68 de los 6 restantes: la regla elige el objetivo que optimiza esa metrica. Incluso en calibracion la diferencia S-solo - S+R no es significativa: +0.315, IC [-0.312, +0.753]. Sharpe casi invariante a la exposicion: `control_1` y `control_2` tienen TP/SL identicos y q 0.2 vs 1.0, las mismas 198 operaciones, dS = +0.023 (IC [+0.001, +0.034]) y dR = -7.5 pp. El "Sharpe-only" con retorno 3.34% frente a 11.06% no elige riesgo: elige una exposicion arbitraria a lo largo de una direccion plana, con un sesgo debil hacia q pequeno por capitalizacion.
- Impacto: "Sharpe-solo gana" no es evidencia sobre que funcion de fitness usar.
- Correccion: comparar objetivos con una metrica externa a los tres (p. ej. utilidad con aversion al drawdown fija de antemano) o con frontera retorno/drawdown; declarar la invariancia de escala del Sharpe; no recomendar por la metrica que uno de los brazos maximiza.

**M3. El tercer objetivo (S+R+A) es inerte y su finalista lo decidieron dos operaciones.**
- Evidencia: V09. El finalista S+R+A es s43 (3.758023) frente a s7 (3.750455) porque hace 196 operaciones frente a 194 (A 0.8167 vs 0.8083 por dia habil: +0.0101 de fitness); es el peor finalista en confirmacion (13.220).
- Impacto: la tabla presenta tres objetivos comparados cuando solo hay dos distintos.
- Correccion: declarar A no identificado en esta entrega; retirar S+R+A como brazo o redefinir la actividad para que el tope diario no la sature.

**M4. DEAP: cotas tocadas, convergencia no demostrada y restriccion nunca activa.**
- Evidencia (poblaciones finales, `a3_index.py`): S+R y S+R+A: TP-largo, q-largo, TP-corto y q-corto en la cota superior en 18-20 de 20 individuos; S-solo: SL-corto en la cota 0 en 19-20 de 20. 7/9 busquedas terminan por `MAX_GENERATIONS`; `sharpe_s7` sigue mejorando en la generacion 100 (+2.5e-4 en las ultimas 10 generaciones, paciencia 0). La tolerancia relativa 1e-8 hace que mejoras de 1e-8 reinicien la paciencia. Solo 2 de 7,297 genomas evaluados fueron no elegibles (los dos extremos q=0 de la sensibilidad): la compuerta de sanidad nunca discrimino durante la busqueda porque el oraculo garantiza ambos lados y cierres. Poblacion 20, cruce SBX p=0.8, mutacion polinomica p=0.2 con indpb 1/6 (`tools/deap_output_calibration.py:56-64`): exploracion muy baja para 6 genes.
- Impacto: el optimo de S+R esta fuera de la caja declarada (TP en el tope de eficacia `1.5*sigma_rule`, tamano maximo); "calibrado" significa "en la cota".
- Correccion: declarar cotas tocadas por gen; ampliar o justificar la caja; criterio de convergencia con tolerancia significativa y multiples reinicios.

**M5. SL/TP se disparan sobre el precio medio, no bid/ask, justo donde la busqueda empuja el SL al minimo.**
- Evidencia: `$MOTOR app/short_horizon/expert_guided/env.py:259-276` usa `h`/`l`/`o` medios en `_protect` (`execution.py:238-259`); el spread se cobra solo como USD (`env.py:278-284`). Spread de `$RECETA` 0.3-3 pips por hora; `min_distance` 0.5 pip. El recomendado tiene SL minimo 0.50 pip (cortos) y 18 operaciones de calibracion con SL < 2 pips.
- Impacto: con disparo bid/ask, un SL corto a 0.5 pip del medio se ejecuta en cuanto el semi-spread lo cruza; la ventaja de SL-corto=0 que encuentra la busqueda Sharpe-solo es en parte un efecto del modelo de disparo.
- Correccion: disparar SL/TP con bid para largos y ask para cortos (o exigir SL >= spread vigente + margen) y repetir la busqueda.

**M6. El control positivo a traves de la red entrenada sigue ausente de la evidencia final; el C4 se ejecuto y fallo.**
- Evidencia: `$C4/STATUS.json`: `LEARNING_QUALIFICATION_FAILED`, 9/9 unidades; `RESULTS.json` brazo `planted` s7 (canal con futuro plantado) TRAIN Sharpe -0.4758, 496 operaciones, 5 `AGENT_CLOSE`. Informe 94 de Musashi: BUY argmax en 21,172 de 21,217 ventanas TRAIN del punto de control plantado. La entrega final inyecta las cintas directamente en `OrderDistribution` (`app/multiscale_strategy_search.py:84`) con `training_steps: 0`: no hay camino de la red a esta estrategia, y el informe final no lo menciona.
- Impacto: la estrategia sintetica no dice nada sobre si el sistema aprendido puede producir esas salidas.
- Correccion: declararlo en el informe final y no presentar la estrategia sintetica como resuelta para el carril aprendido hasta que un control plantado pase por la red.

**M7. La matriz de 47,628 celdas no evalua la estrategia; el titulo "all-scenario verification" lo sobreafirma.**
- Evidencia: en modo diseno, `compare()` inyecta en la cabeza de produccion el ranking calculado por el propio simulador y exige que elija su argmax (`tools/systematic_strategy_matrix.py:160-187`; `LOWER_VALUE_CHOSEN` `:177` es tautologico); los genes del finalista solo cambian niveles y tamano en esos contrafactuales de dos anclas por celda. No se produce Sharpe, retorno ni comportamiento del finalista por celda.
- Impacto: "Finalists and all-scenario verification" se lee como validacion del finalista en todos los escenarios; es validacion del contrato de ejecucion.
- Correccion: renombrar como "verificacion de contrato"; ejecutar el finalista episodio completo en la reja y publicar metricas por celda y por eje.

**M8. El informe omite lo que decide la interpretacion.**
- Omisiones: intervalos y nulidad frente a controles (C1); invariancia de escala del Sharpe (M2); inercia de A en esta corrida (M3); cotas tocadas y no convergencia (M4); caracter determinista y sin gaps de las cintas (C2); fallo del C4 (M6); que la sensibilidad y los controles invertido/NOP son solo del punto medio (V02).
- Correccion: anadir una seccion de limitaciones con estas cifras, generada desde los JSON.

### MENORES

- **m1.** Controles invertido/NOP solo con el punto medio (V02). Comportamiento correcto: invertido S -16.148, R -8.32%, tasa 5.4%, 109/112 lados; NOP 0 operaciones y Sharpe `undefined` (no cero fabricado). Correccion: correrlos tambien con cada finalista.
- **m2.** "Generations 100" son 101 generaciones (0..100); el numero de evaluaciones por busqueda (1,081-1,566) no se publica.
- **m3.** Sharpe con `sqrt(365)` sobre equity diaria de dias de calendario (13 por cinta, fines de semana incluidos) agrupando 24 cintas disjuntas de 2 semanas: magnitud sin interpretacion financiera; la etiqueta deberia decirlo.
- **m4.** `notional_usd = units*(entry_price+exit_price)` (`env.py:312`) es rotacion, no nocional (el doble).
- **m5.** `IDENTITY.json` fija el motor por huellas de archivo, no por commit; coinciden 56/56 con `cd6489f`, que deberia registrarse.
- **m6.** `mean_effective_rr` es media de cocientes TP/SL por operacion, dominada por SL minimos; RR agregado del recomendado en calibracion 4.29 (largos) / 3.61 (cortos), mediana 4.42. Publicar mediana y agregado.
- **m7.** `strategy_matrix_verification.verify` comprueba cobertura, recuentos y huellas, no re-simula; la "independencia" es de implementacion, no de computo. Esta auditoria cerro ese hueco para el finalista recomendado.
- **m8.** Plan 101 promete pruebas F02/F03/F06; en la suite multiescala existen 7 pruebas unitarias (F01, F04, F05, F07, F08) y una de integracion que se salta sin variables de entorno; F02/F06 estan como aserciones de tiempo de ejecucion, no como pruebas. Las 7+1 pasan (con las variables fijadas).
- **m9.** Asimetria de financiacion (-7 / +2.5 USD por lote y noche) interactua con la asimetria de genes; pequena aqui (largos pagan 25.18 USD, cortos cobran 19.81 USD en confirmacion del recomendado) pero no se separa por lado en el informe.

---

## 4. Estado de los hallazgos del dictamen anterior (2026-10-26)

| Hallazgo previo | Estado | Evidencia |
|---|---|---|
| C1 regla fija RR 0.916 pierde por construccion | **CORREGIDO en lo que se pidio publicar** | Cada informe trae `mean_effective_rr`, `mean_barrier_breakeven`, `mean_cost_pips` y la financiacion separada en `costs`; los controles de esta entrega tienen RR 0.93/2.03, equilibrio 0.62/0.43 y tasa 0.89/0.88 sobre cintas de oraculo. No se re-evaluo sobre datos de mercado (fuera de alcance). |
| C2 politicas aprendidas = largo ciego | **NO CORREGIDO** | Informe 94: punto de control plantado elige BUY en 21,172/21,217 ventanas TRAIN. La entrega final no contiene politica aprendida. |
| C3 genomas DEAP peores que controles; SL inertes | **PARCIALMENTE**: SL identificado (V08); controles ya en la tabla de confirmacion. **El problema de fondo persiste**: finalistas indistinguibles de los controles con IC y peores en espejo (C1). |
| C4 sin control positivo por la red | **NO CORREGIDO** (ejecutado y fallido) | `$C4`: `LEARNING_QUALIFICATION_FAILED` 9/9; plantado s7 TRAIN Sharpe -0.4758. Ausente del informe final (M6). |
| M1 INNER_A como bloque de seleccion | **CORREGIDO en este carril** | Calibracion y confirmacion separadas y congeladas en orden verificado (V22-V23); el plan 92 etiqueta INNER_A como seleccion. Matiz: la confirmacion es la misma familia determinista (C2). |

---

## 5. Lo que Musashi hizo bien, y como se comprobo

- **Reproducibilidad completa y verificable.** 9/9 busquedas re-ejecutadas desde cero con `STATE.json` byte a byte identico; reanudacion partida identica; 26/26 cargas re-simuladas con el motor fijado bit a bit; 7,310/7,310 sellos de cache validos (identidad, digest de carga y clave); 47,628/47,628 celdas del recomendado re-simuladas identicas.
- **Contrato de ejecucion correcto.** SL y TP en toda orden del lado correcto; tamano exacto `rel_volume 0.10`, apalancamiento 1:50 en base margen, unidades enteras; costes por evento conciliados por un calculo independiente (`check_costs`); calendario respetado; TP limite con 1 pip de traspaso; SL primero en empate.
- **Disciplina de diseno.** Procedimiento, cintas, controles, escalas y regla de recomendacion fijados en el plan 101 antes de buscar; cintas invariantes a los genes impuestas en tiempo de ejecucion; elitismo de controles verificado; finalistas congelados antes de la confirmacion (82 ms antes, por marca y por codigo); NOP con Sharpe indefinido, no cero; reglas de negocio como restriccion, no como termino del fitness.
- **Controles de cinta con el signo esperado.** Invertido pierde (S -16.15, R -8.32%); NOP no opera.
- **Matrices ligadas a los genes del finalista**, no al punto medio, con identidad, sha del JSON y fuentes verificados.
- **Honestidad parcial en el texto**: el informe dice "This is not learned-market performance" y que una actividad constante o un stop sin efecto se reportarian como no identificados.

---

## 6. Registro de reproduccion

Todos los guiones estan en `docs/audits/evidence/multiscale_deap_20261027/`; reciben las
rutas por argumento (`$ENTREGA`, `$ARBOL`, `$MOTOR`, `$RECETA`). Python del auditor:
entorno con torch 2.13.0, deap 1.4, pandas 3.0.3 (iguales a `IDENTITY.software`). CPU solo.

| Paso | Comando (por papel) | Resultado |
|---|---|---|
| Huellas | sha256 de `IDENTITY.sources` contra `$ARBOL`; `engine_sources` contra todas las ramas de heuristic-strategy | 221/221; 56/56 en `cd6489f` |
| A1 | `python3 -I a1_recompute.py $ENTREGA` | escalas y 9 fitness exactos; 14 filas de confirmacion exactas |
| A2 | `python3 a2_behaviour.py $ENTREGA $ARBOL` | 0 violaciones SL/TP/tamano/apalancamiento; tabla por cinta; 0 gaps en 48 cintas |
| A3 | `python3 a3_index.py $ENTREGA out.json` | 7,310 entradas, 0 sellos invalidos; 1 conjunto de cintas por panel/modo; 2/7,297 no elegibles |
| A4 | `python3 a4_bootstrap.py $ENTREGA` | IC pareados (C1, M2) |
| A5 | `python3 a5_early_risk.py $ENTREGA $ARBOL` | contrafactual de cierres y riesgo/ganancia de aperturas (C3) |
| A6 | `python3 a6_costs.py $ENTREGA` | RR/equilibrio recalculados; distancias SL; financiacion por lado |
| R1 | `python r1_replay.py $ARBOL $MOTOR $RECETA $ENTREGA <12 brazos> calibration,confirmation` (+ invertido, NOP) | 26/26 digest de carga identico |
| R2 | `python r2_deap.py $ARBOL $MOTOR $RECETA $ENTREGA <tmp>` | 9/9 `STATE.json` identicos; 2/2 reanudaciones identicas; 16,328 aciertos de cache, 0 fallos |
| R3 | `python r3_variants.py $ARBOL $MOTOR $RECETA $ENTREGA` | espejo, espejo con genes intercambiados, ruido 1x (3 semillas), escalera del oraculo eta 0.25/0.5/1/2 (3 semillas) |
| R4 | `python r4_matrix.py $ARBOL $MOTOR $RECETA $ENTREGA 40` | `verify()` = `VERIFICATION.json` 3/3; manipulacion rechazada 3/3; 120/120 celdas re-simuladas identicas |
| R5 | `python r5_gaps.py $ARBOL $MOTOR $RECETA $ENTREGA` | gaps adversos desconocidos por el oraculo (V16) |
| R6 | `python r6_fullcells.py $ARBOL $MOTOR $RECETA $ENTREGA sharpe_s7 k 3`, k=0,1,2 | 3 x 15,876 = 47,628/47,628 celdas re-simuladas identicas a los recibos (0 diferencias; ~775 s por bloque) |
| Pruebas | `pytest tests/unit/test_multiscale_strategy_search.py`; `pytest tests/integration/test_multiscale_strategy_search.py tests/unit/test_strategy_multiscale.py` con `PINNED_RECIPE_PATH`/`PINNED_ENGINE_ROOT` | 7 pasan; 8 pasan |

Contraejemplos minimos ejecutables (todos sobre copias en memoria, sin tocar evidencia):
1. **Espejo**: `r3_variants.py`, clave `mirror`: los tres finalistas por debajo de los tres controles en confirmacion.
2. **Escala del Sharpe**: `control_1` vs `control_2` en `CONTROLS.json`/`CONFIRMATION.json`: mismas operaciones, q 0.2 vs 1.0, dS +0.023/+0.028, dR -7.5/-7.4 pp.
3. **Actividad inerte**: `SEARCH_RESULTS.json`, `sharpe_return_s19` y `sharpe_return_activity_s19`: genes identicos.
4. **Gaps adversos**: `r5_gaps.py`, `gap30`: S del recomendado 8.78-10.85, 19-27 SL en la apertura.
5. **Cierre anticipado que no paga**: `a5_early_risk.py`: 30/65 cierres del recomendado en confirmacion peores que mantener.

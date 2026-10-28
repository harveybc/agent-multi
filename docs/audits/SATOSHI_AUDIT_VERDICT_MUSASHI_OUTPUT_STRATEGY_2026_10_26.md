# Dictamen de auditoria cruzada: salidas, estrategia y calibracion DEAP de Musashi

Fecha: 2026-10-26. Auditor: Satoshi. Solicitante: propietario.
Objeto: informes 84, 85, 89 y anexo 90 (agent-multi `c64ab10e`), con el codigo que
citan y los artefactos privados que su indice local senala. Rama de este dictamen:
`satoshi/audit-musashi-output-strategy-20261026`, creada desde `c64ab10e`.
Ningun archivo de Musashi fue modificado; todas las lecturas de evidencia fueron sobre
copias o en modo lectura. Solo se leyo TRAIN `[2020-01-01, 2023-07-01)`, tapes sinteticos
y los recibos ya sellados de INNER_A; ninguna fila con marca `>= 2025-01-01` fue leida.
En este documento las rutas privadas se nombran por su papel en el indice local:
`$ESPEJO` (campana congelada v2), `$ESTUDIO` (27x3 TRAIN), `$SINTETICO` (162+6),
`$DEAP` (nueve busquedas), `$MOTOR` (heuristic-strategy `cd6489f` exportado),
`$ARBOL` (agent-multi `c64ab10e`), `$DATOS` (almacen admitido EURUSD H1).

## 1. Dictamen global

**ACEPTADO CON CORRECCIONES OBLIGATORIAS.**

Aplicando el formato del informe 85, punto 5, los tres dictamenes separados son:

| Objeto | Dictamen | Fundamento resumido |
|---|---|---|
| Contrato de ejecucion (cinco comandos, mascara, TP/SL/volumen, cronologia, costes, recompensa) | **ACEPTADO** | O01-O07 verificados linea a linea contra el adaptador y el motor fijado; 22 testigos y 6 ondas reproducidos exactamente (seccion 5). |
| Reproducibilidad de las simulaciones (81 replays, 162+6 sinteticas, 9 busquedas DEAP, 66 confirmaciones) | **ACEPTADO** | 329+31+3645 hashes releidos sin discrepancia; control midpoint identico al padre en 3/3 semillas; nueve busquedas y 66 confirmaciones reproducidas; tres genomas re-simulados con el motor bit a bit; recibo alterado rechazado. |
| Admisibilidad de los resultados de seleccion y de la comparacion DEAP como base de decision | **RECHAZADO** | (a) La regla fija pierde por construccion: RR 0.916 con tasa de acierto 50.1% frente a un umbral de equilibrio de 54.4%; (b) 55 de 59 recibos TCN del carril son byte a byte la regla ciega "abrir largo cuando se permite"; la politica aprendida no aporta nada medible; (c) el informe 89 omite que los seis genomas optimizados son peores que ambos controles no optimizados en D/E, y que los genes de SL son inertes; (d) no existe control positivo que demuestre que PPO+TCN puede explotar una senal conocida; (e) INNER_A es a la vez bloque de seleccion y unico bloque "fuera de muestra" informado. |

El dictamen global es ACEPTADO CON CORRECCIONES OBLIGATORIAS y no RECHAZADO porque
los informes no reclaman rentabilidad, declaran `NO_FEASIBLE_CANDIDATE`, y cada cifra
que publican se reprodujo. Lo que falla no es la aritmetica sino lo que se omite y la
base inferencial que los informes dan por cerrada. Las correcciones obligatorias son
las de los hallazgos C1-C4 y M1-M4 (seccion 3). Hasta que se incorporen, ningun numero
de este carril debe citarse como evidencia sobre utilidad de features, sobre el termino
de fitness preferible, ni sobre la calibracion de TP/SL.

## 2. Tabla O01-O18

Veredicto tipado con exactamente una de: VERIFIED / UNVERIFIED[razon] / REFUTED[contraevidencia].
Resumen: **17 VERIFIED, 1 REFUTED, 0 UNVERIFIED.**

| ID | Veredicto | Evidencia propia (archivo:linea, comando o numero) |
|---|---|---|
| O01 | VERIFIED | `agent_plugins/hybrid_orders.py:82` y `:136` `nn.Linear(width,5)`; `:22` `masked_fill(~mask,-inf)`; `:41-47` rechazo `COMMAND_CODEC`/`ILLEGAL_COMMAND` si `actions[:,0]!=long()`; `env_plugins/hybrid_orders.py:74-75` `ILLEGAL_CATEGORICAL_COMMAND`; `:29` argmax si `deterministic`, `sample()` si no; `tools/recipe_sac_queue.py:267` evalua con `deterministic=True`. Testigo `invalid` (1.2) reproducido sin trade (seccion 5, O08). |
| O02 | VERIFIED | Sigma: `$MOTOR app/short_horizon/zigzag.py:211-217` (24 diferencias de cierre al cuadrado, media 480 min 120, `shift(1)`); rho `:220-223`. Banda: `env_plugins/hybrid_orders.py:39-49` interseccion de `tp_bounds=[0.25,3]`, `sl_bounds=[0.5,3]` (RECIPE) con `(min_distance*(1+2e-9)/sigma, 1.5*rho*(1-2e-9)/sigma)`; midpoint `:84-85`; signo por direccion `$MOTOR env.py:222` `tp,sl=r+d*dist_tp, r-d*dist_sl`. Formula direccional DEAP `app/directional_risk_calibration.py:36-37` `a+f*(b-a)`. Cabeza Beta inactiva: `learned_risk=False` en `tools/rl_subset_bridge.py:21`. Grid y midpoint recomputados desde recibos (seccion 5, ataque A). Inexactitud documental: el adaptador usa margen 2e-9 propio, no "los margenes del motor" (1e-9, `env.py:214`) como dice el informe 87. |
| O03 | VERIFIED | `$MOTOR env.py:322-326` decide tras b y simula k=b+1; fill `:376` a `o[k]`; niveles desde `c[b]` `:209` y `sigma[b+1]` `:182-184` (serie desplazada); observacion = filas `<= b` (`env_plugins/recipe_native.py:54-59`, `app/frozen_recipe_inputs.py:102`). Los datos de k solo entran tras la decision (fill, gap, protecciones). El oraculo sintetico lee `c[k+4]` solo para emitir comandos (`tools/native_strategy_contract_study.py:130-141`) y nunca entra al extractor. |
| O04 | VERIFIED | Mascara ofrece CLOSE con posicion sin consultar la compuerta de entrada (`env_plugins/hybrid_orders.py:54-58`); motor ejecuta el cierre al open sin `entry_gate` (`env.py:340-352`), que solo se llama dentro de `if want_entry` (`:354-355`). Testigos `close_before_SL` (-6.36 > SL -87.27) y `close_before_TP` (-6.36 < TP +68.41) reproducidos. |
| O05 | VERIFIED | NOP flat no opera (testigo `daily_cap`/`missing_window`); HOLD automatico con `cmd=0` mientras `not valid[bar]` conservando protecciones (`env_plugins/hybrid_orders.py:117-120`); contabilidad y financiacion siguen (`env.py:230-249`). Tres ventanas ausentes reproducidas: -25.45424 largo / +0.45454 corto. |
| O06 | VERIFIED | Orden en `env.py step`: protecciones al open `:330-337` (retorna antes del agente) -> cierre voluntario `:350-352` -> entrada `:354-381` -> protecciones intrabar `:383-389` con SL primero (`execution.py:241-258`) -> margen al cierre con rollover `:391-395` -> forced flat `:397-402`. Coincide con 84 seccion 5 y con "protecciones, margen y despues forced flat", no con la metadata abreviada. |
| O07 | VERIFIED | RECIPE `profile`: spread por hora, reapertura 3e-4, slippage 2e-5/5e-5, comision 3.5/lado/lote 1e5, swap -7/+2.5, triple en indice 3, `limit_through` 1e-4. Codigo `env.py:278-284` (medio spread por fill, comision por lado), `:230-238` (swap con signo: corto acredita), `execution.py:247,254` (TP atraviesa 1e-4 simetrico), `env.py:296,312` (gross sin costes; net = gross - entrada - salida - financiacion). Ledger de los 81 replays reconciliado por `ledger_checks`. Omision documental: el fill por TP lleva slippage 0 (`env.py:281`) y el margin close-out se cobra como stop (`:289`); el informe 84 no lo dice. |
| O08 | VERIFIED | Recomputo propio con el motor fijado y la config del RECIPE: **22/22 testigos** y **6/6 ondas** identicos a `SCENARIOS.json` e `IDEAL_WAVES.json` (net, Sharpe, DD, L/S, comandos). Ninguna confusion con ORDER_ALPHA: el estudio no lo lee. |
| O09 | VERIFIED | `$ESTUDIO`: 27 celdas x 3 semillas = 81 `RESULT.json`, periodo TRAIN, `fits=0`, `test_access=False`; `CONTROL_TRADES_EXACT`/`CONTROL_EQUITY_EXACT` afirmados en codigo (`study.py:239-241`) y verificados por sha256 contra el recibo padre en 3/3 semillas; decision recomputada: 0 elegibles, `selected=null`; histograma de razones sobre 81: NO_HOLD_SHORT 81, NO_EXECUTED_CLOSE_LONG 48, NO_SHORT_TRADES 27, NO_EXECUTED_CLOSE_SHORT 27, ACTIVITY_BELOW_TWO 9, NATIVE_INVALID 9. |
| O10 | VERIFIED | Logits una vez por fila TRAIN (`study.py:219-227`), `TapePolicy.predict` aplica la mascara de SU trayectoria (`:153-158`), indice por fila de inputs igual al de la politica viva (`recipe_native.py:58,67`), `LOGIT_SUPPORT` falla cerrado ante NaN. Paridad del control: 3/3. |
| O11 | VERIFIED | Remocion estructural `app/rl_subset.py:24-56` (reindexa grupos, elimina ramas vacias; `grouped_tcn_orders.py:31-35` una rama por grupo restante); bloques seno/coseno `rl_subset.py:114-121`, SPEC: 113 bloques de 120; filas del padre `rl_subset.py:63-64` + `support_hash` (`rl_subset_bridge.py:89`); 21,221 TRAIN / 2,805 INNER_A en SPEC; fit fresco por tarea (`rl_subset_bridge.py:14-24,106`); remocion/reinclusion `feature_selector search.py:47-71`; presupuesto 128, minimo 20, tolerancia 0 (`rl_selection_run.py:21`). |
| O12 | VERIFIED | `feature_selector evaluators.py:90-96`: score = Sharpe INNER_A; `gap=max(0,train-val)`; `max_gap=None` nunca limita; actividad `>=2` solo en INNER_A; `rl_selection_run.py:175-183` media de semillas y `all(eligible)`; `search.py:77-83` `selected=null`. Periodos en RECIPE: TRAIN `[2020-01-01,2023-07-01)`, INNER_A `[2023-07-01,2024-01-01)`; OBS_2024 cargado pero nunca puntuado; TEST filtrado en `rl_selection_run.py:35-37` y `app/frozen_recipe_inputs.py:90`. Hallazgo M1 sobre el significado de estos numeros. |
| O13 | REFUTED[`SCENARIOS.json`: en `SL`, `close_before_SL`, `TP` y `close_before_TP` los stops son IDENTICOS (SL 1.09825, TP 1.1016249); lo que cambia entre casos es la barra futura k=b+1 reescrita (`study.py:61-77`), no "diferentes stops". El testigo no invoca la red: inyecta CLOSE/HOLD directamente (`:88,:92`). Demuestra que con observacion y mascara identicas la decision correcta depende del futuro; NO demuestra por si mismo que la politica ignore su distancia al stop.] | La limitacion de observabilidad (sin vector de estado) es cierta por codigo: observacion = `features` + `action_mask` (`frozen_recipe_inputs.py:117-119`, `recipe_native.py:43-45`, `hybrid_orders.py:51-64`); la mascara filtra "en posicion y direccion" pero no distancia, PnL flotante ni edad. La frase del informe 84 seccion 10.3 debe corregirse. |
| O14 | VERIFIED | Relectura completa: `$ESTUDIO/CLOSE.json` 329 archivos, 0 discrepancias, sha256 `6e395fb2...`; `$SINTETICO` 31/0, `ff642c2c...`; `$DEAP` 3645/0, `a4709827...`; los cinco hashes de `artifacts_sha256` del resumen publico coinciden con los privados. Mutacion: copia de una celda con `net_usd` de un trade +1000 -> `FROZEN_RECIPE_CELL_ARTIFACT` y, por separado, `FROZEN_RECIPE_LEDGER_PROFIT`. Reanudacion DEAP: ver O17. |
| O15 | VERIFIED | `$SINTETICO/GRID.json`: 18 elegibles de 27; 9 no elegibles todos por `NO_EXECUTED_CLOSE_SHORT`; ranking por (Sharpe ideal medio, -DD, -q, TP, SL) da TP=.75, SL=.25, q=.2 con 13.854280 / DD 0.06405 / USD 321.95; las tres fracciones de SL empatan exactamente (13.854280066337676 las tres); midpoint 12.10657 / USD 1423.20. `CONFIRMATION.json` en C reproduce la tabla 9.4 (922.51/22.38477; 4219.83/22.32588; -590.40; -3665.82; 30 AGENT_CLOSE vs 30 TP; 3.75 ordenes/semana). |
| O16 | VERIFIED | `app/directional_risk_calibration.py:46-58`: S, R (neto tras costes / 10000), A (trades/(5*semanas)); F1/F2/F3 con coeficientes 1; no elegible = `-1e9-len(reasons)`; `FINITE_FITNESS` falla cerrado. q=0 -> NOP sin orden ni tasa (`:33-34`; test de integracion ejecutado: pasa). Fitness final de `sharpe_s7` recomputado desde el cache: S=14.935375237118748 = `SEARCH_RESULTS`; `sharpe_return_s7`: F2=15.112131052399823 exacto; F3=15.966297719066489 exacto. Tabla D/E del informe 89 reproducida a la precision impresa desde `CONFIRMATION.json`. Confirmacion posterior a las nueve busquedas por marca de tiempo (ultimo STATE 09:13:41.04; primera confirmacion 09:13:41.25). Hallazgo C3 sobre lo que la tabla omite. |
| O17 | VERIFIED | Intercambio en copia de `searches/sharpe_s7/STATE.json` con `sharpe_return_s7/STATE.json`: el auditor `tools.audit_deap_output_calibration` (commit `29506fbc`, sha `0c35508c...` = `AUDIT.json.auditor_sha256`) falla con `FROZEN_RECIPE_OBJECTIVE_SCORE`; el productor v1 congelado adopta silenciosamente el estado ajeno (best_score 15.1121 de F2 dentro de la carpeta F1), confirmando la debilidad declarada; v2 rechaza por `ARTIFACT_IDENTITY`. Auditoria completa re-ejecutada en copia: PASS, 3611 genomas, 433,320 trades, 66 confirmaciones. Las nueve trayectorias reproducidas desde el cache (poblacion, scores, RNG, historia). |
| O18 | VERIFIED | `orders_per_day` = 0.8541666666666667 en los 3611 recibos = (35/40 + 25/30)/2; F3 - F2 = 0.854166666666666 en todas las generaciones y poblaciones finales; genomas F2 y F3 identicos por semilla. Mecanismo: cupo diario 1 + oraculo que abre en la primera barra admisible de cada dia (seno de periodo 24) -> 35 entradas en A y 25 en B para cualquier genoma; solo q=0 bajaria la cuenta y nunca fue muestreado (q-L minimo 0.00252). |

## 3. Hallazgos, de mayor a menor gravedad

### CRITICOS

**C1. La regla fija de TP/SL/volumen pierde por construccion; el informe no lo calcula.**
- Afirmacion: el control "TP=.5, SL=.5, volumen=1" se presenta como punto medio simetrico y el barrido como "optimizacion exhaustiva de ejecucion" (84 seccion 9.3).
- Evidencia (ataque A, 81 recibos sellados, `attack_a.py`): la banda TP es `[0.25,U]` y la banda SL `[0.5,U]`, de modo que los puntos medios NO son simetricos: en el control la distancia TP media es 68.2 pips y la SL media 74.5 pips, RR = 0.916. Coste medio por trade 3.06 pips (financiacion incluida). Tasa de acierto de equilibrio `p* = (SL+c)/(TP+SL)` = 0.5436; tasa realizada 0.5010 (s7), 0.5030 (s19/s43). En 72 de 81 celdas la tasa realizada queda por debajo de su equilibrio; en las 9 restantes (TP .5/.75 con SL .25) la diferencia es menor a 0.5 puntos y el neto sigue negativo por los cierres FORCED_FLAT. En el control s7 la financiacion es USD 2,135.76 de una perdida de USD 5,869.59 (36%): el swap largo de -7 USD/lote/noche sobre una politica 99.4% larga con 31.9 h medias de tenencia es por si solo un drenaje de 4.30 USD por trade.
- Impacto: el `NO_FEASIBLE_CANDIDATE` es correcto, pero la frase "esos cambios no resuelven el comportamiento" (84 seccion 10.4) oculta que ninguna configuracion del grid puede ser rentable con acierto 50% porque todas tienen RR < 1/p* tras costes. No afecta a la reproducibilidad.
- Correccion obligatoria: publicar por celda RR efectivo, coste por trade en pips y tasa de equilibrio junto a la tasa realizada; declarar que las bandas `[0.25,3]` y `[0.5,3]` hacen el "midpoint" asimetrico; separar la financiacion del resto de costes en toda tabla de resultados. Prueba: la tabla debe regenerarse desde `TRAIN.trades.json` y la tasa de equilibrio debe aparecer en `GRID.json`.

**C2. Las politicas "aprendidas" son, en 55 de 59 recibos TCN, la regla ciega "abrir largo cuando se permite y mantener"; el carril no mide aprendizaje y su ranking es degenerado.**
- Afirmacion: 84 seccion 10.2 limita la coincidencia a "seed7 ... en INNER_A"; 83 hallazgo 2 dice "TRAIN is not identical; do not extend the equivalence beyond the measured period".
- Evidencia: (i) ataque B (`attack_b.py`, motor fijado, TRAIN real): la regla LONG_HOLD produce net -5507.826148833138, Sharpe -0.5078757855905959, 493 trades, 178 TP / 156 SL / 159 FORCED_FLAT, exactamente el control s19 de `$ESTUDIO` y de `$ESPEJO`; (ii) hash de `TRAIN.trades.json` de la regla ciega contra los 65 recibos del espejo: identico en 55/59 recibos tcn_configured en TRAIN y en 58/59 en INNER_A; (iii) en el espejo, 53 de 57 fits de subconjuntos tienen la MISMA firma (493 493/0 -5507.8 | 65 65/0 -1265.3); 17 de 18 subconjuntos one-out tienen el score INNER_A identico -0.8649941479172091, de modo que el floating esta siendo dirigido solo por el desempate `-gap`, que a su vez lo fijan 1-3 cortos aleatorios de una semilla en TRAIN; (iv) cinco politicas moneda-al-aire con la misma regla fija dan netos entre -7,474 y -263 (media -4,523), rango dentro del cual caen las politicas aprendidas (-5,508 a -5,870): no hay separacion respecto del azar; (v) las semillas zero_cost 7 y 19 produjeron politicas de 888 trades byte a byte identicas.
- Impacto: cualquier conclusion sobre utilidad marginal de features derivada de este carril es nula; el "colapso de acciones" no es un hallazgo lateral sino el estado de todo el experimento. No afecta al contrato ni a la reproducibilidad.
- Correccion obligatoria: (a) ejecutar el comparador ciego LONG_HOLD/SHORT_HOLD/moneda sobre TODOS los recibos (no solo seed 7) y publicar la fraccion identica; (b) declarar en 83/84 que la equivalencia cubre TRAIN en 55/59; (c) no publicar ningun orden de poda como evidencia de utilidad; (d) incorporar el control positivo de C4 antes de reanudar la seleccion.

**C3. El informe 89 omite que los seis genomas DEAP son peores que ambos controles en D/E y que los genes de SL son inertes; la comparacion de fitness se hace sobre una meseta plana.**
- Afirmacion: "Adding return changed the chosen exposure and delivered higher synthetic profit" y la tabla de medias F1/F2/F3 (89, Measured results).
- Evidencia (`CONFIRMATION.json`, componentes recomputados; `sensitivity.py` con el motor fijado): en D/E el midpoint `[.5,.5,1]*2` da Sharpe 12.777314909849562 y retorno 0.2086, el control de grid `[.75,.25,.2]*2` 12.494 / 0.037, mientras los ganadores F1 dan 8.79/8.94/8.83 y los F2=F3 8.83/9.18/8.81. En A/B el midpoint ocupa el puesto 3573 de 3611 por S (12.107 frente a ~14.98 de los ganadores): la ganancia de +2.9 Sharpe en calibracion se invierte en -3.9 Sharpe y -10 puntos de retorno fuera de los tapes. Mecanismo: la busqueda lleva q-largo a ~1 (0.99998, 0.99978, 0.99989 en tres finales) porque en A los largos del oraculo ganan 35/35; en D los largos ganan 1 de 32 (el propio 89 informa "22.5% profitable"). Ademas: suma de `n_close_SL` sobre 3611 x 2 ondas ideales = **0**; poner SL-L o SL-S en 0.0 o 1.0 alrededor del ganador `sharpe_return_s7` deja S, R, A, elegibilidad y cada (entrada, salida, direccion, razon) bit a bit identicos. Las magnitudes: S~15, R~0.13, A=0.854; la diferencia de S entre ganadores F1 y F2 es 0.023 (0.15%), menor que la diferencia de R (0.025): el "cambio de exposicion" lo decide un termino que pesa <1% del fitness sobre una superficie plana. Y la busqueda nunca muestreo q=0 (minimos 0.00252 / 0.0665), por lo que "q=0 no abre" solo esta probado por el test de integracion, no por la busqueda. Con q-L=0 o q-S=0 el fitness sube a 19.9 / 19.3 pero la elegibilidad lo rechaza: es la restriccion, no el fitness, lo que impide soluciones unilaterales.
- Impacto: la unica conclusion comparativa del informe 89 ("Sharpe+retorno dio mayor beneficio") no es una conclusion sobre el fitness sino sobre el sobreajuste de la exposicion direccional a dos tapes; `SUMMARY.json` contiene las filas de control pero el texto no. No afecta a la reproduccion exacta de las nueve busquedas, que es correcta.
- Correccion obligatoria: anadir a la tabla de 89 las filas de los dos controles en D/E; declarar que SL-L y SL-S no estan identificados (0 toques) y retirar toda cifra de SL de los "parametros calibrados"; publicar las magnitudes S/R/A y la diferencia de S entre brazos; declarar la busqueda como "ajuste de q-largo, q-corto y fraccion TP sobre un oraculo con entradas saturadas". Prueba: `COMPARISON.json` regenerado con controles y un test que falle si un gen nunca afecta a ningun trade.

**C4. No existe control positivo a traves de PPO+GroupedTCN; nada certifica que el carril pueda explotar una senal conocida.**
- Evidencia: `CONTROLS.json` del espejo solo contiene `tcn_configured`, `tcn_zero_cost`, `cnn_configured` (fits reales). Busqueda en `app/`, `tools/`, `agent_plugins/`, `env_plugins/`, `tests/` de "cheat|planted|positive control|known signal": sin resultados pertinentes. Las ondas de 84 seccion 9.2 y las 21,666 trayectorias DEAP alimentan comandos del oraculo directamente a `env.step` (`study.py:131-141`) sobre ventanas de ceros: la red nunca se invoca. La admision (`rl_selection_run.py:195-218`) entrena 64 pasos sobre observaciones constantes y solo exige pesos finitos. Por el contrario, mi oraculo de 4 barras SI pasa la compuerta conductual estricta sobre TRAIN real (888 trades, 492/396, 803 AGENT_CLOSE, elegible=True): la compuerta es satisfacible por una senal; falta demostrar que el entrenador la aprenderia.
- Impacto: el colapso a "largo siempre" no se puede distinguir de "PPO con este presupuesto no aprende nada aqui". Afecta a toda interpretacion de 83/84 seccion 10; no afecta al contrato.
- Correccion obligatoria: un control con feature plantada derivada del futuro (p. ej. signo de `c[k+4]-o[k]` como columna 121, o reemplazo de una columna) entrenado con el MISMO `build_model`, presupuesto y evaluacion; debe producir ambas direcciones, AGENT_CLOSE y Sharpe INNER_A positivo. Sin ese control, el carril no puede atribuir ningun resultado a los datos.

### MAYORES

**M1. INNER_A es el bloque de seleccion y a la vez el unico bloque "fuera de muestra" informado; todos sus numeros son de seleccion, no de confirmacion.**
- Evidencia: `evaluators.py:91-95` puntua y gatea actividad solo en INNER_A; `rl_selection_run.py:122-130` aplica la compuerta en `selection`; OBS_2024 esta cargado (6,208 filas disponibles en SPEC) pero `rl_subset_bridge.py:122` evalua solo `[('train','TRAIN'),('selection','INNER_A')]`. INNER_A ha sido leido en 65 `RESULT.json` + 9 replays cruzados; 19 subconjuntos puntuados sobre el. El propio RECIPE declara `internal_selection_bias: "candidate pool used 2023; INNER_A is development, not independent confirmation"`.
- Impacto: cifras como -0.86499415 / 65 ordenes / USD -1,265.29 no son rendimiento fuera de muestra. 84 seccion 8 lo insinua ("esta seleccion no puntua 2024") pero no lo declara como propiedad de cada numero publicado.
- Correccion: etiquetar todo valor INNER_A como "bloque de seleccion"; reservar OBS_2024 para una unica lectura final tras cerrar la seleccion y declararla antes.

**M2. La compuerta conductual de la campana viva cuenta comandos solicitados, no cierres ejecutados; la actividad >= 2 se cumple mecanicamente.**
- Evidencia: `tools/recipe_hybrid_queue.py:249-257` usa `order_counts` (etiquetas del comando, `env_plugins/hybrid_orders.py:99`) mientras la regla estricta vive solo en `app/native_strategy_contract.py:44-63`; los recibos historicos no fueron re-gateados (84 seccion 10.1 lo admite). Con cupo 1/dia y una regla "abrir largo cuando se permite", la actividad observada es 2.41-2.72/semana: el umbral 2 no discrimina. Mi regla ciega LONG_HOLD cumple la actividad en TRAIN e INNER_A.
- Correccion: re-gatear los 65 recibos con `executed_behavior` y publicar cuantos cambian; subir o re-definir la actividad en terminos que una regla diaria ciega no satisfaga (p. ej. exigir cierres voluntarios ejecutados por direccion, que ya existe en la regla estricta).

**M3. El testigo de informacion de posicion esta mal descrito (ver O13).** Correccion: reescribir 84 seccion 10.3 como "observacion y stops identicos, futuro distinto"; si se quiere demostrar la ceguera a la distancia al stop, construir el testigo complementario (futuro fijo, distancia variable) y pasarlo por la red, no solo por el motor.

**M4. Las conclusiones del informe 83 sobre alcance ("TRAIN is not identical") y del 84 seccion 10.2 ("seed7") subestiman la cobertura de la regla ciega.** Evidencia en C2. Correccion: actualizar ambos con la fraccion 55/59 y 58/59 y hacer del comparador ciego el control por defecto de cada recibo (84 seccion 10 ya lo recomienda "antes de atribuir ganancias"; debe ser obligatorio ahora).

### MENORES

**m1.** Informe 87: "fixed random seed 20261026" en el protocolo de busqueda frente a las semillas 7/19/43 ejecutadas; contradiccion interna, prevalecio la segunda.
**m2.** Informe 87: "retaining the engine's float-safe margins": el adaptador usa 2e-9 propio (`env_plugins/hybrid_orders.py:46-47`), el motor 1e-9 (`env.py:214`). Inocuo numericamente.
**m3.** Informe 84 seccion 7 no declara que el fill por TP lleva slippage 0 ni que el margin close-out se cobra como stop.
**m4.** Informe 87 exige "unique evaluations" por busqueda; `RESULTS.md` publica "Evaluation requests" (incluyen cache y duplicados). Unicos por busqueda recomputados: sharpe 620/491/454, sharpe_return 818/530/864 (s7/s19/s43); F3 identicos a F2; union 3611.
**m5.** Informe 89: "Rather than repeat numerical experiments" es exacto: la auditoria de Musashi nunca re-simulo las 21,666 trayectorias de calibracion (solo consistencia de ledger y replay de scores en cache). Esta auditoria re-simulo tres genomas (18 trayectorias) con el motor fijado: bit a bit identicos al cache. Deberia decirse en 89 cual es el alcance de "exact replay".
**m6.** Sharpe del motor (`execution.py:432-453`): equity diaria en dias de calendario con fines de semana rellenados, ddof=1, sqrt(365). Factor frente a 252 dias habiles ~1.02; no material, pero no es el "Sharpe" estandar y la etiqueta deberia decirlo. Ruina -> NaN -> `None` -> subconjunto "no medido" en vez de penalizado.
**m7.** `DirectionalOrders` con q=0 re-etiqueta un OPEN como NOP en `order_counts` (`directional_risk_calibration.py:34`), inflando NOP para `executed_behavior`; cosmetico porque esos genomas fallan por direccion.
**m8.** `app/order_rewards.py:16-17` lanza excepcion si la equity es <= 0 en vez de terminar el episodio; inalcanzable en la practica con SL en toda orden y close-out al 50%, pero un fit abortaria en vez de registrar ruina.
**m9.** Informe 84 seccion 9.2 dice "B / ideal ... 24/1" y 20% de aciertos: verificado; pero la onda B no ejercita CLOSE_SHORT y el texto lo admite. Sin accion.
**m10.** Consistencia entre documentos (ataque H): 84/85/89/90 coinciden en 3611/21666/66, 8.85595/885.33/8.94266/1094.65, 12.10657->13.85428, 1423.20->321.95, 43+57=100, 52, 56, 32 y 47 pruebas (recibos XML con esas cuentas exactas, 0 omitidas). Unica discrepancia numerica entre documentos: m1.

## 4. Lo que hizo bien, y como se comprobo

1. **Contrato de ejecucion correcto y completo.** Cinco clases categoricas con mascara `-inf`, rechazo de comandos fraccionarios en dos capas, SL y TP obligatorios en cada orden (`_levels` es la unica ruta a `Position`; `check_order` exige ambos finitos y a distancia minima), unidades enteras por `floor`, `rel_volume` 0.1 del balance realizado actual, apalancamiento 50, q en `(0,1]` sin elevacion al minimo, q=0 -> NOP sin tasa. Comprobado leyendo adaptador y motor fijado (hashes de los 56 archivos del motor y de los archivos agente iguales a `SPEC.code`), y por los 22 testigos reproducidos.
2. **Causalidad limpia.** Observacion con filas `<= b`, sigma desplazada, fill en `o[b+1]`, protecciones del gap con prioridad sobre el agente, SL primero en barra ambigua, forced flat 12 h antes del cierre, 12 h sin abrir tras reapertura, cupo 1/dia contando solo fills. Comprobado linea a linea (O03-O06).
3. **Reproducibilidad ejemplar.** 329 + 31 + 3645 hashes releidos sin discrepancia; control midpoint identico al padre (sha256 de trades y equity, 3/3); 22 testigos y 6 ondas recomputados identicos; fitness finales de tres busquedas recomputados desde el cache con igualdad exacta; nueve busquedas reproducidas desde semilla; 66 confirmaciones reproducidas; tres genomas re-simulados con el motor con igualdad bit a bit; recibo alterado rechazado por dos verificadores independientes; intercambio de checkpoints rechazado por el auditor v2.
4. **Honestidad del resultado principal.** `NO_FEASIBLE_CANDIDATE` recomputado: 0 de 27 elegibles en las tres semillas; `selected=null`; no se leyo INNER_A ni 2024 para el barrido (`IDENTITY.period='TRAIN'`, `financial_validation_read=False`); no se forzaron operaciones; no se tocaron costes, apalancamiento ni calendario.
5. **Separacion oraculo/aprendizaje declarada.** Cada tabla sintetica lleva la advertencia; los D/E se leyeron tras las nueve busquedas (marcas de tiempo) y no seleccionan nada.
6. **La debilidad v1 de reanudacion fue encontrada y corregida por el, no ocultada**, y su afirmacion de que no contamino resultados se sostiene: la auditoria completa re-ejecutada en copia da PASS con las mismas cuentas.
7. **Anexo 90.** La validacion tipada de entradas se verifico sobre los 81 recibos (`ELIGIBILITY_IMPACT.json`: 0 decisiones cambiadas) y los 21,732 registros; 32 pruebas en recibo XML.

## 5. Registro de reproduccion

Entorno: CPU, un hilo, `CUDA_VISIBLE_DEVICES=''`, `PYTHONDONTWRITEBYTECODE=1`, interprete del indice privado, `PYTHONPATH=$ARBOL`. Computos pesados bajo `crispdm-run -q -m 2G -t 1200s -n audit-m -- python -I ...`. Scripts en el area de trabajo del auditor (no se publican rutas): `attack_a.py`, `attack_b.py`, `o08.py`, `o08_o14_o15.py`, `o15.py`, y los del subauditor DEAP `scan_cache.py`, `confirm.py`, `states.py`, `adopt.py`, `sensitivity.py`.

Pines verificados:
- agent-multi worktree HEAD `c64ab10e1691112600a522b786dfef9d7d9748b3` = `origin/codex/rl-static-temporal-20261018`.
- heuristic-strategy export HEAD `cd6489fb1f65ec6cd9bc7f8e0ce4860d6d39d4e9`; 56 archivos = `SPEC.code.engine`.
- feature_selector `bec66cc`: 13 archivos = `SPEC.code.selector`.
- SPEC sha256 `05c06cb508b3e7bb9e1c33064c9dda4cd524671db62dd78d7defc5539da20f4d`; RECIPE sha256 `de4d8cae...` (= `IDENTITY.json`).
- Auditor DEAP `tools/audit_deap_output_calibration.py` en `29506fbc`: sha256 `0c35508c...` = `AUDIT.json.auditor_sha256`, byte a byte igual al de HEAD.

Hashes de cierre releidos (archivo a archivo):
- `$ESTUDIO/CLOSE.json` `6e395fb29f5594d062a18539dfd50142a6795196835d2480d94ec728ad1da160` (329, 0 fallos).
- `$SINTETICO/CLOSE.json` `ff642c2c5c9be017fca9fdcd0cad8e1721d8b674eced0e34fac20d940f8aaff3` (31, 0).
- `$DEAP/CLOSE.json` `a4709827e15d03539a3459d9980870d75b0dd8e53ddb1d552881a9fda0f58d78` (3645, 0).
- `AUDIT.json` `28a34c36...` (estado PASS; re-ejecucion en copia: PASS, identidad `c5623a6b...`).

Ataque A (`attack_a.py`, lee 81 `TRAIN.trades.json` sellados): tabla por celda de RR, pips TP/SL, coste/trade, `p*` y tasa realizada; control: RR 0.916, TP 68.2 / SL 74.5 pips, coste 3.06 pips, `p*`=0.5436, realizada 0.5010/0.5030/0.5030; 72/81 celdas por debajo de su equilibrio; control s7 por razon de cierre: SL 154 (-37,254.6), FORCED_FLAT 156 (-3,823.0), TP 175 (+34,758.2), AGENT_CLOSE 12 (+449.7).

Ataque B (`attack_b.py`, `SubsetMarket` + `CalibratedMarket` con `RiskParameters()` por defecto, TRAIN real, motor fijado, exit 0):

| Politica | Neto USD | Gross | Sharpe | Trades L/S | AGENT/TP/SL/FF | Financiacion | Elegible |
|---|---:|---:|---:|---|---|---:|---|
| LONG_HOLD (ciega) | -5507.83 | -412.91 | -0.5079 | 493/0 | 0/178/156/159 | 2242.41 | No |
| SHORT_HOLD (ciega) | -4682.93 | -2875.88 | -0.3684 | 0/492 | 0/174/160/158 | -719.67 | No |
| Moneda 0..4 | -7474.09 / -2077.14 / -263.02 / -5959.77 / -6840.71 | -4848 / +1961 / +3134 / -2844 / -3407 | -0.98 / -0.03 / +0.15 / -0.61 / -0.81 | ~250/~250 | 0 AGENT_CLOSE | 511-916 | No |
| Oraculo 4 barras (regla 9.2) | +10,736,242 | +12,128,155 | 12.08 | 492/396 | 803/68/0/17 | 6237.64 | **Si** |
| Oraculo 24 barras | +209,305,812 | +226,368,928 | 11.51 | 435/410 | 542/190/4/109 | 629,723 | Si |
| Oraculo solo direccion, HOLD | +19,527,447 | +20,709,930 | 7.00 | 254/254 | 0/300/52/156 | 327,613 | No |

Lectura: LONG_HOLD == control s19 (y 55/59 recibos TCN); las politicas aprendidas (-5,508 a -5,870) quedan dentro del rango de la moneda; el oraculo demuestra que la compuerta estricta es satisfacible y que los cierres voluntarios en ambas direcciones son alcanzables por una senal; las cifras del oraculo son de composicion con apalancamiento sobre el futuro conocido y no significan nada financiero.

O08: `witnesses(cfg,shape)` y `oracle(cfg,shape,.002,8,0)+oracle(cfg,shape,.0015,6,3)` con `EnvConfig` construido desde `RECIPE.business.evaluation_and_policy`: 22/22 y 6/6 identicos (tolerancia 1e-9 en net/Sharpe/DD, igualdad en L/S y comandos).

O14 mutacion: copia de la celda `s7_014f9c611b50a60a`, `TRAIN.trades.json[0].net_usd += 1000` -> `ValueError FROZEN_RECIPE_CELL_ARTIFACT` (comprobacion de adopcion del estudio) y `ValueError FROZEN_RECIPE_LEDGER_PROFIT` (`ledger_checks`).

O15: ranking recomputado sobre `GRID.json` (18 elegibles; 9 `NO_EXECUTED_CLOSE_SHORT`); ganador y empate triple de SL exactos; `DECISION.json` estado `SYNTHETIC_ONLY_SELECTED`, parametros `(.75,.25,.2)`.

O16/O17/O18 (subauditor DEAP, copias en area de trabajo): recomputo de S/R/A por recibo (3611, 0 discrepancias; `percentual_order_size`==q en 433,320 trades; sum `net_usd` == metrica con |dif| max 1.7e-16); generaciones por parada PATIENCE: sharpe 50/48/40, sharpe_return 66/42/71, activity 66/42/71; poblacion inicial regenerada con `random.seed(seed)` + controles `[.5,.5,1]*2`, `[.75,.25,.2]*2`, `[.75,.25,1]*2`, presente en el cache para las tres semillas; `F3-F2` constante; auditoria completa `--require-complete` en copia sin alterar: PASS; con `STATE.json` intercambiados: FAIL `FROZEN_RECIPE_OBJECTIVE_SCORE`; productor v1 en copia intercambiada: adopta (best_score 15.112131052399823 en carpeta F1). Sensibilidad alrededor de `sharpe_return_s7` (18 trayectorias por punto): SL-L/SL-S en {0,1} -> dS = +0.000e+00, trades identicos; q-L=0 -> S 19.887 no elegible (`DIRECTION_0`); q-S=0 -> S 19.257, R 0.335, no elegible.

Pruebas ejecutadas: `tests/integration/test_native_strategy_witness.py -k exact_frozen` (1 passed, 2.44 s, motor exportado y espejo como raiz); `tests/unit/test_directional_deap_calibration.py`, `tests/unit/test_deap_execution_audit.py`, `tests/integration/test_directional_deap_outputs.py` (9 passed con motor y espejo). Recibos XML del cierre: pytest 17, witness 3, final 43, engine 57, synthetic-extension 47, deap 52, deap-audit 56, eligibility 32; 0 fallos, 0 omitidas en todos.

Comprobacion de no mutacion: `git status` del worktree limpio salvo este documento; `find -newermt` sobre `$ESTUDIO`, `$SINTETICO`, `$DEAP`, `$ESPEJO`, `$MOTOR` y el runtime DEAP congelado: 0 archivos tocados.

## 6. Mediciones afectadas y no afectadas por cada hallazgo

| Hallazgo | Afecta | No afecta |
|---|---|---|
| C1 | Interpretacion del barrido 27x3 y de "riesgo fijo" (84 seccion 10.4) | Reproducibilidad de los 81 replays; contrato |
| C2 | Toda conclusion de utilidad de features del carril; 83 hallazgos 1-2; 84 seccion 10.2 | Contrato; los recibos como mediciones de su politica congelada |
| C3 | Conclusion comparativa F1/F2/F3 del 89; valores de SL "calibrados"; el ganador sintetico 9.4 como punto de partida | Reproduccion exacta de las nueve busquedas y 66 confirmaciones; O17 |
| C4 | Atribucion causal de cualquier resultado del carril a los datos o a TCN | Contrato; reproducibilidad |
| M1 | Etiqueta de todas las cifras INNER_A | Cifras TRAIN del barrido |
| M2 | Compuerta de los 65 recibos historicos | Los 81 replays (ya usan la regla estricta) |
| M3/M4 | Texto de 83/84 | Ningun numero |

Fin del dictamen.

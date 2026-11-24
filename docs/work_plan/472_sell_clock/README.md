# 472. Cuanto del resultado reproduce vender siempre a 48 h

EURUSD, desarrollo 2023, campeon de contexto de 471 sellado. No se optimizaron ni seleccionaron parametros. Se replicaron las cuentas semestrales con el perfil de costes y la ranura unica de 471. No se evaluo 2024.

| Cuenta / control | H1 neto | H2 neto | Ordenes H1/H2 |
|---|---:|---:|---:|
| Selector original | -1.70% | +12.77% | 31 / 29 |
| SELL 48 h en sus mismos instantes y tamanos | -1.45% | +12.66% | 31 / 29 |
| SELL 48 h, permutacion comparable, mediana | -2.59% | +14.16% | 31 / 29 exactas |
| SELL 48 h, reloj probabilistico aprendido en TRAIN, mediana | -1.07% | +10.57% | 31.9 / 28.0 medias |

**Misma entrada.** Se reemplaza el lado y horizonte de cada propuesta ejecutada por SELL 48 h. Barreras relativas del campeon, ATR conocido al entrar, margen y calendario intactos. El replay rechaza solapamientos: en estos paneles no tuvo que saltarse ninguna. En H2 el cambio cuesta solo 0.11 puntos porcentuales; en H1 mejora 0.24. No es una estrategia independiente porque conserva el instante y el volumen que eligio el modelo.

**Control temporal emparejado.** Se permutan plantillas de semanas (ancla lunes UTC, bloque del mes de su jueves), solo si quedan en el mismo mes de entrada y dentro de horas habilitadas. Se conservan dia y hora UTC, todos los tamanos q y el numero de trades, barreras relativas y costes; se excluyen colisiones verificando la barra de salida. Se elige uniformemente entre permutaciones semanales factibles dentro de cada bloque, con 2048 combinaciones sorteadas por semestre. Esta condicion usa el libro retrospectivo y no constituye una politica de produccion. Comprobacion secundaria: todas las combinaciones quedaron dentro de +/-5% de la exposicion temporal del selector, medida en suma de q por barras H1 ocupadas (no minutos exactos).

**Cuanta informacion tiene el control.** En H2 se desplazan, en promedio, 12 de 29 entradas; la ventana central de retornos sorteados es [+12.66%, +14.29%]. Hubo solamente **cuatro** retornos distintos redondeados a ocho decimales. Reasignar semanas con ordenes identicas no cambia el libro economicamente. El 75.9% de sorteos iguala o supera al selector; no es un valor p ni un intervalo sobre beneficios futuros. En H1 hay 216 retornos distintos y la ventana central va de -8.06% a +5.85%.

**Control ejecutable sin senales del selector en 2023.** Solo usa la frecuencia historica de aperturas sobre oportunidades libres de 2020-22 por hora de la semana y su q medio (un pseudo-conteo con la media global). En cada tick elegible sortea entrada con esa probabilidad y vende a 48 h si esta libre, con q del horario. En H2 el intervalo central entre semillas es [+2.00%, +19.87%]. No esta emparejado por exposicion: media 922 q-barras contra 1495 en el selector. Sus ganancias no son evidencia prospectiva ni implican que sea necesario un generador aleatorio para operar.

**Salida observada.** En H2, 24 de las 29 entradas del selector fueron martes y 24 de las 29 salidas ocurrieron por calendario, 3 por SL y 2 por TP. El clon SELL a las mismas horas tuvo 25 por calendario, 3 SL y 1 TP. H1: 25/31 cierres obligatorios en ambos. Por tanto el resultado no demuestra una capacidad especial de apuntar al TP o escoger el horizonte. Tampoco demuestra que vender 48 h sea rentable en general.

**Decision:** no promocionar basandose en H2, ni reoptimizar aperturas con este panel. Si se continua con el predictor estacional, aislar su valor en **cierre** desde las mismas posiciones: comparar HOLD hasta barrera/calendario, salida por reloj y cierre con pronosticos, midiendo ganancia incremental neta por posicion y no solo perdidas evitadas.

Desde agent-multi, mismos insumos que 471:

```bash
OPENBLAS_NUM_THREADS=1 PYTHONPATH=. python tools/role472_sell_clock.py "$SRC" "$BARS" "$PREFIX" "$SIGNALS" docs/work_plan/471_net_selector "$OUT"
```

report.json resume los contrastes; accounts.csv.gz contiene los sorteos y su exposicion; train_clock.csv publica el reloj estimado sin 2023; barrier_audit.csv detalla el primer evento y PnL en USD por USD 10000 de referencia para cada propuesta no permutada. Los retornos de cuenta usan composicion, no suma de esos importes.

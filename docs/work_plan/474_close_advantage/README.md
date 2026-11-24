# 474. Aprender la ventaja neta de cerrar ahora

**Decision:** no activar estos cerradores. La rama estacional mejora en H1 de 2023, pero no en los tres bloques de desarrollo previos ni en H2 frente a HOLD. La rama de solo estado pierde frente a HOLD en los cinco bloques. No se consulto el test de 2024, no hubo DEAP ni despliegue.

## Contrato

- Se congelan las aperturas, barreras, volumen y unidades del campeon de contexto de 471. En TRAIN se generan posiciones hipoteticas BUY/SELL a 6/24/48 h en cada origen habilitado, con las mismas reglas de primer evento y calendario. No se reabren posiciones durante la evaluacion y no se acumulan rendimientos de posiciones virtuales solapadas.
- Para cada tick habilitado de una posicion viva, la etiqueta es (PnL neto de cerrar ahora menos PnL neto de mantener hasta la primera barrera o calendario), por unidad monetaria de riesgo inicial al SL. Spread, ambos lados de comision, deslizamiento y financiacion incluidos. Si ya hay gap ejecutable o liquidacion obligatoria no hay decision. Aperturas y precios posteriores solo sirven para fills y etiquetas; las entradas del modelo usan la barra previa cerrada, las predicciones disponibles y el reloj conocido.
- Dos regresiones ridge con intercepto, penalizacion 10, normalizacion solo con TRAIN y un peso total de uno por posicion con decisiones. El primer modelo recibe lado one-hot, horizonte one-hot, distancias a barreras, beneficio firmado y absoluto, edad, tiempo hasta corte, fraccion transcurrida, ATR y spread relativos al riesgo e interaccion beneficio-tiempo. El segundo agrega los ocho pronosticos estacionales alineados con el lado, y sus magnitudes, relativos a la distancia al SL. Las columnas estan enumeradas en los pesos. Se usa solucion directa, sin epocas ni paciencia. Cada modelo toma la **primera** estimacion estrictamente positiva; si no la hay, HOLD.
- 2019-2022: 14431 origenes, 86586 posiciones hipoteticas, 85144 con decisiones. Hay 2379098 pares posicion-tick pero solo 14228 barras de decision distintas. No se interpretan como millones de muestras independientes. Se excluyen posiciones cuyo corte de liquidacion puede cruzar el limite anual. Para evaluar 2020, 2021 y 2022 se usan solo anos anteriores; para **ambos** semestres 2023 se usa el mismo ajuste 2019-22. El encoder estacional y la politica de apertura tienen seleccion historica en este panel: el resultado es de desarrollo, no una validacion independiente del sistema completo.
- Se verifica el PnL pasivo y su barra de salida para las 271 entradas archivadas; se verifican timestamp y etiqueta de todos los 12616 ticks de decision de esas entradas contra 473. Para comparar estrategias se mantienen las unidades originales, no se deriva un tamano alternativo tras cada cierre. El beneficio porcentual usa USD 10000 al inicio de cada bloque. El regret en USD usa el optimo retrospectivo de 473.

## Resultado

| Salida | H1 retorno | H1 tempranos | H2 retorno | H2 tempranos | H2 USD vs HOLD |
|---|---:|---:|---:|---:|---:|
| HOLD | -1.70% | 0/31 | +12.77% | 0/29 | 0 |
| Primer tick disponible | -0.62% | 31/31 | -3.28% | 28/29 | -1604.69 |
| Mitad del tiempo al corte semanal | -5.98% | 29/31 | +8.05% | 24/29 | -471.93 |
| Estado de orden | -3.76% | 30/31 | -1.42% | 26/29 | -1418.83 |
| Estado + estacional | +3.42% | 30/31 | -7.66% | 28/29 | -2043.10 |

H2: solo 28 de 29 posiciones tenian oportunidad de cierre anticipado. El primer tick ya disparo 18 salidas de estado y 21 de estacional. Entre sus salidas ejecutadas el estado predijo +0.020 R, pero se realizo -0.115 R frente a HOLD; con estacional +0.021 R previsto y -0.141 R observado (medias por posicion cerrada). El brazo estacional mejoro 11 cierres, empeoro 16 y empato uno. Por desenlace pasivo, salvo +1535.56 USD en posiciones destinadas a SL, pero sacrifico -647.88 USD en las destinadas a TP y -2930.79 USD en las de liquidacion por calendario. Con solo una compra en H2 no se certifican ambos lados.

H1: anadir la senal al estado aporto +23.17 USD por posicion [intervalo descriptivo de remuestreo semanal -0.12,+58.76]; H2: -21.53 [-60.20,+5.78]. No demuestra ventaja estable ni prueba que la estacionalidad sea inutil para todo posible cerrador.

## Interpretacion y siguiente contraste

La regresion pregunta si cerrar supera a HOLD *pasivo irrevocable*. La politica, en cambio, puede esperar un tick y volver a decidir. Una ventaja diminuta frente a HOLD pasivo no demuestra que cerrar ya sea mejor que conservar esa opcion; ademas los scores de los cierres ejecutados sobreestiman la ganancia realizada en H2. No se ha aislado una causa unica. No retocar el umbral con H2. Para un siguiente ensayo, contrastar contra esperar un tick y seguir una politica de continuacion aprendida en bloques anteriores, evaluada fuera de bloque, con parada obligatoria en barrera/calendario. Mantener las mismas entradas y los controles HOLD/reloj.

## Reproduccion

```bash
OPENBLAS_NUM_THREADS=1 PYTHONPATH=. python tools/role474_close_advantage.py "$SRC" "$BARS" "$PREFIX" "$SIGNALS" docs/work_plan/471_net_selector docs/work_plan/473_exit_value "$OUT"
```

Insumos como en 473. train_counts.json separa origenes, posiciones y ticks por ano; weights.json guarda cada modelo con columnas, escalas y coeficientes. scores.csv.gz contiene estimacion y etiqueta por tick; positions.csv.gz, primera salida por posicion; summary.csv y paired.csv, resultados contra controles e intervalos semanales (10000 remuestreos).

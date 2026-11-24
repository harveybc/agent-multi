# 471. Seleccion por utilidad neta con y sin pronostico estacional

**Decision:** en este contraste las predicciones estacionales no justifican su inclusion para abrir. El resultado positivo de contexto en H2 no basta para promocionar: 28 ventas de 29 operaciones, y H1 negativo para el campeon DEAP. No se entreno otro predictor, no se uso 2024 para ajustar, seleccionar o puntuar la cuenta y no se envio ninguna orden.

## Diseno

Mismos ticks habilitados, ATR causal de 24 barras y mismos costes de 470. Se ensaya un nuevo mecanismo, no un reentrenamiento de la politica 467. Para no filtrar el pronostico a la rama control mediante sus barreras, se ofrecen las mismas seis alternativas a ambas ramas: BUY/SELL x 6/24/48 h. Sin cierre discrecional. El horizonte dimensiona las distancias, no impone tiempo de salida: TP y SL son sus respectivos multiplicadores por ATR por raiz del horizonte; cierre por primera barrera o calendario. Doble toque sin orden intrabar conocida se resuelve conservadoramente por SL. Una ranura, sin reapertura en la vela de cierre, rel_volume=0.1 y leverage=50.

La etiqueta es la ganancia neta de la alternativa expresada en puntos basicos del balance con volumen normalizado 1. Cada oportunidad de TRAIN tiene seis etiquetas por cada par de multiplicadores; NOP vale cero. No son seis operaciones simultaneas en el replay. Cada regresor ridge (penalizacion 10, columnas estandarizadas solo con sus datos de ajuste) predice las seis utilidades a la vez. Contexto usa ciclos de hora/dia semanal, horas hasta liquidacion, log de ATR/precio y su cuadrado, spread y deslizamiento en unidades ATR. Estacional agrega los ocho incrementos pronosticados/ATR y sus valores absolutos, no las etiquetas futuras. Gana la alternativa de mayor utilidad solo si supera el umbral y aventaja suficientemente a la direccion contraria; en otro caso NOP. El volumen q es min(1,(utilidad-umbral)/saturacion)^potencia.

Se ajustan los modelos finales con 2019-2022 completos (3587,3538,3726,3580 oportunidades respectivamente). Para la busqueda, 2019 inicializa el modelo y cada cuenta 2020/21/22 usa un selector entrenado exclusivamente en los anos anteriores. No se cuenta 2019 como rendimiento fuera de muestra. Se purgan ventanas de riesgo que puedan cruzar el corte. H1 2023 (1863 oportunidades) solo monitorea al lider TRAIN por generacion; H2 (1653) se puntua con parametros fijados. Las cuentas arrancan en USD 10000 en cada bloque. El encadenamiento cronologico es del **selector**, no del encoder: el predictor existente vio hasta junio de 2023 y eligio checkpoints con H2. Tampoco es un panel nuevo para el proyecto. El DD informado se mide en cierres realizados, no intrabar ni mark-to-market.

## DEAP

Primero se evaluaron las dos reglas sin optimizar; despues, tres semillas identicas por rama, poblacion 20, 200 generaciones minimas despues de la inicial, maximo 500, paciencia 30 en retorno neto H1 del lider TRAIN. Fitness reproductivo: media de retornos netos anuales, minimo 24 ordenes por ano; el monitor exige 12 por semestre. No se restaura un campeon por producir cero ordenes. Todos alcanzaron la parada por paciencia en la generacion 200. Se selecciona la semilla por monitor, no por confirmacion. El comparador fijo se publica aunque supere a DEAP en el monitor.

| Gen | Rango efectivo | Escala | Fijo | Contexto DEAP | Estacional DEAP |
|---|---|---|---:|---:|---:|
| TP / (ATR sqrt(h)) | 0.5, 1, 1.5 | discreta | 1 | 1.5 | 1 |
| SL / (ATR sqrt(h)) | 0.5, 1, 1.5 | discreta | 1 | 1.5 | 1.5 |
| Umbral NOP, pb | 0-12 | lineal | 0 | 2.929 | 5.723 |
| Ventaja sobre direccion opuesta, pb | 0-10 | lineal | 0 | 5.744 | 1.762 |
| Exceso para volumen pleno, pb | 2-50 | log | 15 | 10.845 | 30.910 |
| Potencia de volumen | 0.5-3 | log | 1 | 2.399 | 2.174 |

## Resultados netos

| Selector | 2020 | 2021 | 2022 | H1 2023 | H2 2023 | Ordenes H2 | Aciertos H2 | DD H2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Contexto fijo | -14.61% | +0.66% | +9.42% | +2.62% | +2.56% | 41 | 23 | 8.49% |
| Contexto DEAP | -7.38% | +10.74% | +7.86% | -1.70% | +12.77% | 29 | 19 | 5.13% |
| Estacional fijo | -30.76% | +6.48% | -42.86% | -14.52% | -3.44% | 51 | 25 | 10.19% |
| Estacional DEAP | +3.83% | +19.87% | -2.27% | -6.17% | -10.61% | 38 | 18 | 14.10% |

Contexto DEAP: 1 compra ganadora y 18/28 ventas ganadoras en H2; 26 de 29 aperturas fueron ventas a 48 h. Estacional DEAP: 0/4 compras y 18/34 ventas ganadoras. La ventaja del control sin estacionalidad sobre la rama estacional se observa aun sin optimizar. No demuestra que debamos operar exclusivamente ventas, ni mide la utilidad de la estacionalidad para el cierre temprano. Siguiente control: venta 48 h comparable por hora, exposicion, calendario y costes, en bloques cronologicos; no seleccionar una subestrategia por H2.

## Reproduccion

Mismo export de senales pre-2024 que en 469. 972 comparaciones de desenlace y PnL con el motor de barreras de 470. Por rama/semilla se guardan genoma, fenotipo, metricas anuales, lider y paciencia por generacion. Los pesos finales de ambos selectores, escalas y libro de cuentas quedan separados de las metricas; los seis indices de accion son BUY 6/24/48 y SELL 6/24/48 en ese orden. El replay independiente del libro verifica conteo y retorno de los 20 paneles finales.

```bash
OPENBLAS_NUM_THREADS=1 PYTHONPATH=. python tools/role471_net_selector.py "$SRC" "$BARS" "$PREFIX" "$SIGNALS" docs/work_plan/471_net_selector/config.json "$OUT"
```

SRC, BARS y SIGNALS como en 469; PREFIX contiene OHLC historico previo y CLOSURES.json como en 464. El barrido se ejecuta en CPU: las etiquetas de los nueve pares de barreras y las predicciones cronologicas se calculan una vez; cada candidato hace su propio replay secuencial de cuenta.

# 477. Contraste cronologico del pronostico estacional

Objetivo: determinar si la senal estacional agrega retorno neto en apertura y si
sirve para salir antes con entradas y unidades fijas. Sin DEAP, dinero real,
reutilizacion del encoder seleccionado en 2024 ni seleccion de un ano ganador.

## Resultado final

Las ocho redes terminaron por paciencia a las 179 epocas, sin nuevo campeon
desde antes de empezar a contar en la epoca 100. El MAE del estacional mejora
frente a persistencia en 32/32 combinaciones de ano y horizonte (reduccion
1.64%-27.92%, sin afirmar que sean 32 observaciones independientes). No es
MAE de precio. La utilidad para operar es menos estable:

| Ano | Contexto neto | Con senal neto | DD realizado con senal | Compras / ventas con senal | Cambio por cierre temprano con senal, USD |
|---|---:|---:|---:|---:|---:|
| 2020 | +23.15% | -0.78% | 28.41% | 85 / 18 | -37.68 |
| 2021 | -11.27% | -8.71% | 16.12% | 52 / 56 | +1344.83 |
| 2022 | +0.67% | -9.95% | 37.92% | 4 / 200 | +244.49 |
| 2023 | -18.08% | +27.56% | 10.70% | 51 / 62 | -2880.35 |

Cada cuenta de apertura arranca con USD 10000 y mantiene cada orden hasta
barrera o calendario. La ultima columna es **otro contraste**, sobre las
mismas aperturas del selector con senal y sin reciclar capital. En 2022 el
control sin senal hace 0 compras y 250 ventas: el sesgo no requiere que falle
el predictor. En 2023, 53 de los 93 cierres discrecionales del selector con
senal fueron perjudiciales. Ninguna regla de apertura o cierre demuestra
mejora consistente; no promocionar a demo ni lanzar DEAP sobre estos anos.

[Cuentas y aciertos por lado](accounts.csv) | [MAE por horizonte](mae.csv) |
[Curvas por epoca](training.png) | [Datos de las curvas](curves.csv.gz) |
[Auditorias por bloque](audits.json) | [Controles independientes](checks.json).
Las lineas verdes en las curvas marcan el campeon y las grises los rollbacks.
Los dos ejes de error son diferentes: TRAIN en unidades estandarizadas,
monitor en unidades originales.

## Requisitos y evidencia

| Contrato | Evidencia |
|---|---|
| Cuatro ventanas moviles de cuatro anos, luego un ano completo de evaluacion | 2016-19 -> 2020, 2017-20 -> 2021, 2018-21 -> 2022, 2019-22 -> 2023. Ultimos seis meses de cada ventana de cuatro anos exclusivamente para monitoreo del predictor y calibracion posterior del selector. Son 42 meses de ajuste del predictor, no 48 meses de gradientes. |
| Reconstruccion, 61 entradas, ventana 13, stride 2 | Desde OHLC, reloj y calendario, Kalman local lineal y GMM del productor original ajustados por bloque. Escalador en barras finitas del ajuste. No se leen los pesos anteriores. Comparacion de 56 columnas no aprendidas en filas consumidas frente al almacen (rtol 1e-7, atol 1e-8), excluyendo de esa comparacion las primeras 1024 barras de la union de fuentes en 2020 porque la fuente archivada reinicia su calentamiento. Las ventanas de entrenamiento no se acortan por esta exclusion. |
| Causalidad | En cada frontera monitor/evaluacion, repetir con OHLC truncado: las 13 x 61 entradas de la primera decision deben coincidir. Para cada origen del panel verificar ventana dentro de sesion y entrada entre +12 h y -12 h. Calendario de investigacion; no se ha auditado disponibilidad historica de cada publicacion del calendario. |
| Etiquetas | Cambios del nivel estacional a 1-6 y 24/48 horas. Comprobar todas las etiquetas elegibles, su hora de destino y una muestra del STL desde el precio con 192 observaciones pasadas, periodo 48, seasonal 13. Purgar etiquetas que maduren en el split siguiente. Nunca usar el objetivo futuro para filtrar las filas de inferencia de trading. |
| Ajuste de los ocho modelos | Dos por ano, mismo encoder agrupado de 22 ramas, TCN, fusion, atencion y representacion de 208; pesos nuevos desde cero. Cabezal lineal de 6 o 2 salidas. Semilla 123, lotes 256 sin reemplazo por epoca; MAE estandarizado por horizonte; AdamW 3e-4, wd .004. El monitor es MAE medio en unidades del estacional. Gracia LR 20, paciencia 20, factor .2, piso 4.8e-7, restaurar pesos y momentos del campeon en cada descenso. Paciencia de parada 80 desde epoca 100, mejora minima 1e-8, maximo 2000. Checkpoint atomico por epoca. |
| Comparacion por pares | Dos selectores ridge de 471 con penalizacion 10. Misma calibracion en el semestre interno y mismos ticks en el ano evaluado. Uno usa contexto; otro agrega los ocho pronosticos y sus magnitudes / ATR. Sin reajustar parametros por retorno. TP y SL = ATR de 24 barras por raiz del horizonte, multiplicador 1. Umbral y margen 0, saturacion 15 pb, exponente de volumen 1. Una ranura, apalancamiento 50, rel_volume .1, mismos costes y liquidaciones. |
| Control del simulador | Reemplazar el pronostico por cero en todo el panel: ambas cuentas deben coincidir. Verificar el ledger por una segunda ruta y reejecutar la salida HOLD de cada posicion. Para el cierre estacional emplear las reglas SL/pivote fijas de 461 sin reinvertir al cerrar antes. El delta en USD es atribucion pareada, no un retorno compuesto de una nueva estrategia. |

La seleccion historica de columnas, arquitectura, calendarios y familia de
estrategia se mantiene por contrato: esta es una comparacion cronologica
condicional del ajuste, no una reconstruccion de las decisiones de investigacion
que se habrian tomado entonces. No se consulto 2024 para medir rentabilidad.
Los archivos fuente existentes incluyen 2024; no se seleccionaron filas de
ese ano para ajustar, monitorear o puntuar.

## Reproduccion

Separar los procesos: el constructor importa los productores del snapshot
en un proceso aislado; el entrenador usa el entorno Torch de agent-multi;
la cuenta y su control usan el entorno CPU con numba y DEAP instalado (no
se invoca el optimizador). Cada bloque y cada par corto/largo es independiente.

```bash
PYTHONPATH=. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m tools.role477_folds "$YEAR" "$SNAPSHOT" "$BUNDLE/RECIPE.json" "$HISTORY397" "$MARKET" "$DATA459" "$OUT/$YEAR"
PYTHONPATH=. python tools/role477_train.py "$OUT/$YEAR" "$BUNDLE"
PYTHONPATH=. OPENBLAS_NUM_THREADS=1 python -m tools.role477_accounts "$OUT/$YEAR" "$HISTORY397" "$MARKET" "$BUNDLE"
PYTHONPATH=. OPENBLAS_NUM_THREADS=1 python -m tools.role477_verify "$OUT" "$HISTORY397" "$MARKET" "$BUNDLE" "$DATA459" "$DOC/checks.json" "$SNAPSHOT" "$PRODUCER_SPEC"
```

El entrenamiento no depende del servicio de trading. Al reejecutar con la
misma identidad se recuperan epoca, optimizador, LR, paciencia, RNG, pesos
y campeon; el recolector no repite cuentas ya terminadas. En esta ejecucion
el destino privado es `/var/tmp/role477/<ano>/`: batch, ajuste de productores,
scaler, dos checkpoints, predicciones, libro de ordenes y cierres alternativos.

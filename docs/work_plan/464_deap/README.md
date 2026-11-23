# 464. Optimizacion del puente estacional, antes del nuevo barrido

Experimento con oraculo estacional, no PnL de predictores. El ensayo 463
caracterizo una estrategia fija, no optimizada. Se conservan sus resultados.
Se congela la estructura aceptada: 6/24/48 h para abrir, 1-5 h cortas,
cierre anticipado por SL, margen maximo 10%, apalancamiento 50. No optimizar
los parametros de pivote ni de tablas de calibracion: no intervienen en este
brazo. No se ha vuelto a generar ruido en este experimento.

## Diseno y verificacion

Cuatro cuentas TRAIN anuales 2019-22 y validacion separada 2023. Cada cuenta
empieza con USD 10000 y usa todas las barras archivadas. ATR con historia
anterior; se descartan del panel las etiquetas cuyo destino toca o cruza
el limite anual. Es una prueba ideal de la politica, no de aprendibilidad
del STL; se mantiene el STL previamente seleccionado. 2024, ya examinado
en el proyecto, se puntua despues de guardar ambos juegos de parametros.

DEAP: poblacion 20 (incluye control), generacion 0 seguida de al menos
200 generaciones, tope 500, un elite, dos inmigrantes aleatorios,
torneo de tres, SBX acotado, mutacion polinomial acotada. Entradas directas
en [0,1]; la tabla en config.json es el contrato de decodificacion. Para un
gen u: lineal = lo+(hi-lo)*u; logaritmico = lo*(hi/lo)^u; eleccion discreta =
valores[min(floor(n*u),n-1)]. Ninguna conversion de flotante a capas u horas
por redondeo posterior. Los cuatro ATR se calculan desde OHLC, no mediante
una etiqueta de configuracion sin efecto.

Fitness TRAIN: media no ponderada de los cuatro Sharpe anuales sobre retornos
diarios de equity, anualizados con sqrt(252), sin tasa libre de riesgo.
Por generacion solo el ganador de TRAIN va al monitor. Se guarda cuando su
Sharpe VAL mejora el mejor en >1e-8; si no mejora se incrementa el contador.
No se usa VAL en cruce, seleccion reproductiva ni mutacion. Parar si gen>=200
y contador>=30; la espera se cuenta desde el principio. Se restaura el mejor
monitor, no la ultima generacion. Si hay empate entre semillas en VAL, gana la
primera por orden 7,19,43. Las dos politicas se fijan antes de evaluar 2024.
Los candidatos duplicados reutilizan el replay con identico fenotipo.

Interfaz inspirada en app/optimizer.py de heuristic-strategy, no se ejecuto
su optimizador generico sobre este plugin. Se verifico round-trip del control
y dominio/tipo por gen; poner cada gen en sus dos extremos con los otros fijos
cambio las metricas de la cuenta de 2021. Se contrastaron retorno, DD, numero
de trades, compras y aciertos del simulador acelerado con el replay Python para
controles, extremos de genes y campeones. Resultados en results.json.

## Resultado y decision

| Parametros | Ejecucion | Retorno 2023 | Sharpe 2023 | Retorno 2024 | Sharpe 2024 | DD 2024 | Ordenes 2024 |
|---|---|---:|---:|---:|---:|---:|---:|
|Control|Con costes|98.30%|3.91|30.61%|2.12|5.54%|466|
|DEAP con costes|Con costes|151.27%|5.22|41.35%|3.04|5.72%|235|
|DEAP sin costes|Con costes|103.56%|4.49|47.20%|3.20|4.86%|366|
|Control|Sin costes|157.08%|5.37|95.59%|5.05|3.01%|431|
|DEAP sin costes|Sin costes|155.64%|5.77|105.00%|5.59|2.94%|343|
|DEAP con costes|Sin costes|168.18%|5.62|71.73%|4.36|3.10%|226|

No se demuestra un optimo global. El orden de los dos perfiles cambia entre
2023 y 2024 bajo costes; queda prohibido elegir un sustituto mirando 2024.
El siguiente barrido deberia usar cada campeon sellado en su propio brazo y
anadir el mismo campeon con/sin costes para aislar el coste de la reoptimizacion.

## Auditoria y reproduccion

config.json contiene rangos, tipos y protocolo; locked.json conserva los dos
campeones; results.json conserva las seis terminaciones y metricas por ano.
En cada carpeta por brazo y semilla, candidates.jsonl.gz contiene el genoma
normalizado, parametros efectivos y resultados anuales de cada candidato
distinto. generations.json referencia por ID los veinte fenotipos de cada
generacion y contiene fitness del lider, monitor y contador de paciencia.

```bash
PYTHONPATH=. python -m pytest -q tests/test_role464_genes.py
OPENBLAS_NUM_THREADS=1 PYTHONPATH=. python tools/role464_deap.py /var/tmp/role459-seasonal /var/tmp/sac70-warm-v3-20261108/data /var/tmp/sac399-series/history397-61-r2 /var/tmp/sac399-series/model/61/bundle/RECIPE.json docs/work_plan/464_deap/config.json /tmp/role464-reproduction
```

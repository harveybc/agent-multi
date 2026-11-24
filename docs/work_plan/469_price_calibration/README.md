# EURUSD: del incremento estacional previsto al movimiento de precio

Decision: ninguna de estas dos calibraciones justifica pasar a demo. La aparente mejora de retorno se debe a no abrir posiciones.

Se recargaron el encoder y el cabezal afinados de cada grupo y se verifico el MAE del monitor contra el de la mejor epoca guardada (no el de la ultima epoca). Se ajustaron pendientes sin intercepto sobre pronosticos reales del predictor frente a cambios del cierre bid, con cuatro anos 2019-2022. Se excluyeron destinos que cruzaban el corte o un hueco horario. No se ajusto DEAP ni se puntuaron cuentas en 2024. Para el predictor, los pronosticos de ajuste proceden de su propio periodo de entrenamiento, no son out-of-fold. El ano 2023 es desarrollo: enero-junio entro en entrenamiento del predictor; julio-diciembre fue monitor para elegir pesos. El puente no se ajusto en ninguno de esos dos semestres.

Los ocho pronosticos se entregan al controlador en cada origen archivado, sin mascara de disponibilidad de la etiqueta futura. Mismas velas, calendario, cuenta de USD 10000, costes y politica de 467. El control directo equivale a pendiente 1. El global ajusta una sola pendiente agrupando horizontes. El tercero ajusta 16 pendientes independientes, una por horizonte y signo del pronostico, sin obligarlas a ser positivas. Ajuste de minimos cuadrados, sin busqueda de hiperparametros, intercepto nulo para que un cambio previsto cero siga valiendo cero.

| Puente | Compra / venta ejecutadas | Aciertos | Retorno | DD a cierres | Maximo TP/SL propuesto |
|---|---:|---:|---:|---:|---:|
| Directo | 216 / 204 | 150/420 | -19.12% | 34.18% | 6.4202 |
| Escala global | 0 / 0 | No aplica | 0% | 0% | 0.0133 |
| Por horizonte y signo | 0 / 0 | No aplica | 0% | 0% | 0.7937 |

Umbral fijo: TP/SL estrictamente superior a 0.8536643. Medicion separada sobre los mismos 3516 instantes habilitados, suponiendo cuenta plana en cada uno, con el umbral puesto en cero exclusivamente para observar las propuestas. Se verifico fila por fila que superar el umbral real reproduce las decisiones de la politica. El directo propuso 1606 compras, 1230 ventas y 680 NOP; el libro final no puede ejecutar posiciones simultaneas. El directo perdio USD 906.31 en el primer semestre y USD 1005.27 en el segundo, en la misma cuenta continua. Ambas direcciones perdieron (compras -1586.66, ventas -324.91).

La pendiente global fue -0.00208. El ajuste por signo y horizonte bajo el MAE del precio frente al puente directo en los ocho horizontes, pero jamas produjo un TP/SL que habilitara la entrada. Un cambio medio esperado del cierre no es una estimacion de la excursion favorable o adversa intratrade. Esto no descarta una relacion no lineal ni demuestra que la estacionalidad carezca de informacion. No bajar el umbral mirando este resultado para forzar operaciones.

Siguiente contraste propuesto en desarrollo: medir la frecuencia de TP antes de SL, perdida/ganancia neta y excursion intratrade condicionadas a la propuesta original, para ambos lados y contra controles de igual horizonte, volatilidad y exposicion. Validar por bloques cronologicos, purgar etiquetas que crucen bloque, y no confundir velas que tocan las dos barreras con trayectoria intrabar conocida. Sin ventaja respecto a los controles, no invertir en otro optimizador.

Reproduccion, desde agent-multi (SRC/BUNDLE/BARS como en 467):

```bash
PYTHONPATH=. python tools/role469_bridge.py export "$SRC" "$BUNDLE" /tmp/dev469.npz
PYTHONPATH=. python tools/role469_bridge.py evaluate "$SRC" "$BARS" docs/work_plan/467_forward_gate/cost-paid/policy.json /tmp/dev469.npz "$OUT"
```

El export requiere el entorno Torch del bundle. report.json guarda las pendientes (fila negativa luego no negativa; orden 1/2/3/4/5/6/24/48 h), las cantidades de ajuste y las cuentas. errors.csv contiene MAE contra precio y persistencia; books.csv.gz, el libro ejecutado; opportunities.csv.gz, las tres decisiones para cada oportunidad; proposed_rr.csv, los cuantiles antes del umbral.

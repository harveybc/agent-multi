# 473. Valor incremental del cierre estacional con entradas congeladas

Se prueban exclusivamente las salidas. Las 271 posiciones del campeon de contexto DEAP de 471 (2020, 2021, 2022, H1 y H2 de 2023) conservan instante, lado, horizonte, TP/SL, q y **unidades originales**. La estrategia alternativa no abre una posicion adicional al salir antes. No se vuelven a optimizar parametros, no se usa 2024 y no se envia ninguna orden. El predictor ya vio partes de este periodo en entrenamiento y monitor: es evidencia de desarrollo.

## Reglas fijadas

- HOLD: mantener hasta barrera o calendario. Se verifica barra y PnL contra el libro 471 para cada posicion.
- Reloj primero: salir en el primer tick de decision elegible posterior a abrir. Reloj mitad: primer tick elegible despues de transcurrir al menos la mitad del tiempo entre la apertura y el corte semanal obligatorio, sin consultar la duracion realizada.
- Estacional SL (principal): se aplica `exit_signal` de 461 con parametros sellados en 467 (margen SL cero) y los ocho pronosticos reales. Se cierra solo si la trayectoria estacional proyecta alcanzar el SL antes del TP.
- Estacional SL o pivote (contraste): misma regla, admitiendo ademas `early_pivot` con retroceso de 0.25 ATR; `wait` no cierra si el primer paso favorable supera 0.05 ATR. No se toma como verdad una prediccion estacional de precio.
- Cota retrospectiva: el mejor PnL entre HOLD y **todos** los cierres a mercado admisibles de esa misma posicion, incluyendo costes y financiacion. No es una politica realizable. `regret_usd` significa diferencia frente a esta cota antes de la barrera/calendario, no frente a un pivote de apertura.

Para todos los brazos se exige barra previa cerrada y contigua con los ocho pronosticos disponibles. Se usa el lado ejecutable, se contabilizan spread, comision por ambos lados, slippage y roll. En el tick del fill tienen prioridad el calendario y los gaps a traves de las barreras; el cierre discrecional puede ganarle a un toque intrabar que aun no ha ocurrido. Nunca se cierra discrecionalmente en la misma barra de apertura. Un positivo `delta_usd` significa dinero conservado o ganado respecto a mantener **la misma orden**. Para que la comparacion sea pareada, el resultado porcentual es suma de PnL con unidades originales / USD 10000. No se recompone el volumen a partir del saldo alternativo.

## Resultado en 2023

| Salida | H1, 31 posiciones | H2, 29 posiciones | Cierres discrecionales H1 / H2 | USD vs HOLD H2 |
|---|---:|---:|---:|---:|
| HOLD | -1.70% | +12.77% | 0 / 0 | 0 |
| Reloj primero | -0.62% | -3.28% | 31 / 28 | -1604.69 |
| Reloj mitad | -5.98% | +8.05% | 29 / 24 | -471.93 |
| Estacional SL | -1.84% | +12.77% | 1 / 0 | 0 |
| Estacional SL o pivote | -1.47% | +0.87% | 29 / 26 | -1189.26 |

**Mecanismo del resultado.** En H1 solo 1 de 1683 ticks proyecto un movimiento adverso suficiente para tocar el SL. Cerrar esa venta costo USD 14.13 respecto a HOLD. En H2, ninguno de 1600 ticks (28 posiciones con al menos una oportunidad) alcanzo el SL: mediana del cociente movimiento adverso maximo proyectado / distancia restante 0.131, P95 0.386, maximo 0.737. No es falta de ejecucion ni de inferencia en esos ticks. La cota retrospectiva encontro una mejora posible sobre HOLD en 25 de 29 posiciones de H2; eso no demuestra que se pudiera predecir el instante.

La regla con pivote cerro 26 posiciones en H2 (mediana 4 h despues de abrir). Mejoro 9, empeoro 16 y empato 1 respecto a HOLD. Desglose por el desenlace que habrian tenido manteniendolas: 2 SL, +USD 1162.43; 2 TP, -USD 396.93; 22 calendarios, -USD 1954.75. Evitar algunos stops no compensa el coste de cerrar posiciones que convenia mantener. Diferencia media por **todas** las posiciones frente a HOLD: -USD 41.01 (intervalo descriptivo por bloques semanales -144.32 a +54.57). H1: +USD 0.74 (-92.91 a +92.74). No hay evidencia estable de beneficio; los datos no prueban imposibilidad de desarrollar otro cerrador. H2 es predominantemente SELL, no certifica rendimiento BUY.

**Decision:** no activar estas reglas ni reducir a mano los umbrales segun H2. Si se desea continuar, probar directamente un estimador de ventaja CLOSE menos HOLD con estado de posicion en el controlador (lado, distancias a SL/TP, tiempo hasta corte), frente al mismo estimador con senales estacionales adicionales. Formar las etiquetas netas en posiciones TRAIN diversas, validar por bloques con purga de trayectorias y puntuar la primera decision de cierre por posicion, no sumar como ganancias independientes los ticks de una orden. El monitor debe incluir cierres perjudiciales y ganancia sacrificada, no solo stops evitados.

## Evidencia

`summary.csv` incluye los cinco bloques y la cota. `positions.csv.gz` incluye cada salida y su contrafactual; `alternatives.csv.gz` cada tick admisible hasta el desenlace pasivo, incluso despues del instante en que algun brazo habria salido. `contrasts.csv` incluye las diferencias pareadas frente a ambos relojes y HOLD, con 10000 remuestreos de semanas de apertura. Los intervalos son descriptivos, no bandas de resultados futuros.

```bash
PYTHONPATH=. python tools/role473_exit_value.py "$SRC" "$BARS" "$PREFIX" "$SIGNALS" docs/work_plan/471_net_selector docs/work_plan/467_forward_gate/cost-paid/policy.json "$OUT"
```

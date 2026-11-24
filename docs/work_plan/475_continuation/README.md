# 475. Valor de esperar y volver a decidir

**Decision: no promover ninguno de los dos cerradores.** Frente a 474 la etiqueta nueva reduce las salidas en el primer tick, pero ninguno de los dos modelos supera a HOLD en ninguno de los dos semestres de 2023. No se retocaron umbrales, no se uso 2024 y no se lanzo DEAP ni un servicio de trading.

## Que se ha cambiado

El control 474 aprendia el neto de cerrar ahora menos el neto de conservar hasta SL/TP/calendario de forma irrevocable. En este contraste la alternativa a cerrar es **esperar al menos un tick elegible y reanudar una politica de cierre**. La salida que esa politica selecciona no es el mejor precio futuro.

Para cada trayectoria, sea Y(t) el valor neto de cerrar ahora menos HOLD pasivo, normalizado por el riesgo monetario inicial al SL. Se evalua la politica anterior sobre los estados observables de la trayectoria. Sea k el **primer** tick estrictamente posterior a t donde esa politica cierra la misma posicion. La etiqueta es Y(t)-Y(k); si no hay k antes de liquidacion, es Y(t). Esto cancela el desenlace pasivo comun cuando ambos lados son cierres discrecionales. Se respetan la prioridad de gaps/calendario, el spread, el deslizamiento, la comision, la financiacion y la disponibilidad de los pronosticos igual que en 473/474. Las cotizaciones y PnL futuros entran en etiquetas, no en variables explicativas.

Cuatro anos 2019-22, divididos en 202 bloques semanales con origenes habilitados. La primera semana usa HOLD como continuacion. Antes de cada semana se congelan los dos modelos aprendidos con las semanas anteriores; se verifica que la ultima liquidacion usada en el ajuste es anterior al primer origen del bloque. Se etiquetan las **mismas** posiciones de 474 (BUY y SELL, 6/24/48 h) usando la politica anterior de cada brazo. Solo entonces se incorporan sus estadisticas al ajuste para el bloque siguiente. Esto es una aproximacion incremental con distintas politicas historicas, no la solucion exacta de Bellman ni un oraculo. 1945869/2379098 etiquetas en estado y 1961574/2379098 en estacional usaron efectivamente otro cierre discrecional en lugar de HOLD.

Sin cambios respecto a 474: 14431 origenes, 86586 posiciones virtuales, 85144 con decisiones y 2379098 posicion-ticks, 14228 barras de decision distintas. Mismo ridge (10), mismas columnas y escalas de TRAIN, peso total uno por posicion evaluable, mismo umbral cero y primer score positivo. 2020-22 se puntua usando el ajuste de anos anteriores. Para los dos semestres 2023 se congela **el mismo** modelo al terminar 2022. Esto controla el orden temporal del cerrador, no borra el desarrollo historico del predictor y del selector sobre este panel. No es una prueba independiente del sistema completo.

## Resultados

Retorno neto con aperturas y unidades fijas de 471, por USD 10000 originales de cada semestre. No se recicla el capital al salir antes.

| Regla | H1 2023 (31) | H2 2023 (29) | Cierres tempranos H1 / H2 |
|---|---:|---:|---:|
| HOLD a barrera o calendario | -1.70% | +12.77% | 0 / 0 |
| Reloj, primer tick | -0.62% | -3.28% | 31 / 28 |
| Reloj, mitad al corte | -5.98% | +8.05% | 29 / 24 |
| Estado, etiqueta pasiva (474) | -3.76% | -1.42% | 30 / 26 |
| Estado, etiqueta con continuacion | -3.06% | -1.46% | 29 / 28 |
| Estado y estacional, etiqueta pasiva (474) | +3.42% | -7.66% | 30 / 28 |
| Estado y estacional, etiqueta con continuacion | -3.59% | +3.63% | 28 / 27 |

En H2, de las 28 posiciones con al menos un tick para decidir, la regla nueva de estado cerro 0 en el primer tick (18 antes); la estacional cerro **1** (21 antes). Mediana de horas al cierre discrecional: 28.5 para estado, 17 para estacional; antes, 1 para ambos. En H1 los cierres en el primer tick bajan de 20 a 6 y de 19 a 6, respectivamente. Cambiar la etiqueta no garantiza que un punto de salida posterior sea bueno.

El nuevo estacional cerro 27 en H2: 13 cierres beneficiosos, 13 perjudiciales y uno neutro respecto a HOLD. En posiciones que pasivamente irian a SL conservo USD 879.64; en las de TP cedio USD 233.70 y en las de calendario cedio USD 1559.64. **Neto: -USD 913.70**; el de estado perdio USD 1423.14 frente a HOLD. H2: estimacion media de la ventaja estacional en sus cierres +0.0119 R, ventaja observada +0.0147 R *frente a esperar y reanudar esa misma politica*. Esto no contradice el resultado negativo frente a mantener. Con estacional frente a estado: +USD 17.57 por posicion en H2, intervalo descriptivo semanal [-29.00,+69.90]; en H1 -USD 1.73 [-38.83,+25.92]. No hay ventaja consistente. En los bloques 2020/21/22 la delta estacional contra HOLD fue -407.82/+260.48/-688.71 USD.

**Implicacion:** haber corregido la precipitacion del cierre no demuestra una ventaja economica. Tras 473-475 no se justifica otra busqueda sobre este semestre ni promocion a demo. Conservar HOLD como referencia; cualquier nueva propuesta debe demostrar valor incremental frente a HOLD y relojes en bloques posteriores, sin cambiar simultaneamente la politica de apertura.

## Reproduccion y auditoria

```bash
OPENBLAS_NUM_THREADS=1 PYTHONPATH=. python tools/role474_close_advantage.py "$SRC" "$BARS" "$PREFIX" "$SIGNALS" docs/work_plan/471_net_selector docs/work_plan/473_exit_value "$OUT" --target continuation
```

Insumos de 474. `weekly_provenance.csv` contiene corte, ultima etiqueta previa, ultima liquidacion del bloque y conteos por politica. `weights.json` conserva los pesos anuales para reproducir 2020-23; `positions.csv.gz` contiene el primer cierre; `scores.csv.gz` contiene score y etiqueta contra esperar y seguir la politica de evaluacion congelada. En `positions.csv.gz`, delta_usd siempre es contra **HOLD pasivo**, y regret_usd sigue usando la cota retrospectiva de 473. `summary.csv` y `paired.csv` contienen las cuentas y contrastes (remuestreo descriptivo por semanas de apertura). Se verifican las 271 liquidaciones pasivas y los 12616 contrafactuales originales contra 473. La ruta pasiva de 474 se reejecuto y reprodujo posiciones, scores y resumenes a tolerancia numerica. Pruebas incluyen escaneo directo hacia adelante, preferencia por el primer disparo, ausencia de valores futuros de otra posicion, terminal y espera sin disparo.

# 478. Intensidad estacional y valor ejecutable de la direccion

**Decision:** no usar el signo o el cuartil superior como disparador global.
Los 32 MAE inferiores a persistencia de 477 eran del nivel estacional, no del
precio ejecutable. La regla de seguir el signo tiene neto medio negativo en
Q4 en los ocho horizontes al agrupar lados. Hay una asimetria interesante
en SELL a 24 h, pero no justifica promover un filtro, invertir la senal o
optimizarlo con los mismos anos.

## Prueba

Pronosticos guardados de 477, sin reentrenar, desde las mismas 14 360
oportunidades elegibles de 2020-2023. Para cada origen y horizonte h,
tomar el incremento estacional previsto / ATR de la ultima barra cerrada.
Signo positivo propone BUY y negativo SELL; cero se abstiene (no ocurrio).
Dentro de cada lado/horizonte/ano, fijar P25/P50/P75 de la magnitud con el
semestre anterior exclusivamente, sin usar rentabilidad; empates al grupo
superior. La proporcion de Q4 en evaluacion puede diferir del 25%.

Simular ambos lados al mismo instante, con las mismas distancias TP y SL
(ATR x raiz de h), q=1, rel_volume=0.1 y apalancamiento 50, spread, comision,
deslizamiento, financiacion, limite de calendario y prioridad de SL en doble
toque de 477. Horizonte designa la escala de las barreras, **no** liquidacion
obligatoria a h horas. 6/24/48 h son alternativas reales de apertura de 477;
1-5 h extienden el mismo contrato solo como sondas. Un punto basico (pb)
del balance de referencia de USD 10 000 es USD 1. Operaciones hipoteticas
superpuestas, no rentabilidad acumulable. NOP tiene neto incremental cero.

## Resultados agrupados 2020-2023

Medias en pb por oportunidad, ponderadas por filas. Q4 BUY y SELL se miden
frente a NOP. La ultima columna empareja la direccion pronosticada en Q4
con la operacion contraria sobre las mismas velas.

| h | Q1 vs NOP | Q4 vs NOP | Q4 BUY | Q4 SELL | Q4 vs contraria |
|--:|--:|--:|--:|--:|--:|
| 1 | -10.55 | -9.84 | -11.00 | -8.47 | +1.87 |
| 2 | -11.50 | -6.75 | -7.82 | -5.45 | +7.73 |
| 3 | -11.43 | -5.49 | -9.01 | -1.41 | +11.04 |
| 4 | -10.16 | -9.04 | -14.18 | -2.88 | +3.35 |
| 5 | -11.20 | -8.20 | -14.41 | -1.30 | +5.18 |
| 6 | -10.29 | -8.67 | -15.33 | -1.39 | +3.72 |
| 24 | -12.68 | -18.72 | -47.23 | +21.53 | -18.18 |
| 48 | -3.99 | -16.41 | -40.28 | +15.49 | -13.44 |

SELL 24 h: Q1/Q2/Q3/Q4 = -36.99/-16.40/-3.54/+21.53 pb. Q4-Q1 fue
positivo en los cuatro anos; Q4 neto fue -71.7/+63.4/+50.1/+17.5 pb
respectivamente. En Q4 (1742 casos), intervalo descriptivo 95% de neto
[-17.97,+67.51] pb; de ventaja frente al espejo [-17.73,+153.88] pb.
Como comprobacion adicional exploratoria, se emparejo cada Q4 con el
promedio de Q1-Q3 del mismo lado, semana UTC, hora y horizonte en otros
dias. Para SELL 24 h la diferencia fue positiva en los cuatro anos, pero
solo 979 de 1742 Q4 tuvieron pares. Es atribucion retrospectiva, no una
regla ejecutable ni evidencia independiente. El coste y la duracion
efectiva pueden diferir entre las dos trayectorias.

Intervalos de strength.csv: 3000 remuestreos por bloques circulares de
cuatro semanas de apertura dentro de cada ano; razones ponderadas por
oportunidad, no por semana. Misma muestra para neto y contraria. Son
intervalos puntuales sin correccion por busqueda entre horizontes/lados;
no presentar una celda escogida a posteriori como confirmacion.

## Que hizo realmente el selector

En 2022, control sin senal: **50 BUY / 3489 SELL / 41 NOP** antes de la
exclusion mutua. La cuenta bloqueo las 50 compras y 3239 ventas; se
ejecutaron 250 ventas. En 121 de esas 250, el pronostico del horizonte
elegido apuntaba a compra. Con la senal incluida: 217/3330/33, cuenta
ocupada para 213 compras y 3130 ventas, libro final 4 compras y 200 ventas.

Se aislaron sumandos del ridge **ya ajustado**, sin refit ni cambio en los
costes del simulador. Al llevar los dos terminos de coste/ATR a la media
de calibracion, el control pasa a 2958 BUY / 599 SELL / 23 NOP. Al retirar
solo el reloj siguen 3537 preferencias SELL; sin los terminos de volatilidad
logaritmica siguen 3534. Solo el intercepto no abriria nada. La contribucion
conjunta coste/ATR es decisiva para el sesgo de puntuacion en este panel;
no demuestra que las comisiones reales lo causen, que eliminar ambas
variables mejore PnL, ni un efecto causal independiente de la volatilidad.

## Evidencia

[Cuartiles, conteos e intervalos por ano/lado/horizonte](strength.csv) |
[Figura](strength.png) | [Cortes congelados](cuts.csv) |
[Comparacion por semana y hora](matched.csv) |
[Preferencias y ablaciones](preferences.csv) | [Ordenes y aciertos](entries.csv).
side=0 en strength.csv significa ambos lados agrupados, no NOP; net se
compara contra NOP, mirror_delta contra la direccion contraria.

Se verificaron los ocho hashes de predicciones, 576 replays por la ruta
de referencia y las 1100 aperturas de los libros originales, incluidas
direccion, instante de salida, importe, volumen y saldo; ver [audit.json](audit.json).
No se consulto un periodo nuevo ni se opero una cuenta demo.

```bash
OPENBLAS_NUM_THREADS=1 PYTHONPATH=. python -m tools.role478_direction "$RUN477" "$HISTORY397" "$MARKET" "$BUNDLE" "$OUT" --detail "$PRIVATE"
PYTHONPATH=. python -m pytest -q tests/test_role478_direction.py tests/test_role471_selector.py
```

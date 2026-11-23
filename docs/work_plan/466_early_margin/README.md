# 466. Margen de anticipacion del SL

**Decision: conservar 0 ATR.** Experimento con el oraculo estacional, costes
de 462 y siete parametros sellados del campeon con costes de 464. No es una
prueba de predicciones aprendidas. No se altero el SL ni el TP del broker.

## Protocolo

En coordenadas de la direccion de la posicion, sea b = lado * (SL - cierre
liquidable) y a = ATR disponible al decidir. Para el gatillo discrecional
se usa b + min(m*a, max(-b,0)); el tope evita disparar sobre una excursion
favorable cuando el precio aun esta del lado seguro. Se recorren los puntos
futuros disponibles en orden: si antes aparece el TP, mantener. Un m positivo
anticipa, un m negativo exige mas penetracion. Sigue habiendo barreras fisicas,
gaps, SL prioritario en doble toque y cierre obligatorio segun calendario.

Cuadricula completa -2 a +2 ATR, paso 0.05. Solo TRAIN 2019-22 elige al
lider por acierto agrupado con las guardas de 465. Igual score: menor |m|.
Solo ese lider compite contra cero en validacion 2023 para la promocion.
Se fijaron como contrastes descriptivos -0.5, 0 y +0.5; nunca se elige sobre
2024. La seleccion se escribe antes del panel de 2024. Todas las cuentas
anuales se reinician en USD 10000.

## Evidencia

18/81 admisibles en TRAIN. +0.15 ATR pasa de 418/803 a 452/839 aciertos.
En 2023 sube de 116/207 a 123/215, pero el retorno baja de 151.27% a
149.09%, el Sharpe de 5.217 a 5.200 y el DD permanece en 3.479%.
En 2024 sube de 108/235 a 111/240; retorno 41.35% a 40.21%, DD 5.72% a 5.41%.
No se reemplaza el campeon por ese intercambio.

Para aislar la salida se reproducen, por separado, las aperturas originales
con igual fecha, lado, fill, niveles y unidades: manteniendo solo protecciones,
con el margen cero y con cada margen de diagnostico. No se suman estos
contrafactuales al PnL de cuenta. Con +0.15 frente a cero: en 2023, 11 mejores,
3 peores, diferencia +463.07 USD; en 2024, 13 mejores y 8 peores, +155.47 USD
(BUY +21.30; SELL +134.17). En 2024 una perdedora pasa a ganadora pero dos
ganadoras pasan a perdedoras. Los contrastes observan un mayor importe neto
por posicion afectada en conjunto, no una mejora de cuenta completa.
Sin ningun cierre discrecional la cuenta completa deja 72.75% en 2023
y 37.56% en 2024, con DD de 6.39% y 9.34%: no confundir con mantener
las 207/235 aperturas originales, que no constituyen una cuenta simultanea.

## Verificacion y archivos

- Se cotejan las metricas anuales del margen cero con el archivo 464. Para
  cada apertura fija se verifica paridad de razon de salida, tick y PnL con el
  libro de referencia a margen cero, y el delta frente a mantener (1e-8 USD).
- Pruebas de simetria BUY/SELL para anticipacion, penetracion, prioridad del TP,
  saturacion del umbral y stop fisico; tests/test_role466_margin.py.
- train_grid.csv y train_years.csv: todos los valores y metricas de seleccion.
  locked.json: lider TRAIN y decision. accounts.csv: cuentas por margen y ano.
  fixed_entries.csv.gz: libro emparejado; paired_summary.csv: conteos y PnL
  por lado frente al cierre original y frente a mantener. no_early_control.csv:
  cuenta sin cierre discrecional. Positivos y negativos segun PnL neto.

```bash
OPENBLAS_NUM_THREADS=1 PYTHONPATH=. python tools/role466_early_margin.py /var/tmp/role459-seasonal /var/tmp/sac70-warm-v3-20261108/data /var/tmp/sac399-series/history397-61-r2 /var/tmp/sac399-series/model/61/bundle/RECIPE.json docs/work_plan/464_deap /tmp/role466-reproduction
```

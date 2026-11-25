# 476. Recuperacion y auditoria de entradas (2026-11-25)

No se encontro una corrida interrumpida en este carril. Los 12 DONE de 458
coinciden con COMPARISON; se leyeron y calcularon los hashes de los 12
checkpoints. Las etapas estacionales hasta 475 ya tenian entregables.
No se entreno otra red, no se infirio en 2024 ni se activo trading.

## Especialista OPEN_SELL (458)

Retorno neto porcentual de la validacion de desarrollo archivada:

| Encoder / plateau | Semilla 7 | 19 | 43 |
|---|---:|---:|---:|
| Transferido / restaurar | +2.057 | -2.095 | -3.488 |
| Transferido / bifurcar | +2.071 | -3.373 | -3.404 |
| Aleatorio / restaurar | +0.067 | -0.788 | +3.450 |
| Aleatorio / bifurcar | +0.067 | -0.904 | +3.450 |

Las 12 pararon por paciencia (179 a 245 epocas logicas); ninguna fue
promovida. Con inicio aleatorio en semillas 7 y 43 la bifurcacion gasto mas
pasadas y devolvio exactamente el mismo archivo de pesos que restaurar.
El +3.45% no se repite en las otras semillas; su cuenta TRAIN 2021 dio
-5.00%. No adoptar la bifurcacion ni la transferencia como mejora general
por este factorial. En trials.csv: retornos anuales, drawdown, Sharpe,
ordenes, aciertos, epoca del campeon, pasadas y hash. Contrato en
[458](../458_FULL_TRAIN_PLATEAU.md); desarrollo, no confirmacion.

## Auditoria de ventanas (459)

Reconstruccion independiente desde las columnas de origen: 28 991 decisiones
2016-2023 x 13 observaciones x 61 columnas. Para cada decision se localizaron
las barras por su cierre UTC, se aplicaron los offsets de -24 a 0 en pasos de
dos barras admitidas y los valores fijos de la receta. Ninguna barra cerro
despues de la decision. Los 22 989 863 valores coinciden sin tolerancia.
No se verificaron los origenes omitidos por la mascara de negocio.

Desde OHLC historico se reconstruyeron 10 canales wavelet, 2 SSA y 3 Hilbert
en el punto medio de cada ano de 2016 a 2019. Truncar las barras posteriores
o multiplicar su precio por 1.025 no altero ninguna salida en esos cortes.
Modificar la barra actual si altero cada familia. Error maximo contra el
almacen: 0, 3.56e-14 y 5.66e-16 respectivamente. Es evidencia sobre esas
fechas y productores, no certificacion del pipeline completo.

**Disponibilidad historica:** el escalador de las 61 entradas y los ajustes
de Kalman/GMM declarados para cinco canales usan 2020-2023. Son utilizables
despues del ajuste; el replay 2016-2019 con esos parametros no representa
lo que se podia calcular entonces. Esto no prueba fuga desde 2024 al
escalador. Para una evaluacion por bloques verdaderamente prospectiva,
ajustar escalador, productores aprendidos y encoder solo con datos anteriores
al bloque, documentar su procedencia y repetir el control de paridad antes
de atribuir rentabilidad al modelo. No reabrir la optimizacion descartada
en [475](../475_continuation/README.md).

## Reproduccion (CPU)

```bash
python tools/role476_restart_audit.py "$RUNS458" "$RECIPE61" "$WINDOWS459" "$HISTORY397/FEATURES.parquet" "$STORE" "$OUT"
OPENBLAS_NUM_THREADS=1 python tools/role476_spectral_check.py "$PRODUCER_SNAPSHOT" "$HISTORY397" "$OUT/spectral.json"
```

windows.json y spectral.json registran los resultados desglosados y hashes
de insumos/productores. Los analisis no requieren cargar pesos en una GPU.

# 459. Auditoria del estacional como pronostico para corto/largo

2026-11-23. Sin reentrenamiento, seleccion de parametros ni operaciones.
Inferencia CPU de cuatro campeones sdelta de predictor (corto/largo, encoder
congelado/afinado), sobre las ventanas originales del panel 2024. Origenes
cortos: 3303 por horizonte; largos: 1188. No son 16 muestras independientes.
Bundle 61 model.pt SHA256: 5b14e9f64f2fd296df2d0df6c6fdeffe0e16cb2e6e9715233caf400d44447c25.

Se cotejan las etiquetas con el estacional archivado y los tiempos de cierre
con el indice de barras; cero saltos de barra entre origen y destino en estos
pares. El mayor error absoluto al reproducir un MAE archivado es 3.61e-8.
Controles: persistencia del estacional; repeticion de la fase del ultimo ciclo
STL estimado solo hasta el origen; persistencia del cierre real. El puente
fijo propone cierre futuro = cierre actual + cambio estacional pronosticado.
No es una inversion exacta del STL. Los errores de precio se miden sobre las
mismas filas que los del estacional en cada horizonte.

| Horas | MAE estacional persistencia | MAE estacional misma fase | MAE estacional afinado | MAE precio persistencia | MAE precio puente afinado |
|---:|---:|---:|---:|---:|---:|
|1|0.000693|0.002710|0.000677|0.000552|0.000567|
|2|0.001006|0.002703|0.000957|0.000780|0.000816|
|3|0.001245|0.002681|0.001167|0.000971|0.001030|
|4|0.001446|0.002664|0.001335|0.001124|0.001217|
|5|0.001617|0.002637|0.001467|0.001272|0.001399|
|6|0.001766|0.002604|0.001590|0.001415|0.001568|
|24|0.003157|0.002554|0.002621|0.003370|0.004082|
|48|0.003090|0.003090|0.002416|0.004399|0.004907|

El puente pierde contra precio persistente en las 16 comparaciones; eso no
estima el PnL ni descarta utilidad condicional. En 24 h, donde se apoya la
apertura, ambos encoders pierden incluso contra repetir la fase estacional.
En 48 h el mismo control repite el extremo del ciclo por construccion.

## Alcance y siguiente prueba

- El productor del target usa el extremo derecho de 192 barras, periodo 48,
  seasonal 13, robust. No se verifico aqui la causalidad de las 61 entradas
  desde sus productores. El encoder fuente de 458 fue elegido con resultados
  en 2024; no atribuir independencia de punta a punta a este panel.
- Probar invariancia de entradas, mascaras e inferencia al truncar o perturbar
  las fuentes crudas despues del origen; auditar tiempos de publicacion,
  ventanas de normalizacion e historia del encoder. Auditar wavelet, SSA,
  Hilbert y productores aprendidos, no solo el STL. Una ablacion de pesos no
  certifica causalidad.
- Comprobar direccion y retorno ejecutable por intensidad de la senal, con
  cortes fijados en desarrollo. Luego comparar con y sin senal estacional en
  el mismo camino horario completo con OHLC reales, costos y reloj del negocio.
  No usar el nivel estacional como precio ni rellenar predicciones 72-144 h.
- 458 sigue su curso. No autorizar optimizacion de trading con este panel.
  Si el contraste sobre fechas posteriores aporta beneficio neto reproducible,
  considerar el optimizador DEAP de heuristic-strategy en desarrollo.

## Reproduccion

Desde agent-multi, con su entorno de Torch y statsmodels:

```bash
CUDA_VISIBLE_DEVICES="" PYTHONPATH=. python tools/role459_seasonal_bridge.py <directorio-artefactos> <bundle61>
```

Entradas: dataset_meta.json, dataset_sdelta.npz, windows.npz, stl_series.npz
y retsu-phase1-sdelta-{frozen,tuned}/jobs/{modo}_{short,long}/champion.pt.
Salida: BRIDGE.json (16 filas con MAE sin redondear, ventanas y diferencias).

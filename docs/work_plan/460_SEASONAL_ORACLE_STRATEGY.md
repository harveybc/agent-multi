# 460. Estrategia estacional: diseno previo al predictor

Estado: exploracion NON_GOVERNING. No es un backtest ni una promocion de modelo.
La pregunta es si una politica que recibe el estacional ideal puede extraer
beneficio del EURUSD tras costes y cuanto error de pronostico tolera.

## Evidencia inicial

Ejecutar `python tools/role460_seasonal_oracle.py /var/tmp/role459-seasonal`.
Solo pool de desarrollo, destinos anteriores a julio 2023. Origenes: 24895
por horizonte corto y 8934 por horizonte largo. Acuerdo de signo sin empates
entre cambio estacional ideal y cambio de cierre a 1,2,3,4,5,6,24,48 h:
83.65%, 81.46%, 79.37%, 78.69%, 77.30%, 76.45%, 67.41%, 56.41%.
Son ventanas solapadas, no operaciones independientes; no son fills.

## Contrastes propuestos, en orden

1. Estrategia heredada con precios ideales como control del motor. Control
   estacional minimo: alimentar la misma politica con cierre actual + delta
   estacional ideal a cada horizonte. Control sin informacion: persistencia,
   misma fase y trayectorias permutadas por bloques. Comparar en identicos
   origenes y con igual reloj, balance y costes.
2. Alternativa: las entradas son deltas estacionales, no cotizaciones.
   Validar 1-5 vs 6/24/48 y 1-6 vs 24/48 sobre soporte comun. Estimar en
   TRAIN las excursiones de precio favorable y adversa por lado, horizonte,
   fuerza de la senal y volatilidad causal; no usar el maximo o minimo del
   estacional como precio transable. Usar cuantiles y multiplicadores para
   distancias positivas TP y SL; puntuar las dos direcciones por resultado
   neto y riesgo en replay de primer toque. Empate o ventaja insuficiente: NOP.
   Reducir tablas con poco soporte a un grupo mas amplio o abstenerse.
3. Verificar las entradas y los cierres a tamano fijo antes de habilitar el
   volumen. Para cada posicion comparar cerrar ahora vs continuar desde el
   mismo estado, con costes, protecciones y horizonte terminal identicos.
   Medir ahorro, cierres destructivos, beneficio devuelto y regret por lado.
   Luego probar tamano por presupuesto de riesgo y distancia real al SL,
   acotado por margen: q entre 0 y 1, rel_volume 0.1 aplicado una sola vez.
4. Casos sellados: espejo BUY/SELL, lateral con costes, reversal temprano en
   ganancia y en perdida, conflicto corto/largo, falta de horizontes, hueco,
   SL/TP en la misma barra, spread alto y cierre semanal. Ejecutar sobre
   OHLC completo, bid/ask si disponible, primera cotizacion ejecutable despues
   de la decision; definir prioridad conservadora de barreras si no hay ticks.
   No comprimir el calendario ni rellenar horizontes ausentes.
5. Solo tras demostrar utilidad del ideal: optimizacion walk-forward interna
   con purga de etiquetas, costes, drawdown y estabilidad entre bloques.
   Fijar parametros; barrer ruido de corto y largo independientemente.
   Para gaussianas centradas, sigma_h = MAE_objetivo_h * sqrt(pi/2).
   Comparar tambien perturbaciones correlacionadas y errores reales de
   prediccion fuera de muestra por bloques. Publicar MAE realizado por horizonte,
   retorno neto, DD, Sharpe, conteo por lado y razon de salida por celda.
   Contrastar al predictor congelado en fechas posteriores sin reajustar.

No se ha ejecutado la simulacion de cuenta ni identificado un umbral rentable.
Siguiente accion: implementar el replay de los controles con el mismo calendario,
reconciliar sus fills y contrastar el efecto de entradas y salidas por separado.

# 462. Estacional ideal con costes de ejecucion

Replay de 461. Mismos datos, parametros y criterios de seleccion; sin red ni
reoptimizacion. Unica intervencion: perfil configurado de ejecucion de la
receta 61. Perfil supuesto, no tarifas verificadas de una cuenta del broker ni
spread observado tick a tick. Para reproducir, agregar
`--cost-recipe /var/tmp/sac399-series/model/61/bundle/RECIPE.json` a ambos
comandos del informe 461, con un directorio de salida nuevo. REPORT.json incluye
el perfil efectivo; cada fila de los libros CSV incluye fills y costes en USD.

Se usa el bid OHLC real y un ask = bid + spread del tramo horario desde la
apertura de sesion (0.3 a 3 pips). BUY entra ask y sale bid; SELL entra bid y
sale ask. La comision es USD 3.50/lado por 100000 EUR. Deslizamiento adverso
de 0.2 pip para mercado, 0.5 pip para stop, cero para limite. El toque intrabar
de TP requiere atravesar el nivel por 1 pip. Swap por lote/rollover: -7 USD
largo, +2.50 USD corto, triple al tercer rollover de la sesion. Como en 461,
TP/SL anclados al fill. Se ajusta el equity al precio de liquidacion y el
contrafactual de cada cierre incluye el swap hasta la salida alternativa.
Spread y slippage ya estan en el fill; el desglose no los descuenta otra vez.

| Largos | Cierre temprano | 2023 neto | 2024 neto | Ordenes 2024 | DD horario 2024 |
|---|---|---:|---:|---:|---:|
|6,24,48|ninguno|59.70%|30.34%|308|6.38%|
|6,24,48|anticipar SL|98.30%|30.61%|466|5.54%|
|6,24,48|SL y pivote con espera|78.10%|15.12%|523|6.27%|
|24,48|anticipar SL|25.25%|-4.46%|507|10.54%|
|24,48|SL y pivote con espera|11.11%|-18.79%|567|20.07%|

En las 523 operaciones del puente completo en 2024: spread 1333.15 USD,
deslizamiento 1298.04, comisiones 1705.66 y swap neto -49.22. Frente a
mantener cada posicion hasta sus protecciones o corte originales, 174 cierres
early_sl sumaron +2003.69 USD y 35 early_pivot -2026.78 USD. No sumar esos
contrafactuales al resultado de cuenta. El resultado sin costes tiene distinto
numero y tamano de operaciones, no es el PnL bruto de este libro.

Se compararon exactamente todos los campos metricos previos de los 24 paneles
sin coste con el informe 461 principal. Pruebas: 7 de la estrategia y 4 nuevas
de costes (round trip plano por direccion, rollover y activacion de limites).
Siguiente contraste: calibracion del pivote y volumen en desarrollo; medir
degradacion por error de pronostico antes de interpretar lo ideal como operable.

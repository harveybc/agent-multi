# 470. Primera barrera y valor de la entrada estacional

EURUSD 2023, desarrollo, 12 bloques mensuales. No se uso 2024 ni se cambiaron red, umbrales, parametros o costes. Predictor y politica ya habian usado 2023 en su desarrollo; esto es un diagnostico, no un resultado independiente. Pregunta: antes de considerar un cierre discrecional, el lado y el instante de las aperturas propuestas en 469 aportan beneficio sobre controles?

## Resultado

De 3516 decisiones desde estado plano salieron 2836 propuestas. Se purgaron 142 cuyo limite de liquidacion semanal cruzaba el bloque mensual (incluso si la orden acabaria antes). Quedaron 2694 propuestas evaluables; 2589 tienen ademas controles temporales. Los 105 casos sin par se conservan en la comparacion completa con el espejo.

| Mismas 2589 propuestas | TP primero | PnL neto positivo | Bruto a mismos fills USD | Coste USD | Neto USD |
|---|---:|---:|---:|---:|---:|
| Direccion y hora propuestas | 25.53% | 36.50% | +2.77 | 6.87 | -4.09 |
| Direccion inversa, misma hora | 24.95% | 35.42% | -5.26 | 6.50 | -11.76 |
| Misma direccion, otras horas comparables | 27.03% | 37.10% | +4.35 | 6.73 | -2.37 |

Importes medios por propuesta sobre un balance de referencia de USD 10000, no sumables como retorno de cartera. Los cierres por calendario explican por que una ganadora no siempre es TP. Los costes son una atribucion sobre los mismos fills (spread, deslizamiento, comision y financiacion); no se afirma que un replay sin costes abra las mismas ordenes.

La ventaja neta del lado elegido frente al espejo en **todas** las 2694 propuestas es +8.32 USD; intervalo descriptivo por remuestreo mensual [-7.46,+25.11]. Contra el control de tiempo es -1.72 USD [-14.04,+10.14] por propuesta emparejada. Ambos contrastes fueron positivos en 7 de 12 meses. Doce bloques, un ano: no prueba equivalencia ni una ventaja reproducible. Incluso resolviendo todas las velas ambiguas a favor del TP, el neto de la senal emparejada queda en -3.87 USD frente a -4.09 con SL primero.

Por horizonte y lado, ventas a 24 h dieron -1.36 USD frente a -12.00 en otros instantes (239 propuestas), y a 48 h +0.27 frente a -0.20 (877). Compras a 24 h: -8.26 frente a -1.19 (985); compras a 48 h: -4.71 frente a -4.07 (470). A 6 h solo hay una compra y 17 ventas emparejadas. No se selecciona ni descarta un subgrupo por estos datos.

## Metodo y verificacion

- Un ensayo aislado por propuesta, sin suponer capital para ejecutar todas a la vez. Se replica el fill de entrada, barreras absolutas, volumen relativo y primer evento TP/SL/calendario de 461. No hay cierre discrecional; el horizonte define la propuesta, no una nueva fecha arbitraria de liquidacion.
- El espejo intercambia BUY/SELL; el control de tiempo mantiene el lado. Ambos conservan horizonte, distancias TP y SL y fraccion de margen de la propuesta. Los pares temporales se escogen sin mirar resultados: otra sesion semanal del mismo mes, misma hora UTC, diferencia hasta el cierre forzoso de como maximo 24 h y razon entre volatilidades ATR/precio entre 1/1.25 y 1.25. Se incluyen todos los pares admisibles, aunque el modelo tambien habria abierto alli. Se promedia primero dentro de cada propuesta. No son controles NOP deliberadamente malos. Todas las decisiones provienen del mismo calendario habilitado.
- `sl_dual` registra que high/low no establecen el orden; principal conservador SL primero, sensibilidad TP primero. Gaps usan la apertura ejecutable. Las excursiones en R son cotas inferiores observadas: se incluye el precio de salida y las velas completas previas, no el extremo de la vela de salida posterior al fill.
- Paridad: los 3516 estados planos reproducen las acciones archivadas de 469. En las 375 entradas coincidentes con el motor secuencial sin cierre temprano coinciden lado, barra, motivo y PnL al mismo tamano. Test de doble barrera en ambas direcciones y de calendario/financiacion.
- `report.json`: conteos y contrastes pareados, remuestreo de meses completos con 10000 replicas. `summary.csv`: resumen global, mensual y por lado/horizonte, con excursion favorable/adversa en unidades de riesgo. `probes.csv.gz`: ensayo individual, con identificadores de origen y control. En filas espejo, `proposal_side` es el lado propuesto, `side` el efectivamente simulado.

**Decision:** sin evidencia de ventaja neta por elegir el instante; no promocionar, no invertir todas las direcciones ni buscar umbrales con DEAP en 2023. Para una siguiente hipotesis centrada en apertura, aprender en TRAIN la diferencia de utilidad neta de BUY y SELL frente a NOP usando caminos y costes reales, frente a un control que solo conozca volatilidad/hora/costes sin prediccion estacional. Validar cronologicamente en desarrollo antes de tocar cierres o seleccionar activos.

Desde agent-multi, con SRC/BARS/SIGNALS de 469:

```bash
PYTHONPATH=. python tools/role470_barriers.py "$SRC" "$BARS" "$SIGNALS" docs/work_plan/467_forward_gate/cost-paid/policy.json docs/work_plan/469_price_calibration/opportunities.csv.gz "$OUT"
```

# 06 — Broker / simulador / backtester usado en cada evaluación

## Tabla de contenido
- [Los 5 principales](#top5) · [Menciones](#menciones)
- [Nuestros experimentos](#nuestros)

## <a name="top5"></a>Los 5 principales

| Paper | Vehículo de evaluación | Broker real | Detalles |
|---|---|---|---|
| P1 GKX | Backtest propio sobre panel CRSP (portafolios mensuales calculados analíticamente) | NO | Sin motor de ejecución; sin costes; sin slippage |
| P2 Fischer-Krauss | Backtest propio (Python/Keras/H2O) sobre retornos totales diarios | NO | Fills a cierre; costes 5 pb/media vuelta aplicados aritméticamente |
| P3 ZZR DRL | Backtest propio sobre precios de cierre de futuros continuos | NO | Ejecución a cierre siguiente; costes 1–45 pb paramétricos |
| P4 DeepLOB | Clasificación pura; simulación de trading "proof-of-concept" propia sobre datos LSE | NO | Micro-mecánica de fills NO DECLARADA |
| P5 Momentum Transformer | Backtest propio walk-forward (código público en GitHub) | NO | Fills a cierre; costes 0–3 pb paramétricos |

**Hecho transversal verificado**: NINGUNO de los 5 ejecutó en broker
real, ni siquiera en paper-trading — todos son backtests propios sobre
datos históricos con fills a precio de cierre (o etiquetas de
mid-price en LOB). Los costes, cuando existen, son fracciones
aritméticas, no libros de órdenes simulados.

## <a name="menciones"></a>Menciones
- **Deng FDDR**: backtest propio minuto a minuto; costes por punto
  de índice (≈5× comisión real declarada).
- **Jiang EIIE**: backtest propio con datos de la API de Poloniex;
  supuestos DECLARADOS: liquidez suficiente para ejecutar al último
  precio del periodo y capital sin impacto de mercado; comisión
  0,25% por lado.
- **Kronos**: backtest long-only propio sobre CSI300/CSI800 con
  coste 0,15%.
- **FinAgent/FinMem**: backtests propios diarios; FinMem sin costes
  declarados.
- **Benchmarks vivos 2025–26** (contra-evidencia LLM): "When Agents
  Trade" ejecuta agentes EN VIVO; Backtrader-Bench usa el motor
  backtrader estandarizado — primeras evaluaciones del campo fuera
  de backtests ad-hoc.

## <a name="nuestros"></a>Nuestros experimentos
- **Investigación**: simulador propio `gym-fx` (runtime fijado por
  commit `634c3fd3…`, entry point `gym_fx_env` envuelto por
  agent-multi): dinámica de cuenta con solvencia/margen, costes de
  ejecución, trazas de retorno por barra con contabilidad física de
  trades; pipeline `rl_pipeline_with_validation` con roles anidados
  verificados. NO usamos backtrader ni motores de terceros.
- **Demo/paper en vivo, TRES venues reales**:
  1. **MT5 (OANDA Demo)** — terminal MetaTrader 5 con EA propio
     `LtsMt5ModelBridge.mq5` + bridge HTTP autenticado (HMAC + nonce
     + identidad de ruta firmada) + runner Python; ETHUSD activo,
     USDCAD gated.
  2. **Alpaca Paper** — API oficial, runner propio, SPY.
  3. **IBKR Paper (TWS)** — suspendido por el dueño; ledgers
     preservados.
- **Diferencia clave con el estado del arte**: nuestros resultados
  demo provienen de EJECUCIÓN REAL en broker (fills del broker Demo,
  SL/TP nativos, latencia real), no de fills a cierre asumidos — un
  nivel de realismo que ninguno de los 5 papers tiene; el costo es
  muestras pequeñas (23 round-trips ETHUSD) frente a sus miles de
  trades backtesteados.

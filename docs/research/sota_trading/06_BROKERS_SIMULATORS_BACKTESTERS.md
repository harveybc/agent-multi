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

**Hecho transversal (formulación corregida por SOTA-06)**: en las
FUENTES REVISADAS de los 5 papers NO SE REPORTA ejecución en broker ni
paper-trading (`not_reported_in_reviewed_sources`) — la ausencia no
está probada contra materiales suplementarios y código. Lo verificado
es que las evaluaciones publicadas son backtests propios con fills a
precio de cierre (o etiquetas de mid-price en LOB) y costes como
fracciones aritméticas.

Fuentes: [GKX2020 loc:§2], [FK2018 loc:§3-4], [ZZR2020 loc:§4], [DEEPLOB2019 loc:§V], [WOOD2022 loc:§V]

## <a name="menciones"></a>Menciones
- **Deng FDDR**: backtest propio minuto a minuto; costes por punto
  de índice (≈5× comisión real declarada). Fuente: [DENG2017 loc:Tab.I-III,Fig.7], [JIANG2017 loc:Tab.1-2,§5], [KRONOS2025 loc:§4,Fig.4], [FINAGENT2024 loc:Tab.4,§5], [FINMEM2023 loc:§4,Tab.resultados], [TLOB2025 loc:Tab.8,§5], [SIRCONT2019 loc:Tab.1,§3-4], [LOBCAST2024 loc:benchmark propio], [STOCKBENCH2025 loc:benchmark propio]
- **Jiang EIIE**: backtest propio con datos de la API de Poloniex;
  supuestos DECLARADOS: liquidez suficiente para ejecutar al último
  precio del periodo y capital sin impacto de mercado; comisión
  0,25% por lado. Fuente: [DENG2017 loc:Tab.I-III,Fig.7], [JIANG2017 loc:Tab.1-2,§5], [KRONOS2025 loc:§4,Fig.4], [FINAGENT2024 loc:Tab.4,§5], [FINMEM2023 loc:§4,Tab.resultados], [TLOB2025 loc:Tab.8,§5], [SIRCONT2019 loc:Tab.1,§3-4], [LOBCAST2024 loc:benchmark propio], [STOCKBENCH2025 loc:benchmark propio]
- **Kronos**: backtest long-only propio sobre CSI300/CSI800 con
  coste 0,15%. Fuente: [DENG2017 loc:Tab.I-III,Fig.7], [JIANG2017 loc:Tab.1-2,§5], [KRONOS2025 loc:§4,Fig.4], [FINAGENT2024 loc:Tab.4,§5], [FINMEM2023 loc:§4,Tab.resultados], [TLOB2025 loc:Tab.8,§5], [SIRCONT2019 loc:Tab.1,§3-4], [LOBCAST2024 loc:benchmark propio], [STOCKBENCH2025 loc:benchmark propio]
- **FinAgent/FinMem**: backtests propios diarios; FinMem sin costes
  declarados.
- **Benchmarks vivos 2025–26** (contra-evidencia LLM): "When Agents
  Trade" ejecuta agentes EN VIVO; Backtrader-Bench usa el motor
  backtrader estandarizado — primeras evaluaciones del campo fuera
  de backtests ad-hoc.

Fuentes: [DENG2017 loc:Tab.I-III,Fig.7], [JIANG2017 loc:Tab.1-2,§5], [KRONOS2025 loc:§4,Fig.4], [FINAGENT2024 loc:Tab.4,§5], [FINMEM2023 loc:§4,Tab.resultados], [TLOB2025 loc:Tab.8,§5], [SIRCONT2019 loc:Tab.1,§3-4], [LOBCAST2024 loc:benchmark propio], [STOCKBENCH2025 loc:benchmark propio]

## <a name="nuestros"></a>Nuestros experimentos
- **Investigación**: `historical_simulation` — simulador propio
  `gym-fx` (runtime fijado por
  commit `634c3fd3…`, entry point `gym_fx_env` envuelto por
  agent-multi): dinámica de cuenta con solvencia/margen, costes de
  ejecución, trazas de retorno por barra con contabilidad física de
  trades; pipeline `rl_pipeline_with_validation` con roles anidados
  verificados. NO usamos backtrader ni motores de terceros. Fuente: [OURS-PIPELINE loc:configs+manifiestos verificados por ejecución]
- **Clases de ejecución exactas (mandato SOTA-06);
  `live_capital_execution = false` en TODO el programa**:
  1. **MT5 (OANDA)** — `broker_mediated_demo_or_paper_execution` —
     terminal MetaTrader 5 con EA propio
     `LtsMt5ModelBridge.mq5` + bridge HTTP autenticado (HMAC + nonce
     + identidad de ruta firmada) + runner Python; ETHUSD activo,
     USDCAD gated.
  2. **Alpaca** — `broker_mediated_demo_or_paper_execution` — API
     oficial, runner propio, SPY.
  3. **IBKR (TWS)** — `preserved_suspended` — suspendido por el
     dueño; ledgers preservados.
- **Diferencia con el estado del arte (formulación corregida)**:
  nuestros fills provienen de ejecución MEDIADA POR BROKER en
  Demo/Paper (SL/TP nativos, latencia real) — más realista que fills
  a cierre asumidos, pero NO es ejecución con capital real y no
  establece rentabilidad, impacto ni fiabilidad de producción. Coste:
  muestras pequeñas (23 round-trips ETHUSD) frente a los miles de
  trades backtesteados del SOTA. Fuente: [OURS-PIPELINE loc:configs+manifiestos verificados por ejecución]

Fuente: [OURS-PIPELINE loc:configs+manifiestos verificados por ejecución]

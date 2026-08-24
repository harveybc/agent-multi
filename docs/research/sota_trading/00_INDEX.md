# Estado del arte en algorithmic trading — documentación verificada

Fecha: 2026-08-24 · Compilado por: General Satoshi III
Método: cuatro barridos paralelos (DRL, microestructura/LOB, forecasting
supervisado, frontera 2023–2026), extracción **contra fuente primaria**
(PDFs arXiv/journal leídos directamente). Regla de honestidad: todo dato
no confirmable en la fuente está marcado **[NO DECLARADO]** o
**[UNVERIFIED]**. Nada está inventado.

## Alcance y límite de la búsqueda (declaración honesta)

- Cubierto: IEEE TNNLS/TSP, RFS, EJOR, J. Financial Data Science,
  Quantitative Finance, Mathematical Finance, KDD/AAAI/ICLR, arXiv
  q-fin.*/cs.LG, SSRN, benchmarks vivos 2025–2026.
- Criterio de selección: citas + rigor metodológico (out-of-sample real,
  costes, ablaciones, semillas) + economía verificada.
- Límite: la exhaustividad absoluta es inalcanzable; este es el conjunto
  más fuerte VERIFICABLE, no una garantía de optimalidad global. Los
  subcampeones evaluados y descartados están documentados en cada
  archivo con su razón de descarte.
- Advertencia estructural: los agentes LLM 2023–24 (FinAgent, FinMem)
  reportan cifras altas PERO los benchmarks libres de contaminación
  (StockBench arXiv:2510.02209; Backtrader-Bench arXiv:2608.11232;
  "When Agents Trade" arXiv:2510.11695) muestran que la mayoría no bate
  buy-and-hold fuera del cutoff de entrenamiento del LLM.

## Los 5 seleccionados

| # | Paper | Venue/Año | Dominio |
|---|---|---|---|
| P1 | Gu, Kelly & Xiu — Empirical Asset Pricing via Machine Learning | RFS 2020 | ML supervisado, cross-section acciones |
| P2 | Fischer & Krauss — Deep learning with LSTM networks | EJOR 2018 | LSTM diario S&P 500 |
| P3 | Zhang, Zohren & Roberts — Deep RL for Trading | JFDS 2020 | DRL futuros multi-clase |
| P4 | Zhang, Zohren & Roberts — DeepLOB | IEEE TSP 2019 | CNN-LSTM libro de órdenes |
| P5 | Wood, Giegerich, Roberts & Zohren — Momentum Transformer | arXiv 2112.08534 (2022) | Transformer end-to-end futuros |

Menciones documentadas: Deng et al. FDDR (TNNLS 2017), Jiang-Xu-Liang
EIIE (2017), Kronos (2025/AAAI'26), FinAgent (KDD'24), FinMem (2023),
TLOB (2025), Sirignano & Cont (QF 2019), Théate & Ernst TDQN (ESWA
2021), LOBCAST (2024, control negativo).

## Archivos (OCHO archivos de aspecto más este índice; el 09 es la autocrítica)

1. [01_ASSETS_MARKETS_DATA_SOURCES.md](01_ASSETS_MARKETS_DATA_SOURCES.md) — activos, mercados, timeframes, fuentes de datos, fechas exactas.
2. [02_INPUTS_FEATURES.md](02_INPUTS_FEATURES.md) — inputs/features de cada paper, dimensionalidades y ventanas.
3. [03_PREPROCESSING.md](03_PREPROCESSING.md) — normalización, etiquetas/recompensas, splits, anti-lookahead.
4. [04_MODELS_ARCHITECTURES_HYPERPARAMS.md](04_MODELS_ARCHITECTURES_HYPERPARAMS.md) — arquitecturas e hiperparámetros exactos.
5. [05_TRADING_STRATEGY_EXECUTION.md](05_TRADING_STRATEGY_EXECUTION.md) — mecánica exacta de órdenes: apertura, sizing, apalancamiento, costes.
6. [06_BROKERS_SIMULATORS_BACKTESTERS.md](06_BROKERS_SIMULATORS_BACKTESTERS.md) — dónde se ejecutó cada evaluación (backtester/simulador/broker).
7. [07_METRICS_RESULTS.md](07_METRICS_RESULTS.md) — métricas y valores exactos reportados.
8. [08_TRAINING_OPTIMIZATION.md](08_TRAINING_OPTIMIZATION.md) — régimen de entrenamiento, búsqueda de hiperparámetros, ensembles, re-entrenamiento.
9. [09_AUTOCRITICA_COMPARATIVA_PARA_MUSASHI.md](09_AUTOCRITICA_COMPARATIVA_PARA_MUSASHI.md) — autocrítica severa del sistema propio contra este estado del arte (para Musashi).

Cada archivo cubre TODOS los papers sin excepción y cierra con la
sección **«Nuestros experimentos»** (estado a 2026-08-24).

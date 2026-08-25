# 01 — Activos, mercados, timeframes, fuentes de datos y fechas

## Tabla de contenido
- [P1 Gu-Kelly-Xiu (RFS 2020)](#p1)
- [P2 Fischer-Krauss (EJOR 2018)](#p2)
- [P3 Zhang-Zohren-Roberts DRL (JFDS 2020)](#p3)
- [P4 DeepLOB (IEEE TSP 2019)](#p4)
- [P5 Momentum Transformer (2022)](#p5)
- [Menciones documentadas](#menciones)
- [Nuestros experimentos](#nuestros)

## <a name="p1"></a>P1 — Gu, Kelly & Xiu (RFS 2020)
- **Activos**: TODAS las acciones individuales de NYSE, AMEX y NASDAQ;
  ~30.000 acciones en total, promedio >6.200/mes. Sin filtros
  deliberadamente: incluye acciones bajo $5, share codes fuera de
  10/11 y financieras (contra el sesgo de selección). Fuente: [GKX2020 loc:§2.1 p.2248-2249]
- **Timeframe**: MENSUAL (retornos totales mensuales).
- **Fuente de datos**: CRSP (retornos, panel completo incl. deslistadas
  — sin sesgo de supervivencia); características por acción según
  Green, Hand & Zhang (2017) extendidas a 1957; macro de Welch-Goyal;
  T-bill como tasa libre de riesgo. Fuente: [GKX2020 loc:§2.1 p.2248-2249]
- **Fechas**: marzo 1957 – diciembre 2016 (60 años). Test
  out-of-sample: 1987–2016 (30 años).

Fuente: [GKX2020 loc:§2.1 p.2248-2249]

## <a name="p2"></a>P2 — Fischer & Krauss (EJOR 2018)
- **Activos**: TODOS los constituyentes históricos del S&P 500,
  reconstruidos con listas de constituyentes a fin de mes (matriz de
  constituyencia binaria: el índice reproducido en cada punto del
  tiempo — cero sesgo de supervivencia; ~500 acciones por periodo de
  estudio, mantenidas aunque se deslisten dentro de la ventana de
  trading). Fuente: [FK2018 loc:§2.1]
- **Timeframe**: DIARIO.
- **Fuente**: Thomson Reuters — listas de constituyentes dic 1989 –
  sep 2015; índices de retorno total diarios (con dividendos,
  ajustados por splits) ene 1990 – oct 2015. Fuente: [FK2018 loc:§2.1]
- **Fechas de evaluación de trading**: dic 1992 – oct 2015 (5.750
  días de trading, 23 periodos de estudio).

Fuente: [FK2018 loc:§2.1]

## <a name="p3"></a>P3 — Zhang, Zohren & Roberts DRL (JFDS 2020)
- **Activos**: 50 contratos de FUTUROS CONTINUOS ratio-ajustados:
  25 commodities, 11 índices de acciones, 5 renta fija, 9 FX (lista
  completa de tickers en el Apéndice A del paper: ES, EN, LX, XU, TY,
  US, DT, ZG, ZU, ZW, CC, BN, FN, JN, DX, NK, …). Fuente: [ZZR2020 loc:§datos,App.A]
- **Timeframe**: DIARIO.
- **Fuente**: Pinnacle Data Corp CLC Database.
- **Fechas**: dataset 2005–2019; TEST out-of-sample 2011–2019
  (ventana expansiva, re-entrenamiento cada 5 años). Fuente: [ZZR2020 loc:§datos,App.A]

Fuente: [ZZR2020 loc:§datos,App.A]

## <a name="p4"></a>P4 — DeepLOB (IEEE TSP 2019)
- **Dataset A — FI-2010** (benchmark público): 5 acciones de NASDAQ
  Nordic (Kesko, Outokumpu, Sampo, Rautaruukki, Wärtsilä), libro de
  órdenes a 10 niveles, 10 días hábiles consecutivos: 1–14 junio
  2010. Muestreo por bloques de 10 eventos (~394.337 muestras según
  TLOB sobre el mismo benchmark). Fuente: [DEEPLOB2019 loc:§III datasets]
- **Dataset B — London Stock Exchange**: entrenamiento con Lloyds,
  Barclays, Tesco, BT, Vodafone; TRANSFERENCIA testeada en 5 acciones
  jamás vistas: HSBC, Glencore, Centrica, BP, ITV. 3 enero – 24
  diciembre 2017, horario 08:30–16:00. ~150.000 eventos/día/acción,
  >134 millones de muestras. Fuente: [DEEPLOB2019 loc:§III datasets]
- **Timeframe**: EVENT-TIME (cada actualización del libro), no reloj.
- **Fuente**: FI-2010 público; LSE: datos de libro de órdenes a 10
  niveles [proveedor exacto NO DECLARADO en el paper].

Fuente: [DEEPLOB2019 loc:§III datasets]

## <a name="p5"></a>P5 — Momentum Transformer (Wood et al. 2022)
- **Activos**: 50 futuros continuos líquidos, balanceados entre
  commodities, índices, renta fija y FX; series continuas con ajuste
  backward-ratio.
- **Timeframe**: DIARIO.
- **Fuente**: Pinnacle Data Corp CLC Database (la misma que P3).
- **Fechas**: 1990–2020. Test walk-forward: ventanas de 5 años
  1995→2020; análisis específicos 2015–2020 y crisis COVID
  (1 ene – 15 oct 2020). Fuente: [WOOD2022 loc:§V walk-forward]

Fuente: [WOOD2022 loc:§V walk-forward]

## <a name="menciones"></a>Menciones documentadas
- **Deng et al. FDDR (TNNLS 2017)**: futuros chinos por MINUTO — IF
  (índice CSI-300, ene 2014–sep 2015), plata AG y azúcar SU (ene
  2014–ene 2015); S&P 500 DIARIO ene 1990–sep 2015 (evaluado nov
  1997–sep 2015). Fuente de datos china: [NO DECLARADO el vendor].
- **Jiang-Xu-Liang EIIE (2017)**: criptomonedas en Poloniex — BTC como
  efectivo + las 11 monedas de mayor volumen 30 días (preselección
  ANTES de cada backtest, contra supervivencia). Periodicidad 30
  MINUTOS. API pública de Poloniex. Backtests: 2016-09-07→10-28,
  2016-12-08→2017-01-28, 2017-03-07→04-27 (UTC exactos en el paper).
- **Kronos (2025)**: pre-entrenamiento con >12.000 millones de velas
  OHLCV de 45 bolsas globales (acciones, TODO Binance spot, >1.000
  pares FX, futuros), 12 frecuencias de 1-min a semanal; corte de
  entrenamiento jun 2024, TEST estrictamente ≥ julio 2024. Backtest
  económico en CSI300/CSI800. Fuente: [DENG2017 loc:Tab.I-III,Fig.7], [JIANG2017 loc:Tab.1-2,§5], [KRONOS2025 loc:§4,Fig.4], [FINAGENT2024 loc:Tab.4,§5], [FINMEM2023 loc:§4,Tab.resultados], [TLOB2025 loc:Tab.8,§5], [SIRCONT2019 loc:Tab.1,§3-4], [LOBCAST2024 loc:benchmark propio], [STOCKBENCH2025 loc:benchmark propio]
- **FinAgent (KDD'24)**: 5 acciones US (AAPL, AMZN, GOOGL, MSFT, TSLA)
  + ETHUSD, DIARIO; train 2022-06-01→2023-06-01, test
  2023-06-01→2024-01-01; noticias de Bloomberg Tech/Seeking
  Alpha/CNBC. Fuente: [DENG2017 loc:Tab.I-III,Fig.7], [JIANG2017 loc:Tab.1-2,§5], [KRONOS2025 loc:§4,Fig.4], [FINAGENT2024 loc:Tab.4,§5], [FINMEM2023 loc:§4,Tab.resultados], [TLOB2025 loc:Tab.8,§5], [SIRCONT2019 loc:Tab.1,§3-4], [LOBCAST2024 loc:benchmark propio], [STOCKBENCH2025 loc:benchmark propio]
- **FinMem (2023)**: TSLA, NFLX, AMZN, MSFT, COIN, DIARIO; warm-up
  2021-08-17→2022-10-05, test 2022-10-06→2023-04-10; Alpaca News API,
  SEC 10-Q/10-K, Yahoo Finance.
- **TLOB (2025)**: FI-2010 + TSLA/INTC vía LOBSTER (NASDAQ), 2–30
  enero 2015, 10 niveles, muestreo por volumen (1 snapshot por 500
  acciones negociadas), ~24M muestras. Fuente: [DENG2017 loc:Tab.I-III,Fig.7], [JIANG2017 loc:Tab.1-2,§5], [KRONOS2025 loc:§4,Fig.4], [FINAGENT2024 loc:Tab.4,§5], [FINMEM2023 loc:§4,Tab.resultados], [TLOB2025 loc:Tab.8,§5], [SIRCONT2019 loc:Tab.1,§3-4], [LOBCAST2024 loc:benchmark propio], [STOCKBENCH2025 loc:benchmark propio]
- **Sirignano & Cont (QF 2019)**: ~1.000 acciones NASDAQ, ene 2014 –
  mar 2017, NASDAQ Level III (ITCH) reconstruido con LOBSTER;
  event-time en cambios de mid-price (paso medio 1,7 s). Fuente: [DENG2017 loc:Tab.I-III,Fig.7], [JIANG2017 loc:Tab.1-2,§5], [KRONOS2025 loc:§4,Fig.4], [FINAGENT2024 loc:Tab.4,§5], [FINMEM2023 loc:§4,Tab.resultados], [TLOB2025 loc:Tab.8,§5], [SIRCONT2019 loc:Tab.1,§3-4], [LOBCAST2024 loc:benchmark propio], [STOCKBENCH2025 loc:benchmark propio]

Fuentes: [DENG2017 loc:Tab.I-III,Fig.7], [JIANG2017 loc:Tab.1-2,§5], [KRONOS2025 loc:§4,Fig.4], [FINAGENT2024 loc:Tab.4,§5], [FINMEM2023 loc:§4,Tab.resultados], [TLOB2025 loc:Tab.8,§5], [SIRCONT2019 loc:Tab.1,§3-4], [LOBCAST2024 loc:benchmark propio], [STOCKBENCH2025 loc:benchmark propio]

## <a name="nuestros"></a>Nuestros experimentos (estado 2026-08-24)

### Investigación (campaña P1 en curso)
- **Activo**: ETH/USDT.
- **Timeframe**: 4 horas (H4). 2.190 barras ≈ 1 año.
- **Fuente**: dataset consolidado
  `predictor/examples/data/project3/ethusdt_4h_tech_stat_full_model_ready.csv`
  — 18.085 filas × 90 columnas (~8,25 años), derivado de datos spot de
  Binance por el pipeline feature-eng propio [la cadena exacta de
  descarga original está en los repos feature-eng/preprocessor]. Fuente: [OURS-PIPELINE loc:configs+manifiestos verificados por ejecución]
- **Fechas y roles (manifiesto anidado verificado, sha
  `2b31b7770f815b75…`)**: fit_train hasta 2022 (11.509 filas);
  train_monitor = año 2022 (2.190); inner_validation = 2023 (2.190);
  outer_validation = 2024 (2.196); sealed_test = 2025 (2.190) —
  ESTRUCTURALMENTE inmaterializado en modo l1, jamás leído. Fuente: [OURS-PIPELINE loc:configs+manifiestos verificados por ejecución]

### Trading demo/paper en vivo (3 venues)
- **MT5 Demo (OANDA)**: ETHUSD H4 activo (23 round-trips completados);
  USDCAD H4 preparado-inactivo (activación gated). Broker server:
  cuenta Demo OANDA vía terminal MT5.
- **Alpaca Paper**: SPY (10 exposiciones completadas + 1 abierta al
  último corte).
- **IBKR Paper**: USD.CAD 4h — SUSPENDIDO por el dueño (12
  exposiciones históricas preservadas); modelo
  `usdcad-4h-linear-live-v1`.
- Regla de la casa: solo Paper/Demo, capital real jamás.

Fuente: [OURS-PIPELINE loc:configs+manifiestos verificados por ejecución]

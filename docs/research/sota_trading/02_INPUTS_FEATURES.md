# 02 — Inputs y features (qué entra exactamente al modelo)

## Tabla de contenido
- [P1 Gu-Kelly-Xiu](#p1) · [P2 Fischer-Krauss](#p2) · [P3 ZZR DRL](#p3)
- [P4 DeepLOB](#p4) · [P5 Momentum Transformer](#p5)
- [Menciones](#menciones) · [Nuestros experimentos](#nuestros)

## <a name="p1"></a>P1 — Gu-Kelly-Xiu
- **920 features por acción-mes**: 94 características de la acción
  (61 de actualización anual, 13 trimestral, 20 mensual — familias:
  momentum/reversal, liquidez, volatilidad, valoración, fundamentales)
  × (8 predictores macro Welch-Goyal + constante) + 74 dummies de
  industria (2 primeros dígitos SIC).
- Los 8 macro: dividend-price (dp), earnings-price (ep), book-market
  (bm), net equity expansion (ntis), T-bill (tbl), term spread (tms),
  default spread (dfy), varianza del mercado (svar).
- Sin secuencias: vector estático por observación mensual.

Fuente: [GKX2020 loc:§2.1]

## <a name="p2"></a>P2 — Fischer-Krauss
- **LSTM: UNA sola feature** — el retorno simple de 1 día
  estandarizado — presentada como SECUENCIA de 240 pasos (~1 año
  bursátil). Dimensión de entrada: 1×240.
- Benchmarks sin memoria (RF/DNN/logística): 31 retornos acumulados
  multi-periodo con m ∈ {1,…,20} ∪ {40,60,…,240}.

Fuente: [FK2018 loc:§2.1,§3.1]

## <a name="p3"></a>P3 — Zhang-Zohren-Roberts DRL
- Estado = últimas **60 observaciones** de cada una de:
  1. serie de precio de cierre normalizada;
  2. retornos a 1 mes, 2 meses, 3 meses y 1 año, cada uno normalizado
     por la volatilidad diaria escalada al horizonte (p.ej.
     r_{t−252,t}/(σ_t·√252)), σ_t = std móvil exponencial de 60 días;
  3. indicadores MACD multi-escala (Baz et al.): q_t = (m(S) −
     m(L))/std(precio 63d); MACD = q_t/std(q 252d); escalas cortas
     S∈{8,16,32}, largas L∈{24,48,96};
  4. RSI con lookback de 30 días.

Fuente: [ZZR2020 loc:§3 estado]

## <a name="p4"></a>P4 — DeepLOB
- **Imagen 100×40**: los 100 estados más recientes del libro × 40
  features por estado = 10 niveles × 2 lados × (precio, volumen).
- Sin indicadores técnicos, sin features manuales: el CNN aprende la
  estructura precio-volumen/bid-ask/niveles por diseño de filtros.

Fuente: [DEEPLOB2019 loc:§III-IV]

## <a name="p5"></a>P5 — Momentum Transformer
- Por activo y día: retornos vol-normalizados a 5 escalas (diaria,
  mensual, trimestral, semestral, anual); MACD(8,24), MACD(16,48),
  MACD(32,96); opcionalmente 2 features de changepoint detection por
  lookback (severidad ν y localización γ de un módulo GP) con
  lookbacks 21 y 126 días.
- Secuencia de entrada: 1 año de días hábiles (~252 pasos).

Fuente: [WOOD2022 loc:§V features]

## <a name="menciones"></a>Menciones
- **Deng FDDR**: futuros — vector R^50 = cambios de precio crudos de
  los últimos 45 minutos + cambios de momentum vs 3h/5h/1d/3d/10d
  atrás; DELIBERADAMENTE sin indicadores técnicos. S&P: 20 cambios
  diarios (R^20); variante multi-mercado apila S&P+FTSE+HangSeng+
  Nikkei+Shanghai → R^100.
- **Jiang EIIE**: tensor de precios (3, 11, 50): cierre/máximo/mínimo
  de cada periodo de 30 min, 11 activos, lookback 50 periodos (25 h).
  SIN volumen, SIN indicadores.
- **Kronos**: velas OHLCV crudas tokenizadas (cuantización esférica
  binaria k=20 bits → vocabulario 2^20), contexto 512 tokens.
- **FinAgent**: OHLC+ajustado diario, ~7.900–10.000 noticias/activo,
  imágenes de gráficos K-line e historial, textos de analistas
  (~400–600/activo), MACD/KDJ+RSI/Z-score como herramientas.
- **FinMem**: noticias diarias, 10-Q, 10-K en tres capas de memoria
  (retención 14/90/365 días) + OHLCV Yahoo.
- **TLOB**: LOB 10 niveles (40 features), secuencia 128; BiN como capa
  de entrada.
- **Sirignano-Cont**: estado del libro + historia (hasta 5.000 lags
  ≈ 2 h de flujo); definición por-feature exacta [NO DECLARADA
  completa en el arXiv].

Fuentes: [DENG2017 loc:Tab.I-III,Fig.7], [JIANG2017 loc:Tab.1-2,§5], [KRONOS2025 loc:§4,Fig.4], [FINAGENT2024 loc:Tab.4,§5], [FINMEM2023 loc:§4,Tab.resultados], [TLOB2025 loc:Tab.8,§5], [SIRCONT2019 loc:Tab.1,§3-4], [LOBCAST2024 loc:benchmark propio], [STOCKBENCH2025 loc:benchmark propio]

## <a name="nuestros"></a>Nuestros experimentos
- **Observación estructurada (dict de gym-fx)**: bloque `features`
  de forma **(32, 83)** — ventana de 32 barras H4 × 83 features — más
  bloques escalares de estado de cuenta (4 llaves: equity/balance/
  posición/exposición según el contrato de observación).
- **Las 83 features** (dataset project3, 90 columnas menos
  DATE_TIME/OHLCV) por familias semánticas (agrupación verificada
  exhaustiva y disjunta en el extractor agrupado): retornos y momentum
  16; tendencia y nivel 23; osciladores acotados 9; volatilidad y
  distribución 29; volumen y flujo 6. Ejemplos: return_1,
  log_return_1, return_5, return_10, RSI, MACD, ATR, bandas, z-scores
  estadísticos [lista completa en la config
  `project3_ethusdt_4h_sac_grouped_features_v1.json`].
- **Eje temporal verificado**: eje 0 = tiempo, viejo→nuevo (prueba de
  corrimiento de una barra contra el env real).
- **Modelo en producción demo (lts)**: features de barra cerrada vía
  `prediction_provider_mechanics.build_closed_bar_features` sobre las
  últimas 60 barras del bridge (ETHUSD) — contrato de observación
  declarado en el manifiesto del modelo con paridad verificada.

Fuente: [OURS-PIPELINE loc:configs+manifiestos verificados por ejecución]

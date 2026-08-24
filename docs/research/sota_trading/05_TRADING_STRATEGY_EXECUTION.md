# 05 — Estrategia de trading y ejecución: cómo se abren las órdenes,
# sizing, apalancamiento, costes

## Tabla de contenido
- [P1](#p1) · [P2](#p2) · [P3](#p3) · [P4](#p4) · [P5](#p5)
- [Menciones](#menciones) · [Nuestros experimentos](#nuestros)

## <a name="p1"></a>P1 — Gu-Kelly-Xiu
- **Apertura**: al cierre de cada MES se ordenan todas las acciones
  por retorno predicho; se forman DECILES; portafolio long-short =
  largo decil 10, corto decil 1.
- **Sizing**: dos esquemas — value-weighted (por capitalización) y
  equal-weighted dentro del decil. Sin stop-loss, sin take-profit:
  rebalanceo mensual puro.
- **Apalancamiento**: no se declara apalancamiento explícito; el
  long-short es autofinanciado nocional (100/100). [Margen real NO
  MODELADO].
- **Costes**: NO deducidos; turnover reportado (110–130%/mes en
  NN1–NN5) para que el lector los impute.

Fuente: [GKX2020 loc:§2.4 portafolios,Tab.7]

## <a name="p2"></a>P2 — Fischer-Krauss
- **Apertura**: cada DÍA t, tras el cierre, se rankean ~500 acciones
  por probabilidad predicha de superar la mediana en t+1; se abre
  LARGO en las top-k y CORTO en las bottom-k (foco k=10);
  ejecución asumida al cierre de t+1 [convención del paper: entrada
  y salida a precios de cierre del día siguiente]; cartera
  reconstituida a diario (holding de 1 día).
- **Sizing**: equal-weighted 1/k por pata. Sin SL/TP.
- **Apalancamiento**: long-short 1:1 nocional; margen no modelado.
- **Costes**: 5 puntos básicos por media vuelta (0,05%), aplicados
  a las 4 medias-vueltas diarias del par largo-corto.

Fuente: [FK2018 loc:§3.5,§4 k=10]

## <a name="p3"></a>P3 — Zhang-Zohren-Roberts DRL
- **Apertura**: la RED emite directamente la posición objetivo del
  día siguiente por contrato: {−1, 0, +1} = máximo corto / plano /
  máximo largo (A2C: continua en [−1,1]). El cambio de posición se
  ejecuta al precio de cierre siguiente.
- **Sizing (volatility targeting)**: contratos μ=1; exposición
  efectiva = A_t · σ_target/σ_t — la posición se INFLA en mercados
  calmos y se ENCOGE en volátiles. A nivel portafolio, 50 contratos
  equal-weighted con segundo escalado a volatilidad objetivo.
  [Valor numérico del σ_target NO DECLARADO.]
- **Apalancamiento**: implícito en futuros (margen); no se declara
  límite de apalancamiento explícito.
- **Costes**: 20 pb en el entrenamiento (regularizador); evaluación
  a 1–45 pb; sin SL/TP — la salida es el cambio de señal.

Fuente: [ZZR2020 loc:§3 vol-targeting,§4 costes]

## <a name="p4"></a>P4 — DeepLOB
- **No es un paper de estrategia**: el producto es la CLASIFICACIÓN
  del próximo movimiento del mid-price. La simulación de trading en
  LSE es "proof-of-concept": entra según la clase predicha
  (sube→largo, baja→corto) y sale al cambiar la predicción; tamaño
  1 unidad [micro-mecánica de fill y coste NO DECLARADOS en detalle;
  PnL exactos no extraídos — UNVERIFIED].
- Sin SL/TP, sin apalancamiento declarado.

Fuente: [DEEPLOB2019 loc:§V-C simulación]

## <a name="p5"></a>P5 — Momentum Transformer
- **Apertura**: posición CONTINUA z∈(−1,1) por futuro y día, emitida
  por la red; el trade diario es la DIFERENCIA de posición.
- **Sizing**: retornos del activo escalados por σ ex-ante 60d;
  portafolio 50 futuros promediado y llevado a σ objetivo 15% anual.
- **Apalancamiento**: el implícito del vol-targeting sobre futuros
  (posiciones nominales crecen si σ<15%); sin límite explícito
  declarado.
- **Costes**: análisis explícito de sensibilidad — Sharpe reportado
  a C = 0/0,5/1/1,5/2/2,5/3 pb; sin SL/TP.

Fuente: [WOOD2022 loc:§V,Exh.10]

## <a name="menciones"></a>Menciones
- **Deng FDDR**: δ∈{1,0,−1} sobre UN contrato; coste por CAMBIO de
  posición |δ_t−δ_{t−1}|·c con c = 1 pt (IF), 2 pt (AG), 1,5 pt (SU)
  — fijado ≈5× las comisiones reales; S&P: 0,1% del índice. Sin
  SL/TP; sin apalancamiento declarado (futuros margen implícito).
- **Jiang EIIE**: la ACCIÓN es el vector de pesos del portafolio
  (softmax + sesgo BTC); rebalanceo TOTAL cada 30 min al peso
  objetivo; comisión 0,25% por lado (máximo Poloniex) dentro del
  factor μ_t resuelto iterativamente; sin cortos, sin margen, sin
  SL/TP.
- **FinAgent/FinMem**: decisión diaria {buy, sell, hold} sobre UN
  activo; sizing no fraccionado [detalle de sizing NO DECLARADO];
  sin SL/TP nativo; costes: FinMem NO declara costes de transacción.
- **Kronos**: backtest long-only top-k (CSI300 k=50; CSI800 k=200),
  holding mínimo 5 días, coste 0,15% por operación.

Fuentes: [DENG2017 loc:Tab.I-III,Fig.7], [JIANG2017 loc:Tab.1-2,§5], [KRONOS2025 loc:§4,Fig.4], [FINAGENT2024 loc:Tab.4,§5], [FINMEM2023 loc:§4,Tab.resultados], [TLOB2025 loc:Tab.8,§5], [SIRCONT2019 loc:Tab.1,§3-4], [LOBCAST2024 loc:benchmark propio], [STOCKBENCH2025 loc:benchmark propio]

## <a name="nuestros"></a>Nuestros experimentos

### En el simulador de investigación (gym-fx, campaña P1)
- **Acción**: continua cruda del SAC; umbral de decisión
  `continuous_action_threshold = 0.0` en ambas fases (semántica de
  acción IDÉNTICA easy/normal por contrato) — signo⇒dirección;
  zona muerta solo si |acción|<umbral (con 0.0, cualquier valor no
  nulo decide).
- **Apertura/cierre**: el env abre/cierra según la señal por barra
  H4; contabilidad de trades por `closed_trades_cumulative` con
  reconciliación física (liquidación terminal como fila añadida,
  jamás mutación de filas de mercado).
- **Sizing**: cash inicial 10.000; dimensionamiento interno del env
  [posición unitaria por señal en la configuración vigente];
  apalancamiento: dinámica de solvencia del env
  (`normal_realistic`; margen/insolvencia simulados; el modo easy
  relaja SOLO la dinámica de solvencia y es train-only por guard).
- **Costes**: coste de ejecución del env dentro de la recompensa
  (curriculum de coste configurable; en la campaña actual sin
  curriculum — coste pleno desde el inicio).

### En demo/paper en vivo (lts)
- **MT5 ETHUSD (activo)**: el runner evalúa en cada barra H4 cerrada;
  órdenes `open_long`/`open_short`/`close` encoladas con evidencia de
  modelo (3×sha256) e idempotencia; **volumen 0,01 lotes fijo**;
  **SL/TP NATIVOS obligatorios en la entrada** (perfil ETH:
  stop_fraction 1%, take_profit 2%; USDCAD preparado: 0,3%/0,6%);
  cierre temprano controlado por el modelo; magic 26080301 (ETH) /
  26080302 (USDCAD, inactivo); presupuesto diario de entradas
  compartido a nivel cuenta (mandato bridge); techo de riesgo diario
  (fracciones de riesgo en el perfil: risk_fraction_at_stop 2e-5,
  daily_loss_budget 8e-5, gross_notional/margin ≤0,3%);
  max_concurrent_positions 1 por ruta; sin adopción de posiciones
  ajenas; Demo únicamente, capital real jamás.
- **Alpaca (SPY)** e **IBKR USD.CAD (suspendido)**: runners análogos
  con mandatos propios (IBKR: máx 4 entradas/día, riesgo en stop
  6,25e-5, techo 25.000 unidades — mandato verificado).

Fuente: [OURS-PIPELINE loc:configs+manifiestos verificados por ejecución]

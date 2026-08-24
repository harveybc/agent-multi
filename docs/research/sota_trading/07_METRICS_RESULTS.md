# 07 — Métricas y valores exactos reportados

## Tabla de contenido
- [P1](#p1) · [P2](#p2) · [P3](#p3) · [P4](#p4) · [P5](#p5)
- [Menciones](#menciones) · [Nuestros experimentos](#nuestros)

## <a name="p1"></a>P1 — Gu-Kelly-Xiu (OOS 1987–2016)
- **R² OOS mensual por acción** (vs pronóstico cero, %): OLS-920
  **−3,46**; OLS-3+H 0,16; PLS 0,27; PCR 0,26; ENet+H 0,11; GLM+H
  0,19; RF 0,33; GBRT+H 0,34; NN1 0,33; NN2 0,39; **NN3 0,40**
  (mejor); NN4 0,39; NN5 0,36. Top-1.000 acciones: NN3 0,70, NN4
  0,67. S&P 500 mensual: NN3 1,80%.
- **Deciles long-short** (mensual medio / σ / Sharpe anualizado):
  value-weighted: OLS-3+H 0,94/5,33/0,61; PCR 1,22/4,82/0,88; RF
  1,62/5,75/0,98; NN1 1,81/5,34/1,17; NN3 2,12/6,13/1,20;
  **NN4 2,26/5,80/1,35** → 27,1% anualizado. Equal-weighted NN4:
  **Sharpe 2,45** (1,69 excluyendo micro-caps bajo el percentil 20
  NYSE).
- **Alphas** (FF5+momentum): VW NN4 1,76%/mes (t=6,00, IR 1,18); EW
  3,08%/mes (IR 2,40). MDD NN4: 51,8% VW / 14,7% EW; peor mes −9,01%
  (EW). Turnover 110–130%/mes. **Costes NO deducidos.**
- Predictores dominantes: reversal corto, momentum, liquidez,
  volatilidad, ratios de valoración.

Fuentes: [GKX2020 loc:Tab.1,Tab.5,Tab.7,Tab.8]

## <a name="p2"></a>P2 — Fischer-Krauss (k=10, dic 1992–oct 2015)
- **Retorno diario ANTES de costes**: LSTM 0,46%; RF 0,43%; DNN
  0,32%; LOG 0,26% (mercado 0,04%). Sharpe anualizado: **LSTM 5,83**;
  RF 5,00; DNN 2,43; LOG 1,70.
- **DESPUÉS de costes (5 pb/media vuelta)**: LSTM **0,26%/día**
  (t Newey-West = 9,58), RF 0,23%, DNN 0,12%, LOG 0,06% (n.s.).
  Sharpe: **LSTM 2,34**; RF 1,87; DNN 0,52; LOG 0,10 (mercado 0,35).
  Retorno anualizado neto: LSTM 82,29%; RF 67,87%; DNN 24,60%.
  MDD neto LSTM: 52,33% (el menor).
- Accuracy: LSTM 54,3% / RF 53,8% / DNN 53,7% / LOG 52,2%;
  Diebold-Mariano: LSTM superior (p=0,0143 vs RF; 0,0037 vs DNN).
- **Decadencia (verificada)**: 1993–2000 muy alto; 2001–09 moderado;
  **2010–15 ≈ CERO neto** (RF pierde). 52–54% de la varianza
  explicada por una regla de reversal corto.

Fuente: [FK2018 loc:Fig.3,Tab.3,§4.1.4]

## <a name="p3"></a>P3 — ZZR DRL (test 2011–2019, neto de 20 pb)
- **Portafolio 50 contratos** (E(R) anualizada / Sharpe): DQN
  1,258/**1,288** (Sortino 2,22, Calmar 1,03, 54,3% días positivos);
  A2C 1,024/1,050; PG 0,740/0,754 — vs long-only 0,055/0,058,
  momentum-1a 0,429/0,441, MACD 0,089/0,091.
- Por clase: commodities DQN 0,703/0,723 (long −0,710/−0,726); índices
  long-only gana 0,668/0,688 (DQN 0,629/0,648); renta fija DQN
  0,908/0,935; FX DQN 0,528/0,546 (long −0,344).
- **Robustez a costes**: DQN y A2C rentables hasta ~25 pb (~$3,5 por
  contrato); baselines colapsan mucho antes. Win rate 0,49–0,54.
- Nota: MDD (0,002–0,066) es de retornos vol-targeted; el σ objetivo
  numérico NO está declarado.

Fuente: [ZZR2020 loc:Tab.2,Tab.3,Fig.2]

## <a name="p4"></a>P4 — DeepLOB
- **FI-2010 setup-2** (train días 1–7, test 8–10; acc/F1 %): k=10
  **84,47/83,40** (C(TABL) 84,70/77,63; LSTM 66,33 F1; CNN-I 55,21);
  k=20 F1 72,82 (C(TABL) 66,93); k=50 F1 80,35 (C(TABL) 78,44).
- **FI-2010 setup-1** (9-fold anclado): k=10 acc 78,91/F1 77,66;
  k=50 75,01/74,96; k=100 76,66/76,58.
- **LSE out-of-sample** (acc/F1): k=20 70,17/70,15; k=50 63,93/63,49;
  k=100 61,52/60,65. **Transferencia a 5 acciones jamás vistas**:
  k=20 68,62/68,48 — degradación casi nula (evidencia de
  universalidad).
- Simulación de trading: "beneficios consistentes y t-values
  significativos" [PnL exactos NO EXTRAÍDOS — UNVERIFIED].

Fuente: [DEEPLOB2019 loc:tablas setup1/setup2,§V LSE]

## <a name="p5"></a>P5 — Momentum Transformer (señal a 15% vol)
- **Sharpe promedio 1995–2020**: long-only 0,51; TSMOM 1,03;
  LSTM-DMN 1,70; Transformer 1,41; Informer 1,72; **TFT
  decoder-only 2,54; TFT+CPD 2,62** (MDD 1,29%, Calmar 3,22, 57,7%
  días positivos).
- **2015–2020**: TSMOM 0,24; LSTM 0,82; **TFT 1,71; TFT+CPD 2,00**.
- **COVID (ene–oct 2020)**: LSTM **−1,50**; Transformer canónico
  3,38; Decoder-Only 3,01 (retorno 8,02%); TFT+CPD 2,47.
- **Costes (2015–20)**: TFT+CPD Sharpe 2,00/1,61/1,22/0,83/0,44/
  0,04/−0,35 a C=0/0,5/1/1,5/2/2,5/3 pb — break-even ~2,5–3 pb;
  LSTM cae a −1,05 ya con 3 pb.

Fuente: [WOOD2022 loc:Exh.3,Exh.10]

## <a name="menciones"></a>Menciones (cifras clave verificadas)
- **Deng FDDR** (IF, objetivo TP; beneficio total en puntos / SR%):
  FDDR 3.256,6/11,2 vs DDR 2.785,4/9,3, buy&hold 739. A coste 2 pt:
  FDDR **+774,2 con 376 trades** vs LSTM predictivo −822,8 con
  ~2.800 trades. Win rate ≈0,51–0,53.
- **Jiang EIIE** (con 0,25% comisión; fAPV = múltiplo final en ~51
  días): BT1 CNN **29,7×** (SR/30min 0,087; MDD 0,224); BT2 8,0×;
  BT3 bRNN **47,1×**/CNN 31,7×; mejor no-DL (RMR) 7,0×.
- **Kronos**: RankIC 0,0345 (vs 0,0179 mejor TSFM); IC retorno
  0,0665; volatilidad MAE 0,0384/R² 0,249; backtest CSI300/800 el
  mejor AER/IR de 25 baselines [valores exactos solo en figuras —
  UNVERIFIED].
- **FinAgent** (jun 2023–ene 2024; ARR%/Sharpe/MDD%): TSLA
  **92,27/2,01/12,14**; GOOGL 56,15/1,78/8,45; ETHUSD
  43,08/1,18/12,72; baseline B&H TSLA 37,4/0,72. Ablación TSLA:
  39,0→57,2→89,3→92,3%.
- **FinMem** (oct 2022–abr 2023; retorno acum.%/Sharpe/MDD%): TSLA
  **61,78/2,68/10,80** vs B&H −18,63/−0,54/55,32; COIN 34,98/0,72.
  ⚠ Ambos agentes LLM: ventana de test dentro del cutoff del LLM —
  riesgo de contaminación señalado por benchmarks 2025–26.
- **TLOB** (F1 h=10/20/50/100): FI-2010 81,55/82,68/90,03/92,81 (SOTA
  +3,7 medio); pero TSLA real 60,5/49,7/43,5/39,8 — la brecha
  FI-2010→realidad cuantificada. **LOBCAST 2024** (control negativo):
  los F1 altos de FI-2010 no se transfieren a datos frescos ni a
  beneficio neto.
- **Sirignano-Cont**: modelo universal > modelos por-acción en 25/25
  acciones jamás vistas (+1,45% medio); estable 18 meses sin
  recalibrar; entrenar con 19 meses > 1/3/6 meses en 100% de 50
  acciones (+7,2/+3,7/+1,6%).

Fuentes: [DENG2017 loc:Tab.I-III,Fig.7], [JIANG2017 loc:Tab.1-2,§5], [KRONOS2025 loc:§4,Fig.4], [FINAGENT2024 loc:Tab.4,§5], [FINMEM2023 loc:§4,Tab.resultados], [TLOB2025 loc:Tab.8,§5], [SIRCONT2019 loc:Tab.1,§3-4], [LOBCAST2024 loc:benchmark propio], [STOCKBENCH2025 loc:benchmark propio]

## <a name="nuestros"></a>Nuestros experimentos (métricas y estado)
- **Métricas de selección (nunca de reporte final)**: composite del
  comparador jerárquico pareado sobre monitor-2022/inner-2023 (RAP =
  retorno − λ·drawdown, λ=1; gap penalizado β=0,25); gates de
  actividad episódica (sentinela −100 solo a episodios sin trades).
- **Métrica de decisión (endpoint)**: UNA evaluación post-selección
  en outer-2024: retorno ajustado por riesgo (retorno −
  1,0·max_drawdown) con filas (2.196) y sha256 del CSV atados y
  re-hasheados pre-evaluación; crudos visibles: retorno, drawdown,
  Sharpe, trades, exposición, diversidad de acción.
- **Reglas pre-declaradas**: dirección FOR/AGAINST exige ≥3/4
  semillas del mismo signo + mediana concordante; EN-W y EN-F jamás
  fusionados; 4 semillas = direccional, no concluyente;
  `treatment_divergence` obligatorio (easy inerte ⇒ uninformative).
- **Resultados**: campaña P1 4×3 EN CURSO (sin veredicto; primeros
  hechos por-brazo registrados, incluida la inercia del tratamiento
  easy a escala real en las semillas 303/404 — 148/148 tensores
  idénticos). Screens de meseta-LR CERRADOS: INCONCLUSIVE (tardío) y
  SIGNAL_AGAINST (temprano) — spec rechazada como gen DOIN.
- **Demo en vivo** (muestras pequeñas, honestidad ante todo): MT5
  ETHUSD 23 round-trips; Alpaca SPY 10+1; IBKR USD.CAD 12
  (suspendido). Sin reclamos de rentabilidad con n así — son rutas de
  VALIDACIÓN del simulador, la prioridad P1 del dueño.

Fuente: [OURS-PIPELINE loc:configs+manifiestos verificados por ejecución]

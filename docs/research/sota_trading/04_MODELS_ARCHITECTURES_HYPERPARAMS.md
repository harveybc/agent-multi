# 04 — Modelos, arquitecturas e hiperparámetros exactos

## Tabla de contenido
- [P1](#p1) · [P2](#p2) · [P3](#p3) · [P4](#p4) · [P5](#p5)
- [Menciones](#menciones) · [Nuestros experimentos](#nuestros)

## <a name="p1"></a>P1 — Gu-Kelly-Xiu
- 13 familias: OLS(+Huber), OLS-3, PLS, PCR, elastic net(+H), GLM con
  group lasso(+H), Random Forest, GBRT(+H), NN1–NN5.
- **Redes (pirámide geométrica)**: NN1=32; NN2=32,16; NN3=32,16,8;
  NN4=32,16,8,4; NN5=32,16,8,4,2. ReLU, totalmente conectadas,
  batch-norm tras cada ReLU.
- **Entrenamiento NN (Internet Appendix Tabla A.5, exacto)**: L1
  λ1∈(1e-5,1e-3); Adam LR∈{0.001,0.01}; batch 10.000; 100 épocas;
  early-stop paciencia 5; **ensemble de 10 semillas promediado**.
- RF: 300 árboles, profundidad 1–6, features/split
  ∈{3,5,10,20,30,50}; GBRT: profundidad 1–2, 1–1000 árboles,
  LR∈{0.01,0.1}; ENet ρ=0,5; Huber ξ = cuantil 99,9%.

Fuentes: [GKX2020 loc:§1.3-1.7], [GKX2020-IA loc:Tab.A.5]

## <a name="p2"></a>P2 — Fischer-Krauss
- **LSTM**: 1 capa, 25 neuronas ocultas, dropout recurrente 0,1 (Gal
  & Ghahramani), densa 2 + softmax. **2.752 parámetros LSTM**.
- Entrenamiento: cross-entropy, RMSprop, early stopping (máx 1.000
  épocas, paciencia 10, restaura mejores pesos). Keras/TensorFlow.
  [Batch size NO DECLARADO].
- Benchmarks: RF 1.000 árboles prof. 20 m=√p; DNN 31-31-10-5-2
  maxout 2 canales, dropout 0,5, L1 1e-5 (H2O); logística L2 (100
  valores 1e-4..1e4, 5-fold CV, L-BFGS).

Fuente: [FK2018 loc:§3.3-3.4]

## <a name="p3"></a>P3 — Zhang-Zohren-Roberts DRL
- **Arquitectura común**: LSTM 2 capas 64→32, Leaky-ReLU — idéntica
  para Q-network, políticas, actor y crítico.
- **Algoritmos**: DQN (con target fijo, Double, Dueling); Policy
  Gradient vanilla; A2C síncrono. Acciones: {−1,0,1} discreto
  (DQN/PG); [−1,1] continuo (A2C).
- **Hiperparámetros (Tabla 1)**: DQN — LR crítico 1e-4, Adam, batch
  64, **γ=0,3**, replay 5.000, target cada τ=1.000 pasos, bp
  entrenamiento 0,0020. PG — LR 1e-4, γ=0,3. A2C — LR crítico 1e-3,
  actor 1e-4, batch 128, γ=0,3. [Épocas NO DECLARADAS].

Fuente: [ZZR2020 loc:Tab.1,§4]

## <a name="p4"></a>P4 — DeepLOB
- **CNN estructural**: bloque 1 conv 1×2 stride 1×2 (precio-volumen
  por nivel) → bloque 2 conv 1×2 stride 1×2 (bid-ask) → bloque 3 conv
  1×10 (todos los niveles); Leaky-ReLU α=0,01; zero-padding; sin
  pooling fuera del Inception.
- **Inception @32 filtros**: conv 1×1 → ramas 3×1 y 5×1 + rama
  max-pool (stride 1).
- **LSTM 64 unidades** → softmax 3 clases. **~60.000 parámetros**
  (vs 768k del CNN-I baseline).
- Entrenamiento: ADAM **LR 0,01, epsilon 1**, batch 32,
  cross-entropy categórica, early-stop cuando validación estanca 20
  épocas (~100 épocas FI-2010, ~40 LSE).

Fuente: [DEEPLOB2019 loc:§IV arquitectura]

## <a name="p5"></a>P5 — Momentum Transformer
- **Decoder-Only Temporal Fusion Transformer**: Variable Selection
  Network → LSTM (procesamiento local/encoding posicional) → atención
  multi-cabeza interpretable ENMASCARADA (pesos de value compartidos
  entre cabezas) → FFN, con skip-connections aprendibles.
- Salida: posición z∈(−1,1) por activo/día; pérdida = −Sharpe
  anualizado; mini-batch SGD.
- Comparado contra: Transformer canónico, Decoder-Only, Conv
  Transformer, Informer, LSTM-DMN, TSMOM clásico, long-only.
- [Anchos de capa/dropout exactos viven en los apéndices B/C del
  paper — NO EXTRAÍDOS: UNVERIFIED].

Fuente: [WOOD2022 loc:§III-IV]

## <a name="menciones"></a>Menciones
- **Deng FDDR**: fuzzy(R^150) → 4 capas sigmoides 128-128-128-20 →
  capa RL directa δ_t = tanh(⟨w,F⟩+b+u·δ_{t−1}), δ∈{1,0,−1}; BPTT
  task-aware con enlaces virtuales; 100 épocas, decay LR 0,97;
  robustez recomienda τ=2 stacks, N=128, l=3. [LR inicial NO
  DECLARADO numéricamente].
- **Jiang EIIE**: evaluadores idénticos independientes con pesos
  compartidos por activo + Portfolio-Vector Memory (pesos previos
  inyectados pre-softmax) + sesgo de efectivo. CNN: conv 1×3 (2
  mapas) → conv 1×48 (20 mapas) → +w_{t−1} → conv 1×1 → softmax;
  variante RNN/LSTM 20 unidades×50 pasos. Gradiente determinista
  sobre R = media de ln(μ_t·y_t·w_{t−1}); batches de inicio
  geométrico P_β. [λ, batch, β, pasos: solo en el código PGPortfolio
  — UNVERIFIED en el paper].
- **Kronos**: decoder-only autorregresivo sobre tokens; familias
  24,7M/102,3M/499,2M params (8L d512 / 12L d832 / 18L d1664); AdamW
  coseno, warmup 15k pasos, LR 1e-3/5e-4/2e-4, WD 0,01/0,05/0,10;
  inferencia T=0,6 (forecast) con N=10 rollouts MC promediados.
- **FinMem/FinAgent**: agentes LLM prompteados sin gradiente
  (GPT-4-Turbo T=0,7 top-K=5 en FinMem; backbone de FinAgent NO
  DECLARADO en el cuerpo).
- **TLOB**: bloque dual — atención temporal + atención sobre features
  + MLPLOB como FFN; secuencia 128; TLOB 4 capas LR 1e-4; 1 cabeza;
  Adam. [Parámetros totales/batch/épocas NO DECLARADOS].
- **Sirignano-Cont**: 3 capas LSTM + 1 feed-forward ReLU, 50
  unidades/capa (variante 150 mejor); SGD asíncrono distribuido en
  ~25 nodos GPU (500 para modelos por-acción). [LR/batch/épocas NO
  DECLARADOS].

Fuentes: [DENG2017 loc:Tab.I-III,Fig.7], [JIANG2017 loc:Tab.1-2,§5], [KRONOS2025 loc:§4,Fig.4], [FINAGENT2024 loc:Tab.4,§5], [FINMEM2023 loc:§4,Tab.resultados], [TLOB2025 loc:Tab.8,§5], [SIRCONT2019 loc:Tab.1,§3-4], [LOBCAST2024 loc:benchmark propio], [STOCKBENCH2025 loc:benchmark propio]

## <a name="nuestros"></a>Nuestros experimentos
- **Algoritmo**: SAC (Stable-Baselines3), política MlpPolicy (flat)
  para la identidad P1; entropía automática (log_ent_coef aprendido).
- **Red**: net_arch (256, 256) actor y críticos gemelos (defaults del
  plugin `sac_agent`); observación aplanada (FlattenObservation) en
  la identidad flat-MLP.
- **Hiperparámetros de campaña (materializados en manifiestos de
  lanzamiento)**: learning_rate FIJO 3e-4 (la meseta-LR fue rechazada
  con evidencia — dos screens); batch 64; learning_starts 128; buffer
  200.000; epoch_timesteps 20.000; máx 2.000 épocas/fase; paciencia
  60 inactiva antes de la época 40; [gamma/tau del plugin en defaults
  SB3-compatibles declarados en `sac_agent.plugin_params`].
- **Extractor agrupado (rama experimental separada, NO en P1)**:
  Dict-observation preservada; ramas semánticas — TCN causal
  (retornos 16; canales [64,64], kernel 3, dilatación 2), Transformer
  (tendencia 23; d=64, 4 cabezas, 2 capas), GRU (osciladores 9 y
  volumen 6; hidden 64), TCN (volatilidad 29), MLP (estado 4);
  fusión gated (común 64 → salida 128); §1 verificado (ejes, formas,
  gradientes, refusal de orden de columnas); §2–§6 en curso.
- **Continuidad probada**: bundles coherentes por checkpoint con hash
  por-tensor (148 tensores verificados exactos tras cada warm-start);
  EN-F carga modelo+replay de la MISMA época seleccionada.

Fuente: [OURS-PIPELINE loc:configs+manifiestos verificados por ejecución]

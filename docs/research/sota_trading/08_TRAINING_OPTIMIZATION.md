# 08 — Entrenamiento y optimización: régimen, búsqueda de
# hiperparámetros, ensembles, re-entrenamiento

## Tabla de contenido
- [P1](#p1) · [P2](#p2) · [P3](#p3) · [P4](#p4) · [P5](#p5)
- [Menciones](#menciones) · [Nuestros experimentos](#nuestros)

## <a name="p1"></a>P1 — Gu-Kelly-Xiu
- **Búsqueda de hiperparámetros**: sí, sistemática — cada modelo
  ajusta sus hiperparámetros (λ de L1, LR, profundidad de árboles,
  nº de árboles, features/split…) sobre la VENTANA DE VALIDACIÓN de
  12 años, separada del test.
- **Ensembles**: 10 redes con semillas aleatorias distintas,
  pronósticos PROMEDIADOS — reduce varianza de inicialización.
- **Re-entrenamiento**: anual, ventana de train expansiva +
  validación rodante — 30 re-ajustes en el periodo de test.
- **Early stopping**: paciencia 5 sobre validación; batch-norm; 100
  épocas máximo; batch 10.000.

Fuentes: [GKX2020 loc:§1.2,§2.2], [GKX2020-IA loc:Tab.A.5]

## <a name="p2"></a>P2 — Fischer-Krauss
- **Búsqueda**: mínima y declarada — arquitectura LSTM fijada (25
  unidades) sin grid search reportado; la logística sí busca L2 en
  100 valores con 5-fold CV.
- **Ensembles**: NO para el LSTM (una red por periodo de estudio).
- **Re-entrenamiento**: CADA 250 días de trading (23 veces) — modelo
  fresco por periodo de estudio, entrenado sobre los 750 días
  previos.
- **Early stopping**: split interno 80/20, paciencia 10, máx 1.000
  épocas, restaura mejores pesos.

Fuente: [FK2018 loc:§3.3-3.4]

## <a name="p3"></a>P3 — Zhang-Zohren-Roberts DRL
- **Búsqueda**: hiperparámetros de la Tabla 1 fijos [procedimiento de
  selección NO DECLARADO]; γ=0,3 notablemente bajo (horizonte
  efectivo corto, deliberado para trading).
- **Ensembles**: no de semillas; sí agregación por CLASE de activo
  (un modelo por clase, mejor que por contrato).
- **Re-entrenamiento**: cada 5 años con ventana expansiva; parámetros
  CONGELADOS durante los 5 años de trading OOS.
- Replay 5.000 (pequeño), target-network cada 1.000 pasos.

Fuente: [ZZR2020 loc:Tab.1,§4]

## <a name="p4"></a>P4 — DeepLOB
- **Búsqueda**: no reportada como grid; arquitectura diseñada por
  estructura del problema; LR 0,01/ε=1 de ADAM declarados.
- **Ensembles**: no.
- **Re-entrenamiento**: no rodante — un entrenamiento por dataset
  (FI-2010: ~100 épocas; LSE: ~40), early-stop paciencia 20 sobre
  accuracy de validación.
- Transferencia sin fine-tuning a acciones nunca vistas (el hallazgo
  de universalidad sustituye al re-entrenamiento).

Fuente: [DEEPLOB2019 loc:§IV-D entrenamiento]

## <a name="p5"></a>P5 — Momentum Transformer
- **Búsqueda**: longitud de secuencia elegida por validación (1 año;
  vs ~63 días óptimo del LSTM-DMN); [grid completo en apéndices — NO
  EXTRAÍDO].
- **Ensembles/semillas**: CADA experimento repetido 5 VECES, medias
  reportadas — control explícito de varianza de semilla.
- **Re-entrenamiento**: walk-forward expansivo cada 5 años
  (1990–95→test 95–00; …→2020).
- Pérdida = −Sharpe directamente (optimización del objetivo económico
  end-to-end, sin proxy de clasificación).

Fuente: [WOOD2022 loc:§V protocolo]

## <a name="menciones"></a>Menciones
- **Deng FDDR**: 5 entrenamientos por ventana, se queda el mejor en
  validación; re-entrenamiento ONLINE cada 5.000 barras con
  warm-start; inicialización por autoencoders; 100 épocas, LR decay
  0,97.
- **Jiang EIIE**: hiperparámetros elegidos UNA vez sobre un rango de
  CV previo a los tres backtests [valores NO TABULADOS]; batches con
  inicio muestreado geométricamente (favorece datos recientes);
  entrenamiento online continuo durante el backtest.
- **Kronos**: pre-entrenamiento masivo único (12B velas) + escalado
  de familia (24,7M→499M params); warmup 15k pasos, coseno; N=10
  rollouts MC en inferencia.
- **Sirignano-Cont**: SGD asíncrono distribuido (~25 nodos GPU
  universal; ~500 para por-acción); hallazgo: entrenar con TODA la
  historia (19 meses) > ventanas recientes en el 100% de los casos.

Fuentes: [DENG2017 loc:Tab.I-III,Fig.7], [JIANG2017 loc:Tab.1-2,§5], [KRONOS2025 loc:§4,Fig.4], [FINAGENT2024 loc:Tab.4,§5], [FINMEM2023 loc:§4,Tab.resultados], [TLOB2025 loc:Tab.8,§5], [SIRCONT2019 loc:Tab.1,§3-4], [LOBCAST2024 loc:benchmark propio], [STOCKBENCH2025 loc:benchmark propio]

## <a name="nuestros"></a>Nuestros experimentos
- **Búsqueda de hiperparámetros**: HOY INEXISTENTE para la línea SAC
  P1 — LR fija 3e-4 (la adaptativa fue evaluada y rechazada con dos
  screens pareados), net_arch (256,256) heredada, batch 64, buffer
  200k, γ/τ en defaults del plugin: NINGUNO buscado sistemáticamente.
  El plan DOIN (optimización distribuida de genes) existe pero NO ha
  corrido para esta línea.
- **Ensembles**: no hay ensemble de semillas por brazo — cada brazo
  de la campaña es UNA semilla; la campaña usa 4 semillas como
  réplicas direccionales, no como ensemble de política.
- **Re-entrenamiento**: NO HAY esquema rodante — un único
  entrenamiento con fit→2022 y evaluación outer en 2024 (brecha de
  ~2 años entre fin de train y evaluación de decisión, sin
  adaptación). Contrasta con: GKX anual, Fischer-Krauss cada 250
  días, ZZR/Wood cada 5 años, Deng cada 5.000 barras.
- **Early stopping**: paciencia 60 inactiva antes de la época 40
  sobre el comparador jerárquico monitor/inner — bien especificada y
  con bundles coherentes por mejora (fortaleza propia).
- **Régimen de curriculum (easy→normal)**: la hipótesis insignia de
  la campaña; la parametrización actual de la relajación demostró
  ser INERTE (148/148 tensores idénticos easy vs normal a escala
  real en las semillas verificadas) — el factor experimental activo
  restante es la continuidad de replay.

Fuente: [OURS-PIPELINE loc:configs+manifiestos verificados por ejecución]

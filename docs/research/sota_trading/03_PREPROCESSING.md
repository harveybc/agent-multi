# 03 — Preprocesamiento: normalización, etiquetas/recompensas, splits

## Tabla de contenido
- [P1](#p1) · [P2](#p2) · [P3](#p3) · [P4](#p4) · [P5](#p5)
- [Menciones](#menciones) · [Nuestros experimentos](#nuestros)

## <a name="p1"></a>P1 — Gu-Kelly-Xiu
- **Normalización**: cada característica RANKEADA cross-seccionalmente
  por periodo y mapeada a [−1,1]. Faltantes → mediana cross-seccional
  del mes.
- **Anti-lookahead**: características mensuales rezagadas 1 mes;
  trimestrales ≥4 meses; anuales ≥6 meses.
- **Split**: 18 años train (1957–74) + 12 validación (1975–86) + 30
  test (1987–2016). Re-ajuste UNA vez al año; train expansivo,
  validación rodante de 12 años. SIN cross-validation (orden temporal
  preservado). Objetivo: retorno excedente del mes siguiente
  (regresión).

## <a name="p2"></a>P2 — Fischer-Krauss
- **Normalización**: retornos estandarizados con μ/σ calculados SOLO
  sobre el conjunto de entrenamiento del periodo de estudio.
- **Etiqueta**: binaria — clase 1 si el retorno t+1 de la acción ≥
  mediana cross-seccional de todas las acciones en t+1 (Takeuchi &
  Lee 2013).
- **Split**: periodos de estudio rodantes de 750 días train + 250 días
  trading NO solapados; 23 periodos 1990–2015; ~380.000 secuencias
  solapadas de 240 días por periodo (~255k train, ~125k OOS); dentro
  del train, 80/20 para early stopping.

## <a name="p3"></a>P3 — Zhang-Zohren-Roberts DRL
- **Volatility scaling en todo**: posiciones escaladas por
  σ_target/σ_{t−1} (σ = EWM std 60 días) — normaliza exposición y
  recompensas entre 50 contratos de precios heterogéneos.
- **Recompensa** (aditiva): R_t = μ[A_{t−1}·(σ_tgt/σ_{t−1})·r_t −
  bp·p_{t−1}·|Δ posición escalada|], con **bp = 0,0020 (20 puntos
  básicos) EN ENTRENAMIENTO** — coste deliberadamente punitivo como
  regularizador de rotación.
- **Split**: ventana expansiva, re-entrenamiento cada 5 años, 5 años
  fijos de trading OOS cada vez; test 2011–2019. Un modelo por CLASE
  de activo (agrupar dentro de la clase mejoró resultados).

## <a name="p4"></a>P4 — DeepLOB
- **Normalización**: z-score por feature usando μ/σ de los **5 días
  hábiles previos** (rodante). En FI-2010: la normalización z-score
  provista por el benchmark.
- **Etiquetas** (3 clases: sube/estable/baja): movimiento porcentual
  del mid-price suavizado. FI-2010 (Eq. 3): l = (m₊(k) − p_t)/p_t con
  m₊ = media de los próximos k mid-prices; LSE (Eq. 4): l = (m₊ −
  m₋)/m₋ con suavizado en AMBOS lados (media de k pasados y k
  futuros) porque las etiquetas Eq. 3 en LSE eran "rather stochastic".
  Clase por umbral α: FI-2010 α=0,002; LSE α elegido para balancear
  clases [valor numérico NO DECLARADO]. Horizontes k = 10/20/50/100
  eventos.
- **Split LSE**: 6 meses train / 3 validación / 3 test (temporal).

## <a name="p5"></a>P5 — Momentum Transformer
- **Normalización**: retornos escalados por σ ex-ante (EWM 60 días);
  el PORTAFOLIO se lleva a volatilidad objetivo 15% anualizada
  (convención TSMOM).
- **Pérdida = objetivo de trading**: Sharpe anualizado negativo de los
  retornos vol-escalados del portafolio (Eq. 11) — sin etiqueta de
  predicción; supervisión directa sobre la economía.
- **Split**: walk-forward expansivo — train/val 1990–95 → test
  1995–2000; expandir → test 2000–05; … hasta 2020. CADA experimento
  repetido 5 veces (semillas), medias reportadas.

## <a name="menciones"></a>Menciones
- **Deng FDDR**: capa fuzzy (3 membresías gaussianas por input,
  centros/anchos por k-means) → R^150; inicialización por
  autoencoders capa a capa; esquema online rodante: 15.000 barras
  iniciales (12k train + 3k validación, mejor de 5 entrenamientos),
  trade 5.000 fuera de muestra, desliza 5.000 (warm-start); S&P:
  train 2.000 días, refresco cada 100.
- **Jiang EIIE**: cada precio DIVIDIDO por el último cierre del
  tensor (v_{t−i} ⊘ v_t) — "empíricamente mejor que otras";
  monedas jóvenes: NaN → movimientos planos falsos; sin validación
  interna en backtests.
- **Kronos**: tokenizador BSQ (autoencoder transformer 3+3 capas,
  d=256, k=20 bits, subtokens jerárquicos coarse/fine); corte
  temporal ESTRICTO train ≤ jun 2024 / test ≥ jul 2024.
- **TLOB**: capa BiN (normalización bilineal APRENDIDA) como entrada;
  etiquetado propio desacoplando suavizado de horizonte (w₊/w₋ con
  desplazamiento h) — OJO: infla F1 de horizonte largo vs etiquetas
  clásicas FI-2010.
- **Sirignano-Cont**: hallazgo clave — normalizaciones estándar (por
  volatilidad, nivel de precio, spread) NO mejoran el entrenamiento;
  particionar por sector/tick-size tampoco.

## <a name="nuestros"></a>Nuestros experimentos
- **Normalización**: z-score RODANTE por feature con ventana de 256
  barras H4 (`feature_scaling: rolling_zscore`,
  `feature_scaling_window: 256`). CAVEAT verificado y documentado: las
  primeras ~256 barras de un episodio emiten CEROS hasta llenar la
  ventana (probes de gradiente deben tomarse post-calentamiento).
- **Splits**: manifiesto de roles ANIDADO verificado (contrato
  `eth_nested_split_contract_v1.json`): fit_train→2022 /
  train_monitor 2022 / inner_validation 2023 / outer_validation 2024
  / sealed_test 2025 estructuralmente inmaterializado. Filas de
  contexto causal (context prefixes) inicializan la observación pero
  JAMÁS puntúan ni actualizan (wrapper de prefijo instalado antes del
  rollout). Hash del CSV re-verificado antes de cada evaluación outer.
- **Recompensa/selección**: recompensa por paso económica del env +
  objetivo episódico de actividad/economía aceptado (sentinela −100
  SOLO a episodios con cero trades; NOP intra-episodio jamás
  penalizado; transformación de pérdida acotada m/(1+m)); selección
  de checkpoints por comparador jerárquico pareado
  (`paired_generalization_weekly_v1`) sobre monitor 2022 + inner
  2023; coste de transacción del env dentro de la recompensa.
- **Anti-lookahead estructural**: sellado 2025 inaccesible fuera de
  modo release; evaluación outer post-selección únicamente; holdout
  de 40 días renombrado `diagnostic_holdout` en la ruta de screens.

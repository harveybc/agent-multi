# 10 — Gap matrix orientado a decisiones (WP2, orden 2026-08-24)

Cada fila es una DECISIÓN ABIERTA del programa, no un paper famoso.
Campos por candidato: mecanismo, evidencia a favor/en contra,
similitud con ETH H4, insumos que no tenemos, coste computacional,
experimento de falsación más barato, colisión con trabajo vigente.
Ningún mecanismo se llama SOTA sin decisión y comparador definidos.

## Tabla de contenido
- [D1 Representación de la acción](#d1)
- [D2 Riesgo: restricciones/distribucional](#d2)
- [D3 Offline RL / evaluación off-policy](#d3)
- [D4 POMDP/recurrencia y regímenes](#d4)
- [D5 Cadencia de re-entrenamiento](#d5)
- [D6 Selección estadística multi-trial](#d6)

## <a name="d1"></a>D1 — ¿Qué contrato de acción debe emitir el agente?
**Decisión**: signo-solo (actual, umbral 0) vs exposición objetivo
continua vs ternario con deadband/histéresis vs close/hold explícito.
- **Candidato A — posición objetivo continua con penalización de
  rotación** (DMN): la red emite posición ∈[−1,1], término |Δposición|
  en la pérdida. A favor: >2× sobre TSMOM clásico; el regularizador
  internaliza el coste. En contra: supervivencia a costes solo hasta
  2–3 pb — fino frente a taker fees cripto (5–10 pb); supervisado, no
  RL. Similitud: media (futuros diarios; la formulación transfiere al
  head de SAC). Insumos faltantes: ninguno. Coste: trivial.
  Falsación más barata: añadir |Δposición|·fee al reward del SAC a
  fees H4 realistas; si el Sharpe neto no mejora, deadband/close-hold
  tampoco pagarán. Colisión: compite con el screen de acción WP4-B.
- **Candidato B — no-trade region (teoría)**: la anchura óptima de la
  zona de no-operación crece con el coste incluso para costes
  pequeños. Justifica histéresis a priori. Falsación: incluida en el
  brazo ternario del screen de acción.
- **Candidato C — continuo vs discreto en cripto** (TD3 BTC): reclama
  continuo > discreto; venue débil, costes no declarados —
  evidencia de baja calidad, solo orientativa.

Fuentes: [LIM2019 loc:abstract+§métodos], [MUHLEKARBE2017 loc:§1
no-trade region], [ZZR2020 loc:§3 vol-scaling], [MAJIDI2022
loc:abstract]

## <a name="d2"></a>D2 — ¿Riesgo en el objetivo: restricción, distribucional, o envolvente externa?
**Decisión**: mantener envolvente externa (SL/TP nativos + techos) vs
crítico distribucional CVaR vs SAC restringido lagrangiano.
- **Candidato A — crítico distribucional con dial CVaR** (gas
  natural): C51/IQN maximizando CVaR; bajar α aumenta la aversión
  verificablemente (C51 +32%). En contra: dataset propietario; QR-DQN
  errático; familia DQN discreta. Similitud: estructural buena (un
  instrumento volátil). Falsación: crítico cuantílico en nuestro SAC
  con dos α; si los drawdowns no ordenan monótonos con α, el dial no
  funciona en H4.
- **Candidato B — WCSAC** (safety-critic distribucional + peso
  lagrangiano adaptativo): receta canónica para "SAC + restricción de
  drawdown en CVaR". En contra: cero experimentos financieros;
  supuesto gaussiano dudoso con colas PnL. Insumos: señal de coste
  por paso (incremento de drawdown) — la tenemos. Falsación: WCSAC vs
  lagrangiano en expectativa, un lote de semillas; si las colas de
  drawdown no se estrechan a retorno similar, no paga.
  Colisión: interactúa con la envolvente SL/TP viva — el screen debe
  declarar cómo coexisten (WP4).

Fuentes: [HECHE2025 loc:abstract+resultados], [WCSAC2021 loc:§método]

## <a name="d3"></a>D3 — ¿Sirven offline RL / OPE para seleccionar sin gastar GPU online?
**Decisión**: selección de candidatos por FQE/OPE vs walk-forward
online puro.
- **Candidato A — CQL/IQL/TD3+BC sobre trayectorias históricas**: la
  evidencia financiera es débil (preprint monoautor con datos sin
  declarar); el shape del pipeline coincide con el nuestro. Insumos
  faltantes: definición de la política de comportamiento que generó
  las trayectorias. Falsación: IQL/TD3+BC sobre replays de checkpoints
  SAC existentes vs SAC online en el mismo split; si no igualan,
  offline añade riesgo sin beneficio.
- **Candidato B — FQE para ranking de candidatos**: un ajuste extra de
  crítico por candidato (barato). Cuidado: importance sampling explota
  en varianza con episodios largos H4. Falsación (LA MÁS BARATA DEL
  PROGRAMA): FQE sobre checkpoints congelados vs PnL walk-forward
  realizado; correlación de rangos ≈0 mata la selección OPE en un
  experimento. Colisión: ninguna — usa artefactos ya existentes.

Fuentes: [YUN2024 loc:abstract], [BANDI2020 loc:abstract]

## <a name="d4"></a>D4 — ¿Recurrencia/POMDP o régimen explícito en el estado?
**Decisión**: ventana MLP actual vs SAC recurrente vs probabilidades
de régimen en el estado.
- **Candidato A — SAC recurrente bien hecho** (Ni et al.): RNNs
  SEPARADAS actor/crítico + longitud de contexto afinada iguala
  métodos especializados en 18/21 POMDPs. En contra: cero entornos
  financieros; sensible al tuning; varias veces más lento. Falsación:
  sustituir encoders por GRUs separadas a 2 longitudes de contexto;
  si el PnL de validación no mejora, la ventana ya resume la
  historia. Colisión: compite con el extractor agrupado — entra al
  screen de arquitectura, no antes.
- **Candidato B — probabilidades de régimen en el estado** (Macrì):
  prob-DDPG (posterior de régimen) > forecast puntual > embedding
  opaco. Falsación barata: HMM 2–3 estados sobre retornos/vol H4,
  añadir posteriors al estado vs (a) nada y (b) forecast; su
  resultado predice el orden. Insumos: solo un filtro HMM — lo
  podemos construir en CPU.

Fuentes: [NI2022 loc:abstract+§5], [MACRI2025 loc:abstract]

## <a name="d5"></a>D5 — ¿Cadencia de re-entrenamiento y cómo actualizar?
**Decisión**: congelado (actual) vs 168h/24h/12h; fresco vs
warm-start vs actualización adaptada.
- **Candidato A — DoubleAdapt** (KDD'23): cada incremento
  walk-forward como tarea de meta-learning (adaptador de datos +
  adaptador de modelo). A favor: SOTA vs rolling naive, código
  mantenido (qlib). En contra: supervisado, métricas IC no PnL neto.
  Falsación: 3 cadencias × {fresco, warm-start, adaptado} a compute
  igual; si adaptado no bate a fresco, se omite la capa meta.
- **Candidato B — warm-start gap + shrink-and-perturb** (Ash-Adams):
  el warm-start alcanza el mismo train loss pero PEOR generalización;
  encoger+perturbar lo repara. En contra: visión/clasificación, no
  RL; en mercados no estacionarios algo de olvido es DESEABLE.
  Falsación: en el primer roll, {fresco, warm, shrink-perturb} mismo
  presupuesto — un ciclo responde si el gap existe aquí. Colisión:
  el screen de re-entrenamiento WP4-C ya contempla los tres brazos.

Fuentes: [DOUBLEADAPT2023 loc:abstract+§método], [ASHADAMS2020
loc:abstract]

## <a name="d6"></a>D6 — ¿Cómo seleccionar entre muchos trials sin autoengaño?
**Decisión**: regla actual (≥3/4 semillas + mediana) vs DSR/SPA/IQM
formales.
- **Candidato A — Deflated Sharpe Ratio**: SR probabilístico contra
  el máximo esperado bajo N trials (con skew/kurtosis). Requiere
  contar N HONESTAMENTE (cada config × semilla × sweep cuenta) —
  nuestro cubo OLAP ya registra trials: el insumo EXISTE. Falsación/
  adopción: computar DSR del campeón vigente con el N verdadero; si
  p<0,95, el campeón es indistinguible de ruido best-of-N — ese
  cálculo es EL experimento más barato e importante del programa.
- **Candidato B — Reality Check / SPA de Hansen**: bootstrap del
  máximo sobre el panel completo de retornos por barra de TODOS los
  candidatos vs benchmark; SPA estudentiza y es más potente.
  Insumos: series de retorno POR BARRA de cada candidato — hay que
  empezar a almacenarlas de forma disciplinada (hoy: trazas por
  corrida, suficiente). Falsación: SPA del sweep existente vs
  buy-and-hold; p alto ⇒ multiplicar brazos es data snooping.
- **Candidato C — IQM + CIs bootstrap (rliable)**: con pocas
  semillas, medias puntuales revierten conclusiones; IQM +
  intervalos estratificados es la práctica correcta. Adopción:
  recomputar el ranking de brazos vigente con IQM/CIs.

Fuentes: [DSR2014 loc:fórmula DSR], [WHITE2000 loc:abstract],
[HANSEN2005 loc:abstract], [RLIABLE2021 loc:abstract+§3]

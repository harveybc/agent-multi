# Propuesta doctoral — Un marco operacional para medir en bits la capacidad cognitiva, la adquisición de conocimiento y la autorreferencia en sistemas individuales y colectivos

**Borrador para revisión de Retsu y Musashi — 2026-09-04**
**Programa objetivo:** Doctorado (modalidad por tesis), con énfasis en
epistemología de sistemas cognitivos y economía del conocimiento
distribuido.

---

## 1. Resumen

Se propone un marco teórico-experimental que trata la *capacidad
cognitiva* como una magnitud física medible en **bits**, y la
*conciencia* — definida operacionalmente como clase de realimentación
del sistema, jamás como experiencia fenoménica — como una **variable de
control metodológico**: dos sistemas solo son comparables en desempeño
si pertenecen a la misma clase, porque un sistema capaz de adquirir
conocimiento durante la operación domina estructuralmente a uno que
solo infiere. El marco unifica tres literaturas maduras que hoy no se
hablan entre sí — la psicometría universal (Hernández-Orallo), la
medición de capacidad de almacenamiento en redes (MacKay, Gardner;
Morris et al. 2025) y la ciencia de indicadores de conciencia (Butlin
et al.) — y las extiende a **sistemas cognitivos colectivos** (equipos,
mercados, tradiciones orales), donde el factor-c de Woolley y la
cognición distribuida de Hutchins ya demostraron que el colectivo es
unidad legítima de medición. La validación es experimental y barata:
perceptrones multicapa y redes recurrentes pequeñas cuya capacidad
teórica en bits se conoce con exactitud, entrenadas sobre datos de
complejidad de Kolmogorov controlada, bajo la disciplina de
predeclaración, controles negativos y verificación adversarial que el
autor practica en su plataforma de experimentación.

## 2. Motivación y problema

Tres preguntas simples carecen hoy de respuesta operacional unificada:

1. **¿Cuánta información almacena un sistema cognitivo, cuánta puede
   almacenar, y qué fracción de esa capacidad usa?** Existen respuestas
   parciales por familia (perceptrón: 2 bits/peso; GPT: ~3.6
   bits/parámetro) pero ninguna metodología aplicable uniformemente a
   individuos, máquinas y colectivos.
2. **¿Cuándo es justa una comparación de desempeño entre dos sistemas
   cognitivos?** Comparar un sistema que aprende en línea contra uno
   estático confunde capacidad con acceso a información; comparar un
   sistema con automonitoreo contra uno sin él confunde representación
   con control. La literatura de evaluación de IA sufre este confusor
   de forma crónica (Hernández-Orallo 2017; Chollet 2019).
3. **¿Las propiedades que llamamos conciencia y autoconciencia añaden
   desempeño medible, o son epifenómenos?** La pregunta es empírica si
   — y solo si — se definen como clases funcionales de realimentación
   y se controla la capacidad en bits.

El problema central: **no existe un marco que fije las clases de
comparación, la moneda (bits) y el protocolo experimental al mismo
tiempo.** Cada literatura tiene una pieza; ninguna tiene el conjunto.

## 3. Estado del arte

### 3.1 Medición de la inteligencia como disciplina

- **Psicometría universal** (Hernández-Orallo & Dowe 2013; *The
  Measure of All Minds*, CUP 2017): medición de capacidades cognitivas
  para el *machine kingdom* — "cualquier sistema cognitivo, individual
  o colectivo, artificial, biológico o híbrido". Define el programa;
  no fija la moneda ni las clases de comparación.
- **Inteligencia como eficiencia de adquisición** (Chollet 2019, *On
  the Measure of Intelligence*): la habilidad exhibida confunde; lo
  medible es la eficiencia con que se adquiere habilidad nueva,
  controlando priors y experiencia. Converge exactamente con nuestro
  confusor №2.
- **Inteligencia universal** (Legg & Hutter 2007): desempeño esperado
  ponderado por 2^−K(entorno) — la complejidad de Kolmogorov ya es
  central en la definición teórica; falta bajarla al laboratorio.

### 3.2 Capacidad en bits

- **Clásicos:** perceptrón = 2 bits/peso (MacKay, que lo leyó como
  codificador de Shannon; Gardner-Derrida 1988 por mecánica
  estadística); Hopfield ≈ 0.138·N; leyes de escalamiento para redes
  modernas (Friedland et al. 2018).
- **Moderno:** Morris et al. 2025 (*How much do language models
  memorize?*): entrenando GPTs sobre **bitstrings uniformemente
  aleatorios** — incompresibles por construcción, K(x) ≈ |x| — separan
  memorización no intencionada de generalización y miden **~3.6
  bits/parámetro** con meseta de saturación; al exceder la capacidad
  empieza el *grokking*. Metodología directamente reutilizable.
- **Complejidad y pesos:** MDL precuencial (Blier & Ollivier 2018),
  información en los pesos (Achille & Soatto 2018), y la norma de
  pesos como longitud de descripción de Kolmogorov (2026).

### 3.3 Conciencia como variable científica

- **Método de indicadores** (Butlin, Long et al. 2023; actualización
  *Trends in Cognitive Sciences* 2025): de las teorías (procesamiento
  recurrente, espacio de trabajo global, orden superior, procesamiento
  predictivo, esquema de atención) se derivan **14 propiedades
  indicadoras computacionales** evaluables en sistemas artificiales —
  el estándar para hablar del tema sin compromiso metafísico.
- **IIT** (Tononi): Φ como medida graduada; las redes puramente
  feedforward tienen **Φ = 0** — la recurrencia es el umbral. Objeción
  conocida: el *unfolding argument* (para toda red recurrente existe
  una feedforward con idéntica función entrada-salida) — que esta
  propuesta ESQUIVA por diseño al definir clases funcionales, no
  fenoménicas (§5.3).
- **Metacognición** (Fleming): automonitoreo medible por
  calibración de confianza — la operacionalización natural de la
  clase autorreferente.

### 3.4 Sistemas cognitivos colectivos

- **Cognición distribuida** (Hutchins 1995, *Cognition in the Wild*):
  la navegación de un buque como cómputo distribuido entre individuos,
  instrumentos y cultura oral — el colectivo es la unidad cognitiva.
- **Factor-c** (Woolley et al., *Science* 2010; meta-análisis 2021 con
  1356 grupos): existe un factor general de inteligencia colectiva,
  débilmente correlacionado con la inteligencia media individual,
  predictivo fuera de muestra; resultados 2024 lo separan en
  dimensiones tipo fluida/cristalizada.
- **Evolución cultural** (Boyd & Richerson; Henrich): la transmisión
  con mutación y cruce de variantes — formalizable como canal de
  Shannon con ruido.
- **Conocimiento distribuido en sociedad** (Hayek 1945, *The Use of
  Knowledge in Society*): el mercado como mecanismo que agrega
  información dispersa que ningún individuo posee — el sistema
  cognitivo colectivo por excelencia, aún sin métricas informacionales
  operacionales.
- **Vida y mente** (Friston; Maturana & Varela): la tesis de
  continuidad vida-mente vía minimización de energía libre deja
  ABIERTA la pregunta de si la vida exige conciencia — nuestra
  pregunta de largo plazo (§6, P5).

## 4. El hueco

Nadie ha unificado: **(i)** clases de conciencia operacionalizadas
como **clases de equivalencia de comparación** (la regla metodológica
que impide comparaciones confundidas); **(ii)** una sola moneda —
bits — aplicada uniformemente a capacidad total, capacidad usada,
memorización, generalización y transmisión cultural; **(iii)**
validación por **ablación controlada a capacidad emparejada** (misma
cuenta de bits, ± recurrencia, ± metacognición, ± aprendizaje en
línea); **(iv)** la extensión de (i)-(iii) a colectivos con las
mismas definiciones. La psicometría universal propuso el territorio;
esta tesis propone el sistema métrico y lo somete a experimento.

## 5. Marco teórico propuesto

### 5.1 Definiciones operacionales

- **Sistema cognitivo** S: proceso físico con estado interno θ que
  mapea historias de entrada a salidas, evaluado sobre una familia de
  tareas T.
- **Capacidad total** C(S): bits máximos de un conjunto de datos
  incompresible que S puede almacenar y recuperar (protocolo Morris:
  meseta de memorización sobre datos aleatorios).
- **Capacidad usada** U(S, D): bits de D efectivamente almacenados
  (memorización no intencionada medible); **ocupación** = U/C.
- **Conocimiento** K̂(S, T): reducción de longitud de descripción de
  las tareas dada θ (MDL precuencial).
- **Eficiencia de adquisición** η: bits de desempeño ganados por bit
  de experiencia consumida (formalización tipo Chollet).
- **Generalización en bits**: U(S, D) − K̂ᶜ(D), donde K̂ᶜ es la
  complejidad de D estimada por compresor de referencia — lo
  almacenado por encima de lo comprimible no existe; el déficit
  respecto del techo es estructura generalizada.

### 5.2 Clases de realimentación (las "clases de conciencia")

- **R0 (nulo):** sin realimentación — inferencia pura feedforward;
  θ fijo durante la operación.
- **R1 (consciente funcional):** realimentación de salidas o estados
  hacia el propio cómputo (recurrencia; memoria de trabajo); puede
  integrar información temporal y aprender en línea.
- **R2 (autoconsciente funcional):** además, realimentación de una
  **representación del propio estado interno** (lectura de sus
  parámetros/activaciones durante la cognición activa: automodelo,
  metacognición, calibración de confianza sobre sí mismo).

**Postulado metodológico (el corazón de la tesis):** las métricas de
desempeño solo se comparan DENTRO de una clase Rk a capacidad C
emparejada; entre clases, lo que se mide es el **valor marginal de la
clase**: ΔRk = desempeño(Rk) − desempeño(Rk−1) a C constante.

### 5.3 Defensa frente al *unfolding argument*

Las clases Rk son **funcionales y arquitecturales**, no fenoménicas:
clasifican el proceso físico real (¿existe el lazo de realimentación
en el sistema implementado?), no la función entrada-salida abstracta.
El unfolding muestra que dos arquitecturas distintas pueden computar
la misma función — irrelevante aquí, porque la tesis no afirma que R1
"sienta": afirma que el LAZO tiene consecuencias medibles en
capacidad, eficiencia de adquisición y costo — afirmación empírica y
falsable a función igual (la ablación a capacidad emparejada es
exactamente ese experimento).

### 5.4 Extensión a colectivos

Un colectivo (equipo, cadena de relevo oral, mercado) es un sistema S
cuyo θ está distribuido en individuos y artefactos. Las mismas
definiciones aplican: C del colectivo (¿cuánta tradición conserva una
cadena de relevo con N eslabones y tasa de error ε? — canal de Shannon
con redundancia), U (repertorio efectivo), η (velocidad de
incorporación de innovaciones), clase R (¿el colectivo monitorea su
propio estado — instituciones de memoria, precios como señal
agregada?). Hipótesis puente con Hayek: el sistema de precios es un
mecanismo R1-colectivo de compresión con pérdida de información
distribuida, cuya eficiencia informacional es medible en simulación.

## 6. Preguntas e hipótesis

- **P1.** ¿Cuál es la capacidad en bits/parámetro de las familias MLP
  y recurrente pequeñas, medida con el protocolo de datos
  incompresibles, y coincide con la teoría clásica? *(H1: MLP ≈ 2
  bits/peso ± factor de profundidad; medible con meseta nítida.)*
- **P2.** ¿La ocupación U/C predice la transición
  memorización→generalización cuando la K del dataset se controla con
  un dial algorítmico? *(H2: la transición ocurre cuando U alcanza
  K̂ᶜ(D), no cuando alcanza C.)*
- **P3.** A capacidad emparejada, ¿ΔR1 y ΔR2 son positivos en tareas
  con estructura temporal y cambio de distribución, y ~nulos en
  tareas estáticas? *(H3: la ventaja de clase existe solo donde hay
  algo que la realimentación puede explotar — resultado en cualquier
  dirección es publicable.)*
- **P4.** ¿Las mismas métricas (C, U, η, entropía del canal) son
  computables y estables en colectivos simulados de agentes con
  relevo cultural, y reproducen firmas conocidas (factor-c, pérdida
  cultural por cuello de botella demográfico — Henrich/Tasmania)?
- **P5 (horizonte).** ¿Qué clases R son necesarias para la
  persistencia de sistemas vivos/autopoiéticos en entornos no
  estacionarios — basta evolución + descentralización, o el lazo R1/R2
  paga su costo? *(Tratamiento en simulación; formulación formal como
  contribución, respuesta completa fuera de alcance.)*

## 7. Metodología

**Fase 0 — Formalización** (meses 1-6): definiciones §5.1-5.2 con
teoremas de consistencia básicos (monotonía de C, invarianza de clase
bajo isomorfismo de implementación, cotas U ≤ C, U ≤ |D|).

**Fase 1 — Calibración de la regla** (meses 6-12): MLPs (10³-10⁷
parámetros) sobre bitstrings aleatorios; curva bits/parámetro,
meseta, comparación con Gardner/MacKay y con la ley de escalamiento;
réplica del protocolo Morris a escala laptop.

**Fase 2 — El dial de Kolmogorov** (meses 12-20): datasets
algorítmicos con K controlada (salidas de programas mínimos +
fracción de ruido variable); medición de U, K̂ᶜ por compresores de
referencia (zstd/PAQ/modelo), verificación de H2. **Controles
negativos predeclarados:** centinela de fuga (el objetivo como
entrada debe saturar), etiquetas permutadas (nada debe generalizar).

**Fase 3 — Valor marginal de las clases** (meses 20-30): tripletas a
capacidad emparejada {MLP, GRU pequeña, GRU+cabeza metacognitiva}
sobre familias de tareas estáticas / temporales / no estacionarias;
ΔR1, ΔR2, η por clase; cinco semillas por celda, bootstrap
consciente de dependencia, corrección de multiplicidad, márgenes
prácticos predeclarados.

**Fase 4 — Colectivos** (meses 30-40): simulación de cadenas de
relevo cultural (agentes con capacidad individual fija, canal con
ruido, mutación/cruce de variantes) y de agregación tipo mercado;
métricas §5.4; validación contra firmas empíricas publicadas.

**Disciplina transversal:** todo experimento con contrato
predeclarado y sellado antes de resultados, ledger de unidades,
agregación recomputada de registros terminales, verificador
independiente y demostraciones de mutación — la infraestructura ya
existe y está auditada en la plataforma del autor.

**Redacción y defensa** (meses 40-48).

## 8. Contribuciones esperadas

1. El marco formal C/U/η/R con sus teoremas de consistencia (teórica).
2. La regla de comparación por clases de realimentación como estándar
   metodológico anti-confusor (metodológica).
3. Mediciones reproducibles de bits/parámetro y del punto de
   transición memorización→generalización con K controlada (empírica).
4. La primera medición de ΔR1/ΔR2 a capacidad emparejada (empírica).
5. Extensión operacional a colectivos con puente formal a la economía
   del conocimiento distribuido (interdisciplinar — núcleo del interés
   para el programa).
6. Software y protocolos abiertos, con contratos predeclarados.

## 9. Riesgos y mitigaciones

| riesgo | mitigación |
|---|---|
| "Conciencia" incendia al jurado | terminología de indicadores y clases funcionales R0/R1/R2 en TODO el texto; la palabra aparece solo citando la literatura |
| Unfolding argument | §5.3: clases arquitecturales del proceso implementado, tesis empírica sobre el lazo, no fenoménica |
| K no computable | siempre cotas: K̂ᶜ por compresores de referencia declarados; datasets aleatorios y algorítmicos donde K se CONOCE |
| ΔR ≈ 0 en todo | resultado negativo publicable de primera clase (la clase no paga su costo — relevante para diseño de IA y para P5) |
| Escala de cómputo | todo el programa corre en CPU/GPU de laboratorio; ese es un rasgo de diseño, no una limitación |
| Solapamiento con la otra línea doctoral del autor | objeto disjunto (medición fundacional vs selección de representaciones RL); comparten solo la disciplina experimental |

## 10. Referencias principales

Hernández-Orallo & Dowe (2013) *Cognitive Systems Research*;
Hernández-Orallo (2017) *The Measure of All Minds*, CUP; Chollet
(2019) arXiv:1911.01547; Legg & Hutter (2007); MacKay (2003) cap. 40;
Gardner & Derrida (1988); Friedland et al. (2018) arXiv:1708.06019 y
1810.02328; Morris et al. (2025) arXiv:2505.24832; Blier & Ollivier
(2018); Achille & Soatto (2018); *Neural Weight Norm = Kolmogorov
Complexity* (2026) arXiv:2605.10878; Butlin, Long et al. (2023)
arXiv:2308.08708 y *TiCS* (2025); Tononi & Koch (2015); Doerig et al.
(2019) — unfolding; Fleming (2021) *Know Thyself*; Hutchins (1995)
*Cognition in the Wild*; Woolley et al. (2010) *Science* 330; Riedl et
al. (2021) *PNAS* 118; Boyd & Richerson (1985); Henrich (2004, 2016);
Hayek (1945) *AER* 35(4); Friston (2013) *J. R. Soc. Interface*;
Maturana & Varela (1980).

# Cobertura de preguntas hostiles — Propuesta "Medición de la inteligencia en bits"

**Compañera de:** `PROPUESTA_DOCTORAL_MEDICION_INTELIGENCIA_EN_BITS.md`
**Formato:** por sección de la propuesta; cada pregunta con la
respuesta modelo de 30-60 segundos. Las marcadas 🔴 son las que más
duelen; ensáyelas en voz alta. Tres lentes de jurado: **[M]**
metodólogo/estadístico, **[C]** científico cognitivo/filósofo,
**[E]** economista/pragmático (perfil del programa).

---

## §1-2 Resumen y motivación

**1. [C] 🔴 "Usted no puede definir conciencia. Nadie puede. ¿Por qué
debería dejarle usar la palabra?"**
No la defino: la *sustituyo*. La tesis trabaja con clases de
realimentación R0/R1/R2 — propiedades arquitecturales verificables por
inspección del sistema. La palabra "conciencia" aparece solo al citar
la literatura que motivó la taxonomía (Butlin et al. usan exactamente
esta estrategia con sus 14 indicadores, con Chalmers y Bengio entre
los autores). Si el tribunal prefiere, puedo tachar la palabra del
documento entero sin perder una sola afirmación medible.

**2. [M] "¿'Bits' no es una metáfora? ¿Qué bit físico está contando?"**
Bits de Shannon operacionalizados por protocolos concretos: capacidad
= meseta de recuperación sobre datos incompresibles (Morris 2025);
conocimiento = reducción de longitud de código precuencial (Blier &
Ollivier). Cada número de la tesis tiene un procedimiento de medición
declarado antes del experimento. No hay metáfora: hay compresión.

**3. [E] "¿A quién le sirve esto?"**
A tres audiencias: evaluación de IA (comparaciones no confundidas —
hoy los benchmarks comparan peras con manzanas), diseño de sistemas
(¿pago el costo de la recurrencia/metacognición o no?), y economía
del conocimiento (medir cuánta información conserva y agrega una
institución — §5.4). El resultado negativo también sirve: si ΔR2 ≈ 0,
media agenda de "IA metacognitiva" pierde su justificación de
desempeño.

**4. [C] "¿Esto no es solo teoría de la información con otro nombre?"**
La teoría de la información aporta la moneda; la contribución es el
SISTEMA de medición: qué comparar con qué (clases de equivalencia),
cómo separar memorización de generalización con K controlada, y la
validación por ablación a capacidad emparejada. Shannon no dice nada
de eso, como el metro no dice cómo hacer un ensayo clínico.

## §3 Estado del arte

**5. [C] 🔴 "Hernández-Orallo lleva 15 años en esto. ¿Qué añade usted
que *The Measure of All Minds* no tenga?"**
Tres cosas concretas: (i) la psicometría universal no fija moneda —
mide por dificultad de tareas; yo fijo bits con protocolos de
capacidad; (ii) no tiene la regla de clases de comparación por
realimentación — su marco compara cualquier par de sistemas; (iii) no
ejecuta ablaciones a capacidad emparejada. Es el programa de esa
escuela llevado del territorio al sistema métrico — y lo citaré como
fundamento, no como rival.

**6. [M] "Morris et al. midieron 3.6 bits/parámetro en GPTs. ¿Su fase
1 no es una réplica menor?"**
La fase 1 es deliberadamente una réplica — así se calibra una regla
antes de medir con ella. Lo nuevo empieza en fase 2 (el dial de K,
que Morris no varía: sus datos son o aleatorios o texto) y fase 3
(clases a capacidad emparejada, que nadie ha hecho).

**7. [C] "Cita IIT y Φ. IIT fue acusada de pseudociencia por más de
100 investigadores en 2023. ¿Se apoya en ella?"**
No me apoyo: la cito como el ejemplo más conocido de medida graduada
y tomo UNA observación suya que es matemática, no metafísica: el lazo
recurrente es una propiedad estructural binaria verificable. Mi marco
sobrevive aunque IIT caiga entera, porque no uso Φ en ningún
experimento.

**8. [E] "¿Hayek en una tesis de redes neuronales? ¿No es
oportunismo hacia este programa?"**
Es la conexión sustantiva más antigua del campo: Hayek 1945 describe
el mercado como agregador de información dispersa — un sistema
cognitivo colectivo — y *The Sensory Order* (1952) es literalmente
teoría conexionista. La fase 4 formaliza esa intuición con canales de
Shannon y la somete a simulación. Si algo, es la deuda del
conexionismo con Hayek, no al revés.

## §5 Marco teórico

**9. [C] 🔴 "El *unfolding argument*: toda red recurrente tiene una
gemela feedforward con la misma función entrada-salida. Sus clases R
colapsan."**
El unfolding refuta teorías que atribuyen *experiencia* a la
estructura. Yo no atribuyo experiencia: atribuyo CONSECUENCIAS DE
PROCESO — capacidad por parámetro, costo de cómputo, eficiencia de
adquisición en línea. La gemela desenrollada tiene la misma función
pero OTRO costo (parámetros explosivos, sin aprendizaje en línea).
Ese contraste es exactamente mi experimento de fase 3: si el lazo no
paga, mi H3 se falsea y lo publico. Una objeción que se convierte en
diseño experimental no es una amenaza; es un regalo.

**10. [M] "Sus clases R son discretas. ¿La realimentación no es un
continuo?"**
La membresía es discreta por inspección (¿existe el lazo físico o
no?), como "tiene memoria" o "es diferenciable". El GRADO de uso del
lazo es continuo y lo capturan las métricas (η, calibración
metacognitiva). Clase discreta + intensidad continua es la estructura
estándar de cualquier taxonomía útil.

**11. [C] "R2 exige 'representación del propio estado'. ¿Cómo
distingue eso de una entrada más?"**
Operacionalmente: la señal proviene causalmente de los
parámetros/activaciones del propio sistema en el mismo episodio, y su
ablación (cortar el lazo, conservar la capacidad) degrada la
calibración de confianza. Test causal, no semántico — el mismo
estándar de Fleming en metacognición humana.

**12. [M] "¿'Capacidad emparejada' por cuenta de parámetros o por
bits? No es lo mismo."**
Por bits medidos con el protocolo de fase 1, no por parámetros — ese
es precisamente el punto de calibrar la regla primero. Si dos
arquitecturas difieren en bits/parámetro, empatarlas por parámetros
sería el confusor que denuncio.

**13. [C] "K de Kolmogorov es incomputable. Su 'dial de K' es una
ficción."**
K exacta es incomputable; COTAS SUPERIORES no: todo compresor da una.
Y el dial va en la otra dirección — CONSTRUYO datasets desde
programas mínimos, donde la cota superior es la longitud del programa
generador más el ruido inyectado, conocidos por construcción. Uso K
computablemente acotada, jamás K platónica; los compresores de
referencia quedan predeclarados.

## §6 Hipótesis

**14. [M] 🔴 "H3 es un 'si hay efecto, hay efecto'. ¿Dónde está la
falsabilidad?"**
H3 afirma una INTERACCIÓN específica: ΔR1 > 0 en tareas temporales Y
ΔR1 ≈ 0 en estáticas, a capacidad emparejada, con margen práctico
predeclarado. Se falsea de tres maneras: ventaja en estáticas
(sobra la explicación del lazo), cero en temporales (el lazo no
paga), o ventaja no anulada al emparejar bits (era capacidad, no
clase). Cada rama tiene consecuencia teórica distinta.

**15. [C] "P5 — vida y conciencia — es filosofía, no ciencia."**
Por eso el alcance declarado es: formalización de la pregunta +
resultados en simulación de sistemas autopoiéticos mínimos. Entregar
la PREGUNTA bien formulada con un marco de medición es una
contribución (como la formulación de Hilbert); no prometo el teorema.

**16. [E] "¿Y si todas sus hipótesis salen negativas?"**
Entonces habré medido, con controles adversariales, que la
recurrencia y la metacognición no pagan en desempeño a capacidad
igual — un resultado que reordena prioridades de investigación e
inversión en IA. Mi historial demuestra que sé publicar negativos
con autoridad: mi plataforma cerró una cadena completa de hipótesis
de representación con veredictos negativos auditados y replicación
fuera de tiempo.

## §7 Metodología

**17. [M] "Cinco semillas no son muestra estadística."**
Correcto, y la propuesta lo dice explícitamente: las semillas
describen variabilidad de optimización; la unidad estadística son
bloques de tareas/datos con bootstrap consciente de dependencia y
corrección de multiplicidad — disciplina que ya opero a diario con
contratos sellados.

**18. [M] 🔴 "¿Quién audita sus predeclaraciones? Autodisciplina no
es garantía."**
Los contratos se comprometen en repositorio público con hash ANTES de
los resultados (verificable por terceros por el orden del grafo git),
los agregados se recomputan desde registros terminales por un
verificador independiente, y cada test de refusal demuestra que
muerde por mutación dirigida. Es más auditoría de la que exige
cualquier revista del área — y toda la maquinaria ya existe y fue
auditada externamente en mi plataforma.

**19. [C] "Su fase 4 simula colectivos. ¿Qué valida que la
simulación diga algo del mundo?"**
Validación contra firmas empíricas publicadas e independientes: el
factor-c de Woolley (estructura factorial), y la pérdida cultural por
cuello de botella demográfico (el caso Tasmania de Henrich, con su
modelo matemático). Si el marco no reproduce lo ya sabido, se rechaza
antes de afirmar nada nuevo.

**20. [E] "¿Cuarenta y ocho meses? ¿Con qué financiación de cómputo?"**
El programa entero corre en hardware de laboratorio — esa es una
decisión de diseño: los objetos de fase 1-3 son redes cuya capacidad
SE CONOCE, no modelos frontera. El costo dominante es tiempo de
investigador, no GPU.

**21. [M] "¿Por qué MLPs y GRUs pequeñas y no transformers?"**
Porque la teoría de capacidad clásica (Gardner/MacKay) da valores
EXACTOS para esas familias — son el sistema donde la regla se calibra
contra verdad conocida. Los transformers entran después vía la ley de
escalamiento y el valor Morris, como extrapolación, no como base.

## §8-9 Contribuciones y riesgos

**22. [C] "¿Esto no son dos tesis — una de IA y una de cognición
colectiva?"**
Es UNA tesis: el mismo cuarteto de métricas (C, U, η, clase R)
aplicado a dos instancias del mismo objeto formal. La fase 4 usa las
definiciones de la fase 0 sin cambiar una letra — esa uniformidad es
exactamente la contribución que ninguna de las dos literaturas tiene.

**23. [E] 🔴 "Usted ya tiene otra propuesta doctoral en curso
(selección multi-fidelidad para RL). ¿Está postulando lo mismo dos
veces? ¿Le alcanza la vida para ambas?"**
Son objetos disjuntos: aquella optimiza la selección de
representaciones para agentes de RL bajo presupuesto; esta mide
fundamentos de capacidad y clases cognitivas. Comparten solo la
disciplina experimental — que es transferible, no duplicada.
Postular a dos programas es práctica estándar y transparente;
matricularé y ejecutaré UNO. Si me preguntan directamente, esa es la
respuesta completa y sin evasivas.

**24. [M] "Si la palabra conciencia es tan tóxica, ¿por qué está en
la motivación?"**
Porque la literatura que motiva las clases la usa, y fingir que no
existe sería deshonesto con las fuentes. La estrategia es la de
Butlin et al.: motivación con la palabra citada, ciencia con
indicadores operacionales. Motivar ≠ afirmar.

**25. [C] "¿Su marco implica que un termostato con lazo es 'más
consciente' que GPT-4 feedforward?"**
Implica que el termostato es R1 y el transformer de inferencia pura
R0 — como clasificación DE PROCESO, sí, y no me sonroja: también un
gusano tiene lazos que una enciclopedia no tiene. El marco no ordena
por 'nivel de conciencia'; ordena comparaciones legítimas. La
enciclopedia gana en C; el gusano en η ante cambio. Justamente por
eso no se comparan a secas.

**26. [M] "¿Cómo mide U en un colectivo humano real sin
instrumentarlo todo?"**
En la tesis, no lo hago: fase 4 es simulación validada contra firmas
empíricas. La medición de campo queda declarada como trabajo futuro
con diseño esbozado (repertorios culturales documentados, cadenas de
transmisión experimentales tipo Mesoudi). Prometo lo que puedo
entregar.

**27. [E] "Deme la frase de una línea para el comité de admisión."**
"Propongo el sistema métrico — en bits, con clases de comparación y
controles adversariales — que la ciencia de la inteligencia
individual y colectiva definió como programa pero nunca construyó, y
lo entrego calibrado contra redes cuya capacidad conocemos con
exactitud."

## Preguntas de sorpresa (fuera de secciones)

**28. [C] "¿Qué pasa con la conciencia fenoménica — los qualia? ¿Su
marco dice algo?"**
Nada, por diseño, y lo declaro en la primera página. Es una tesis
sobre magnitudes medibles de proceso. El problema difícil sigue
difícil; yo no lo toco ni lo necesito.

**29. [M] "¿Cuál es su variable dependiente EXACTA en fase 3?"**
Pérdida propia media por observación (log-loss/MDL) en el bloque de
evaluación, por celda {clase, familia de tarea, capacidad}, con ΔRk =
diferencia pareada entre clases a capacidad emparejada, IC por
bootstrap de bloques y margen práctico predeclarado antes de
ejecutar.

**30. [E] "¿Riesgo de que alguien grande (DeepMind/Anthropic) lo haga
antes?"**
Morris demuestra que las piezas interesan a los grandes; ninguno ha
publicado la regla de clases ni la ablación emparejada — y la ventana
es ahora. Si publican una pieza durante la tesis, me convierto en la
réplica independiente + extensión colectiva, que es donde no
competirán: no es su negocio.

**31. [C] "Los memes de Dawkins están académicamente muertos. ¿Por
qué revivirlos?"**
No uso memética doctrinaria: uso evolución cultural cuantitativa
(Boyd-Richerson, Henrich, Mesoudi), que está viva, publica en
*Nature*/*PNAS* y tiene modelos formales. "Meme" no aparece en la
propuesta; "variante cultural en canal ruidoso", sí.

**32. [M] "Si la ocupación U/C depende del optimizador, ¿mide el
sistema o mide a Adam?"**
Mide el sistema BAJO un protocolo declarado de entrenamiento —
igual que la capacidad de canal se mide bajo un código. El
optimizador queda fijo en el contrato; la sensibilidad al optimizador
es un análisis secundario explícito, no un confusor silencioso.

**33. [E] "¿Qué se lleva este programa doctoral que no se lleve
usted?"**
El marco completo con software abierto, los protocolos
predeclarados reutilizables para evaluación de IA, la línea de
cognición colectiva/conocimiento distribuido plantada en el programa
— y un doctorando que ya opera con estándares de auditoría
adversarial que puede enseñar a otros.

# Perfiles de memorización y ganancia predictiva para dimensionar redes neuronales

## Estudio en bits con tareas sintéticas de estructura conocida

**Propuesta de investigación doctoral**  
**Harvey Demian Bastidas Caicedo**  
**Área:** inteligencia artificial y ciencias de la computación  
**Versión:** 2026-09-04

## Resumen

El número de parámetros de una red neuronal indica su tamaño, pero no permite saber por sí solo cuánto puede memorizar, cuánta estructura reutilizable aprendió ni cuál es el menor tamaño suficiente para una tarea. Estas diferencias importan tanto para comprender la generalización como para evitar modelos innecesariamente grandes. Existen resultados clásicos sobre la capacidad de una neurona de umbral [1]–[3], reglas de capacidad para redes de perceptrones [4], [5] y mediciones recientes de memorización en transformadores [8]. Sin embargo, esas medidas no deben confundirse con la información que un modelo puede aprovechar para predecir datos no vistos.

Esta investigación propone un perfil en bits que mantenga separadas tres magnitudes: la capacidad empírica de memorizar ejemplos aleatorios, la memorización específica del conjunto de entrenamiento y la ganancia predictiva fuera de muestra. El estudio usará tareas sintéticas cuya regla generadora y nivel de ruido se conocen. Así será posible medir, además, qué parte del comportamiento de la regla fue recuperada y observar cuál es la red más pequeña que alcanza un criterio de generalización fijado con anticipación.

La pregunta principal es si ese perfil permite anticipar el tamaño suficiente de una red con mayor precisión que la cuenta de parámetros y las reglas de capacidad existentes. El estudio principal empleará perceptrones multicapa en familias de reglas booleanas canónicas; una familia de autómatas finitos servirá como confirmación secuencial con redes recurrentes pequeñas. Las tareas, no las semillas de entrenamiento, serán las unidades independientes. El diseño separará piloto, ajuste, calibración y prueba final, e incluirá comparadores que puedan derrotar la propuesta.

El resultado esperado es un protocolo reproducible para estudiar memorización, aprendizaje de estructura y tamaño de modelos con cantidades expresadas en bits y denominadores explícitos, sin afirmar que todas representan el mismo tipo de información. Si el perfil no mejora el dimensionamiento, la investigación establecerá de manera precisa dónde dejan de ser útiles estas medidas.

## 1. Problema de investigación

David MacKay estudió el aprendizaje de una neurona de umbral como un canal de comunicación. Bajo entradas en posición general y etiquetas binarias aleatorias, una neurona con \(K\) pesos presenta una transición de capacidad cercana a \(2K\) asociaciones binarias [1]. Los resultados de Cover y Gardner explican esa transición desde la geometría de las dicotomías linealmente separables [2], [3]. Esta afirmación es importante, pero tiene un alcance concreto: no dice que toda red profunda almacene dos bits útiles por parámetro ni que sus pesos estén ocupados en esa proporción durante el aprendizaje de una tarea real.

En redes con varias capas, el panorama es más complejo. Friedland y Krell derivaron reglas de capacidad que escalan con el número de pesos en redes de perceptrones [4] y propusieron métodos prácticos de dimensionamiento [5]. A la vez, redes sobredimensionadas pueden ajustar etiquetas aleatorias [6], aunque durante el entrenamiento suelen aprender primero regularidades compartidas y después ejemplos ruidosos o excepcionales [7]. Más recientemente, Morris y colaboradores separaron la memorización específica de un conjunto de datos de la información sobre su proceso generador. Al entrenar transformadores sobre secuencias uniformes, donde no existe una regla que generalizar, estimaron una capacidad empírica cercana a 3,6 bits por parámetro [8]. Los autores aclaran que la capacidad alcanzada mediante descenso de gradiente es una cota inferior de lo que podría lograr un procedimiento de entrenamiento mejor.

Estos resultados dejan una dificultad práctica y conceptual. Dos redes con el mismo número de parámetros pueden usar su capacidad de manera distinta: una puede recordar ejemplos particulares y otra extraer una regla que funciona en casos nuevos. Incluso si ambas reducciones de pérdida se expresan en bits, no representan el mismo objeto. Los bits ahorrados al predecir un conjunto grande de prueba pueden crecer con el número de ejemplos sin que el modelo necesite almacenar cada respuesta. Por ello, la información predictiva no puede interpretarse sin más como una fracción del espacio de memoria de los pesos.

El problema de esta tesis es determinar si un **perfil que mantenga separadas esas cantidades** puede servir para una decisión concreta: estimar el menor tamaño de red que generaliza en una tarea nueva de una familia conocida. Esto exige fuentes controladas. En datos naturales no conocemos la regla verdadera ni la complejidad mínima del problema; en tareas sintéticas sí podemos fijar la regla, el ruido, la cantidad de ejemplos y las intervenciones que distinguen una regla de otra.

### Pregunta principal

> ¿Puede un perfil en bits que separa memorización específica de la muestra y ganancia predictiva fuera de muestra anticipar, en tareas sintéticas de estructura conocida, el menor tamaño de una red que generaliza, mejor que la cuenta de parámetros y las reglas de capacidad existentes?

## 2. Objetivos

### Objetivo general

Diseñar y evaluar un protocolo reproducible que relacione capacidad de memorización, memorización específica y ganancia predictiva con el tamaño suficiente observado de redes neuronales.

### Objetivos específicos

1. **Calibrar la capacidad empírica de memorización** de perceptrones multicapa y redes recurrentes pequeñas sobre datos aleatorios de entropía conocida, bajo protocolos de entrenamiento y precisión numérica declarados.
2. **Caracterizar la dinámica del aprendizaje** en tareas con regla y ruido conocidos, midiendo por separado la memorización específica del entrenamiento, la ganancia predictiva fuera de muestra y la recuperación de la regla generadora.
3. **Evaluar un predictor de tamaño suficiente** basado en ese perfil y compararlo, sobre tareas intactas, con la cuenta de parámetros, reglas de capacidad publicadas y curvas de aprendizaje de bajo costo.

## 3. Marco teórico y estado del arte

### 3.1 Capacidad de memorización

La capacidad clásica de una neurona de umbral depende de la familia de funciones, la posición de las entradas y el criterio de aprendizaje [1]–[3]. No es una constante universal de las redes neuronales. Las extensiones de Friedland y colaboradores muestran que ciertos puntos de capacidad de redes de perceptrones escalan linealmente con sus pesos, pero también distinguen entre cotas geométricas y lo que un algoritmo logra en la práctica [4], [5].

La investigación reciente de Morris et al. ofrece un protocolo empírico compatible con una interpretación en bits [8]. En datos uniformes, el modelo no puede inferir una regularidad que prediga ejemplos nuevos; la mejora respecto de la entropía de la fuente se atribuye a memorización. El máximo observado al variar el tamaño del conjunto sirve como capacidad empírica bajo una arquitectura y un procedimiento de entrenamiento. Esta tesis adaptará el protocolo a modelos pequeños. La calibración no se presentará como una nueva ley universal, sino como el denominador experimental necesario para estudiar los otros componentes.

### 3.2 Memorización y generalización

El ajuste del conjunto de entrenamiento no distingue entre recordar una excepción y aprender una regla. Zhang et al. mostraron que redes profundas pueden ajustar etiquetas aleatorias [6]. Arpit et al. observaron que, en datos con ruido, las regularidades simples suelen aprenderse antes que los ejemplos sin estructura [7]. Morris et al. formalizaron la separación entre memorización no intencional y conocimiento del proceso generador [8]. Estas contribuciones impiden tratar la pérdida de entrenamiento, la pérdida de prueba y la información en los pesos como si fueran una sola magnitud.

En esta propuesta, la **memorización específica** representa información sobre realizaciones particulares del conjunto de entrenamiento que no se explica por la regla generadora conocida. Para las tareas binarias, el generador proporciona la probabilidad verdadera \(p_\star(y_i\mid x_i)\), incluido el ruido. Se usará el estimador:

\[
\widehat M_{spec}(t)=\sum_{i\in S}
\left[\log_2 p_{\theta_t}(y_i\mid x_i)
-\log_2 p_\star(y_i\mid x_i)\right]_+,
\]

donde \(S\) es el conjunto de entrenamiento y \([a]_+=\max(a,0)\). Asignar al resultado observado más probabilidad que el proceso verdadero solo puede lograrse aprovechando la realización particular de la muestra; por eso el exceso se interpreta como memorización específica bajo este generador. Se reportarán tanto los bits totales como los bits por ejemplo. Esta cantidad puede aumentar durante el entrenamiento aun cuando el desempeño fuera de muestra deje de mejorar.

### 3.3 Ganancia predictiva fuera de muestra

La información mutua de Shannon no incorpora las restricciones del modelo que intenta aprovechar una señal. La información predictiva \(\mathcal{V}\), propuesta por Xu et al., mide cuánto reduce la pérdida logarítmica una familia de predictores al observar una entrada, comparada con la mejor predicción de la misma familia sin esa entrada [9]. Este antecedente motiva una medida más directa para cada punto del entrenamiento:

\[
\widehat G_{pred}(t)=\frac{1}{|T|}\sum_{(x,y)\in T}
\left[\log_2 q_0(y)-\log_2 p_{\theta_t}(y\mid x)\right],
\]

donde \(T\) es el conjunto intacto, \(q_0\) es un predictor nulo que no observa \(x\), y \(p_{\theta_t}\) es el modelo en el paso \(t\). El predictor nulo se ajusta sin usar \(T\). El resultado expresa cuántos bits por ejemplo ahorra el modelo frente a no observar la entrada.

\(\widehat G_{pred}\) es una ganancia de predicción bajo pérdida logarítmica, no una medida literal de bits almacenados en los pesos. Por eso no se dividirá por la capacidad de memorización para producir una supuesta ocupación. La longitud de descripción y la compresibilidad de modelos siguen siendo comparadores pertinentes [10], [11], pero tampoco sustituyen una medición fuera de muestra.

### 3.4 Identificación de reglas y límites de información

Las tareas sintéticas se construirán a partir de una variable latente \(Z\) que identifica la regla generadora dentro de un catálogo finito de \(M\) reglas. Sea \(B(W)\) el comportamiento del modelo entrenado sobre un conjunto de sondas. Si un observador puede identificar \(Z\) desde \(B(W)\) con probabilidad de error \(P_e\), la desigualdad de Fano y el procesamiento de datos establecen [12]:

\[
I(Z;W)\geq I(Z;B(W))
\geq \log_2 M-h_2(P_e)-P_e\log_2(M-1),
\]

donde \(W\) representa el estado entrenado y \(h_2\) es la entropía binaria. Este resultado relaciona el reconocimiento de la regla con información mínima, pero no entrega por sí solo el número exacto de parámetros. Traducir bits a parámetros requiere supuestos adicionales sobre precisión y familia de modelos. La tesis separará esa referencia de la estimación empírica de tamaño.

### 3.5 Predicción de desempeño y búsqueda de arquitecturas

Anticipar el resultado de un entrenamiento costoso tampoco es una idea nueva. Domhan et al. y Klein et al. extrapolan curvas parciales para detener configuraciones poco prometedoras [14], [15]. Los métodos de búsqueda de arquitecturas sin entrenamiento usan propiedades de inicialización y gradientes como proxies de desempeño [16], y NAS-Bench-Suite-Zero muestra que varios de esos proxies contienen información complementaria, pero también sesgos dependientes del espacio de búsqueda [17].

Por tanto, el vacío no puede formularse como “predecir el desempeño antes de entrenar”. La pregunta más estrecha es si separar memorización específica, ganancia predictiva y recuperación de la regla aporta información adicional para estimar **tamaño suficiente**, con el mismo presupuesto de observación y en tareas no usadas para ajustar el predictor. La tesis deberá superar extrapoladores y proxies compatibles; de lo contrario, el perfil puede conservar valor descriptivo, pero no justifica una nueva regla de dimensionamiento.

## 4. Definiciones operativas

Para cada familia de arquitectura \(F\), algoritmo de entrenamiento \(A\) y precisión numérica \(b\), se usarán las siguientes cantidades:

| Símbolo | Definición operativa | Interpretación permitida |
|---|---|---|
| \(\widehat C_{mem}(F,A,b)\) | Máximo observado de memorización sobre conjuntos uniformes al variar su tamaño. | Capacidad empírica bajo el protocolo; no capacidad universal. |
| \(\widehat M_{spec}(t)\) | Información sobre ejemplos de entrenamiento no explicada por el generador conocido, en el paso \(t\). | Memorización específica de la muestra. |
| \(\widehat G_{pred}(t)\) | Reducción de pérdida logarítmica del modelo frente al predictor nulo sobre ejemplos intactos, en bits por ejemplo. | Ganancia predictiva del punto de entrenamiento; no memoria en los pesos. |
| \(R(t)\) | Acuerdo del modelo con la regla verdadera en entradas contrafactuales que no se usaron para entrenar. | Recuperación conductual de la regla plantada. |
| \(N_{min}^{obs}\) | Menor número de parámetros de la grilla cuya cota inferior de desempeño supera el umbral y cumple calibración. | Tamaño suficiente observado dentro de la grilla. |

El cociente entre \(\widehat M_{spec}\) y \(\widehat C_{mem}\) podrá reportarse como análisis secundario de carga relativa de memorización. No se llamará ocupación útil, no mezclará información de prueba con memoria y no se supondrá limitado a \([0,1]\), porque el denominador es una estimación empírica.

## 5. Hipótesis

**H1 — Calibración de capacidad.** Dentro de cada familia, precisión y protocolo de entrenamiento, \(\widehat C_{mem}\) crecerá aproximadamente de forma lineal con el número de parámetros en el intervalo estudiado, pero su pendiente dependerá de la arquitectura y no se fijará de antemano en 2 ni en 3,6 bits por parámetro. La hipótesis falla si un modelo lineal deja residuos sistemáticos o si no aparece una meseta reproducible.

**H2 — Separación durante el aprendizaje.** En tareas con ruido idiosincrático, existirá una región de entrenamiento en la que \(\widehat M_{spec}\) aumente sin una mejora correspondiente de \(\widehat G_{pred}\) ni de \(R\) en datos intactos. La hipótesis falla si las tres trayectorias son indistinguibles dentro de la precisión predeclarada o si la memorización adicional mejora de manera estable la regla recuperada.

**H3 — Predicción de tamaño.** En tareas nuevas de la familia primaria, un predictor sencillo que use el perfil \((\widehat C_{mem},\widehat M_{spec},\widehat G_{pred},R)\), medido con un sondeo de costo fijo, estimará \(N_{min}^{obs}\) con menor error absoluto en escala logarítmica que los comparadores predeclarados, sin aumentar la tasa de subdimensionamiento. La hipótesis falla si no mejora el error o si su aparente ahorro se obtiene recomendando redes insuficientes.

## 6. Metodología

### 6.1 Familias de tareas

La familia principal estará formada por funciones representadas mediante diagramas de decisión binarios reducidos y ordenados. Con un orden de variables fijo, esta representación es canónica: dos diagramas representan la misma función si y solo si son idénticos. El número de nodos se usará como complejidad estructural declarada, sin presentarlo como complejidad de Kolmogorov. Cada tarea elegirá una regla latente, una distribución de entradas, un nivel de ruido de etiquetas y un número de ejemplos. Para dimensiones pequeñas, la tabla de verdad completa servirá como evaluación contrafactual; para dimensiones mayores se usará un conjunto separador generado antes de entrenar.

La confirmación secuencial usará lenguajes regulares generados por autómatas finitos mínimos. El número de estados del autómata ofrece una medida estructural verificable y permite comprobar si el protocolo se transporta de MLP a una red recurrente pequeña [13]. Esta fase no afirmará que ambas arquitecturas tienen la misma capacidad por compartir una cuenta de parámetros.

### 6.2 Partición y unidad de análisis

Las tareas generadas se dividirán por identidad de regla en cuatro grupos:

1. **piloto**, para depurar rangos y asegurar que las medidas son computables;
2. **ajuste**, para estimar el predictor de tamaño;
3. **calibración**, para fijar umbrales e intervalos;
4. **prueba intacta**, abierta una sola vez para las hipótesis confirmatorias.

La unidad independiente será una tarea generada. Los tamaños y métodos serán tratamientos pareados dentro de ella. Las semillas de inicialización serán repeticiones anidadas y se reportará su dispersión sin contarlas como tareas adicionales. El piloto determinará mediante análisis de potencia el número definitivo de tareas; se fijará un mínimo de 30 tareas intactas por estrato confirmatorio cuando la varianza observada lo permita.

### 6.3 Modelos y entrenamiento

El estudio principal recorrerá una grilla logarítmica de MLP desde cientos hasta aproximadamente un millón de parámetros. La fase secuencial usará GRU pequeñas en un rango comparable de costo de entrenamiento. Optimizador, precisión, presupuesto de pasos, inicialización y regla de parada formarán parte del protocolo. La sensibilidad a esas decisiones será secundaria y acotada.

Antes de recorrer la grilla completa de una tarea, todos los métodos recibirán el mismo **presupuesto de sondeo**: dos tamaños pequeños fijados en el piloto y un número limitado de actualizaciones. El perfil para H3 solo podrá usar los checkpoints de ese sondeo y la calibración de capacidad obtenida con anterioridad. Los comparadores recibirán las mismas curvas parciales y el mismo costo. Ningún método podrá consultar resultados de modelos cercanos a \(N_{min}^{obs}\) antes de emitir su estimación.

El punto de parada se elegirá únicamente con la partición de calibración. Las medidas que deciden H2 y H3 se calcularán sobre tareas y ejemplos de prueba no utilizados para seleccionar ese punto. Esto evita que una reducción de pérdida explique otra reducción de pérdida elegida con los mismos datos.

### 6.4 Comparadores

El predictor propuesto deberá superar comparadores útiles, no controles débiles:

- cuenta de parámetros y dimensión de entrada;
- reglas de capacidad de MacKay/Friedland cuando sus supuestos sean aplicables;
- extrapolación probabilística de la curva parcial [14], [15];
- proxies sin entrenamiento compatibles con las arquitecturas estudiadas [16], [17];
- predictor monotónico que use solo tamaño, cantidad de datos y ruido.

No se escogerá el comparador ganador después de ver la prueba final.

### 6.5 Variables y análisis

La variable primaria de H3 será el error absoluto de \(\log_2 N_{min}^{obs}\). La tasa de subdimensionamiento será una condición separada: una recomendación cuenta como insuficiente si no alcanza el umbral fuera de muestra. H1 se evaluará con modelos de escalamiento y diagnóstico de residuos. H2 usará diferencias pareadas entre ventanas de entrenamiento. H3 empleará intervalos por bootstrap de tareas y corrección de Holm para las comparaciones confirmatorias.

Los umbrales de desempeño, calibración y reconocimiento se fijarán con el piloto y se publicarán antes de abrir la prueba intacta. Los resultados negativos, las tareas excluidas y los fallos de entrenamiento permanecerán en el denominador correspondiente.

## 7. Aportes esperados

1. Una definición operativa que evita confundir bits memorizados con ganancia predictiva en bits.
2. Una calibración reproducible de capacidad empírica para modelos pequeños y precisiones declaradas.
3. Un banco de tareas con reglas conocidas, ruido controlado y pruebas contrafactuales de reconocimiento.
4. Una evaluación fuera de muestra de si el perfil informativo mejora el dimensionamiento de redes.
5. Un resultado teórico de identificabilidad y una caracterización explícita de los casos en que no puede traducirse a tamaño de modelo.

En el alcance de esta tesis, la utilidad práctica se limita a familias donde puede obtenerse una corrida piloto y existe una estructura comparable a la usada para ajustar el predictor. Allí podría reducir búsquedas al descartar tamaños claramente insuficientes o excesivos. La utilidad científica es más general: distinguir qué parte de una decisión de tamaño proviene de memorización, generalización o estructura de la tarea.

## 8. Factibilidad, riesgos y ética

Los datos serán sintéticos y no contendrán información personal. El experimento no requiere participantes humanos ni decisiones sobre personas. El código, las reglas generadoras, los registros de ejecución y los análisis se publicarán con versiones y semillas reproducibles.

El principal riesgo científico es que el perfil no prediga \(N_{min}^{obs}\) mejor que una curva de aprendizaje sencilla. Ese resultado refutaría H3 y delimitaría el valor práctico de las medidas en bits. Un segundo riesgo es que la capacidad no sea estable frente al optimizador; en ese caso se reportará como propiedad del sistema de entrenamiento y no de la arquitectura aislada. Un tercero es que la confirmación secuencial no transporte el resultado de la familia booleana; esa diferencia será evidencia sobre el alcance del protocolo, no motivo para unir ambas poblaciones.

El cómputo es viable porque los modelos son pequeños y las tareas se generan localmente. Un preflight medirá tiempo y memoria antes de congelar la grilla. Cada fase tendrá un presupuesto y una condición de detención; no se intentará reproducir la escala de miles de millones de parámetros de [8].

## 9. Plan de trabajo

| Periodo | Actividades y entregables |
|---|---|
| Meses 1–6 | Revisión sistemática, formalización, generadores, piloto y preregistro. |
| Meses 7–12 | Calibración de capacidad, validación de estimadores y primer artículo metodológico. |
| Meses 13–20 | Trayectorias de memorización y ganancia predictiva en la familia booleana. |
| Meses 21–27 | Predictor de tamaño, comparadores y evaluación intacta. |
| Meses 28–32 | Confirmación secuencial con autómatas y análisis de alcance. |
| Meses 33–36 | Integración de resultados, publicación de artefactos y escritura de tesis. |

El alcance mínimo defendible comprende la familia booleana, H1–H3 y el resultado de identificabilidad. La confirmación con autómatas se recortará si el estudio principal requiere más tareas o mayor precisión estadística.

## Referencias

[1] D. J. C. MacKay, *Information Theory, Inference, and Learning Algorithms*. Cambridge, Reino Unido: Cambridge University Press, 2003, cap. 40.

[2] T. M. Cover, “Geometrical and statistical properties of systems of linear inequalities with applications in pattern recognition,” *IEEE Transactions on Electronic Computers*, vol. EC-14, no. 3, pp. 326–334, 1965.

[3] E. Gardner, “The space of interactions in neural network models,” *Journal of Physics A: Mathematical and General*, vol. 21, no. 1, pp. 257–270, 1988.

[4] G. Friedland and M. Krell, “A capacity scaling law for artificial neural networks,” arXiv:1708.06019, 2018.

[5] G. Friedland, A. Metere, and M. Krell, “A practical approach to sizing neural networks,” arXiv:1810.02328, 2018.

[6] C. Zhang, S. Bengio, M. Hardt, B. Recht, and O. Vinyals, “Understanding deep learning requires rethinking generalization,” in *Proc. ICLR*, 2017.

[7] D. Arpit et al., “A closer look at memorization in deep networks,” in *Proc. ICML*, pp. 233–242, 2017.

[8] J. X. Morris et al., “How much do language models memorize?” arXiv:2505.24832, 2025.

[9] Y. Xu, S. Zhao, J. Song, R. Stewart, and S. Ermon, “A theory of usable information under computational constraints,” in *Proc. ICLR*, 2020.

[10] L. Blier and Y. Ollivier, “The description length of deep learning models,” in *Advances in Neural Information Processing Systems*, vol. 31, 2018.

[11] J. Bernstein and Y. Yue, “Computing the information content of trained neural networks,” arXiv:2103.01045, 2021.

[12] T. M. Cover and J. A. Thomas, *Elements of Information Theory*, 2nd ed. Hoboken, NJ, EE. UU.: Wiley, 2006.

[13] J. J. Michalenko, A. Shah, A. Verma, R. G. Baraniuk, S. Chaudhuri, and A. B. Patel, “Representing formal languages: A comparison between finite automata and recurrent neural networks,” arXiv:1902.10297, 2019.

[14] T. Domhan, J. T. Springenberg, and F. Hutter, “Speeding up automatic hyperparameter optimization of deep neural networks by extrapolation of learning curves,” in *Proc. IJCAI*, pp. 3460–3468, 2015.

[15] A. Klein, S. Falkner, J. T. Springenberg, and F. Hutter, “Learning curve prediction with Bayesian neural networks,” in *Proc. ICLR*, 2017.

[16] J. Mellor, J. Turner, A. Storkey, and E. J. Crowley, “Neural architecture search without training,” in *Proc. ICML*, pp. 7588–7598, 2021.

[17] A. Krishnakumar, C. White, A. Zela, R. Tu, M. Safari, and F. Hutter, “NAS-Bench-Suite-Zero: Accelerating research on zero cost proxies,” in *Advances in Neural Information Processing Systems*, vol. 35, 2022.

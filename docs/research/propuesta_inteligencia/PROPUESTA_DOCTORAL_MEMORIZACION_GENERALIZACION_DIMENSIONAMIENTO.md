# Memorización, generalización y dimensionamiento de redes neuronales

## Evaluación experimental con procesos generadores conocidos

**Propuesta de investigación doctoral**<br>
**Harvey Demian Bastidas Caicedo**<br>
**Área:** inteligencia artificial y ciencias de la computación<br>
**Versión:** 2026-09-04

## Resumen

Elegir el tamaño de una red neuronal sigue siendo una decisión costosa. Una red demasiado pequeña puede no aprender la estructura necesaria; una innecesariamente grande aumenta el costo de entrenamiento y dificulta distinguir generalización de memorización. La cuenta de parámetros describe el tamaño del modelo, pero no determina por sí sola cuántos ejemplos puede memorizar, cuánto aprende de una regla reutilizable ni cuál es el menor tamaño que alcanza un nivel de desempeño solicitado.

Esta investigación estudiará si combinar una calibración previa de capacidad con dos mediciones del entrenamiento inicial ayuda a tomar esa decisión. Las tres cantidades serán la capacidad empírica de memorizar asociaciones aleatorias, el exceso de ajuste específico del conjunto de entrenamiento y la mejora predictiva fuera de muestra. Cuando los modelos produzcan probabilidades, las dos últimas se expresarán como diferencias de pérdida logarítmica en base dos. Esta unidad común facilita la comparación numérica, pero las mediciones conservarán interpretaciones separadas. La calibración se realizará una vez por familia de modelo, precisión numérica y protocolo de entrenamiento; su costo se contabilizará y se distribuirá entre las tareas en las que pueda reutilizarse.

El estudio principal empleará perceptrones multicapa y familias de reglas booleanas cuyo proceso generador, complejidad y ruido se conocen. Una fase complementaria usará redes recurrentes pequeñas y señales temporales controladas para establecer si las conclusiones se mantienen en tareas de pronóstico. Para cada tarea se observará una etapa inicial con presupuesto fijo; a partir de ella se intentará estimar el menor modelo que alcanza una fracción previamente definida del desempeño recuperable. El método se comparará con la cuenta de parámetros, extrapoladores de curvas de aprendizaje y criterios que no requieren entrenamiento.

El aporte esperado es un procedimiento reproducible para decidir cuándo el comportamiento temprano de una red contiene evidencia suficiente para dimensionarla. Un resultado negativo también será informativo: mostrará que estas mediciones describen aspectos del aprendizaje, pero no mejoran la selección de tamaño frente a alternativas más simples.

## 1. Problema de investigación

La representación numérica de los pesos y la capacidad de clasificación describen propiedades diferentes. MacKay analizó una neurona binaria de umbral bajo entradas en posición general y etiquetas aleatorias, y obtuvo una transición cercana a dos asociaciones binarias por peso [1]. Cover y Gardner estudiaron la geometría que sustenta este resultado [2], [3]. Su alcance es preciso: caracteriza una familia de clasificadores y una distribución de problemas, no establece una constante universal para redes profundas.

En redes multicapa, Friedland y Krell propusieron relaciones entre número de pesos y capacidad de clasificación, junto con reglas prácticas de dimensionamiento [4], [5]. Sin embargo, las redes actuales pueden ajustar etiquetas aleatorias [6] y, ante datos parcialmente estructurados, suelen aprender regularidades compartidas antes de memorizar excepciones [7]. Morris et al. separaron la memorización de ejemplos particulares del conocimiento sobre el proceso generador y estimaron, en transformadores tipo GPT entrenados sobre secuencias uniformes, una capacidad cercana a 3,6 bits por parámetro [8]. Esa cifra es un resultado empírico de su arquitectura y protocolo, no una ley aplicable a cualquier modelo.

La dificultad práctica aparece cuando se necesita elegir un modelo antes de pagar el costo de entrenar todas las alternativas. Dos redes con igual número de parámetros pueden alcanzar resultados distintos debido a su arquitectura, precisión numérica, optimizador y datos. Además, una pérdida baja en entrenamiento puede provenir de aprender estructura reutilizable o de ajustarse a realizaciones particulares del ruido. Las curvas parciales y los indicadores calculados en la inicialización ya intentan anticipar el desempeño [13]-[16], pero todavía es razonable preguntar si observar por separado memorización y generalización aporta información adicional para dimensionar el modelo.

Esta tesis abordará esa pregunta en procesos generadores conocidos. El control experimental permite conservar la regla verdadera, la realización del ruido y una referencia probabilística óptima. Con ello se puede definir qué significa desempeño suficiente, medir el ajuste particular de la muestra y comprobar la recuperación de la regla sin atribuir esas propiedades directamente al contenido físico de los pesos.

### Pregunta principal

> ¿Una calibración previa de la capacidad de memorización, combinada con mediciones tempranas de ajuste específico y generalización, mejora la estimación del menor tamaño de red que alcanza una meta de desempeño en una tarea nueva, frente a curvas parciales e indicadores sin entrenamiento?

## 2. Objetivos

### Objetivo general

Diseñar y evaluar un procedimiento para estimar, con un presupuesto inicial fijo, el menor tamaño de red que alcanza un nivel de generalización definido, utilizando tareas controladas que permiten distinguir memorización y aprendizaje de estructura.

### Objetivos específicos

1. **Caracterizar la capacidad empírica de memorización** de perceptrones multicapa y redes recurrentes pequeñas al variar arquitectura, número de parámetros, precisión numérica y protocolo de entrenamiento.
2. **Medir por separado ajuste específico y generalización** durante el aprendizaje de tareas con regla y ruido conocidos, y establecer en qué condiciones esas trayectorias pueden distinguirse de la variación estadística.
3. **Evaluar la estimación temprana del tamaño suficiente** en tareas no utilizadas para construir el método, frente a comparadores que reciben el mismo presupuesto de observación.

## 3. Marco teórico y estado del arte

### 3.1 Tamaño suficiente como decisión dependiente de una meta

Una red no es suficiente en abstracto: lo es respecto de una tarea, un criterio de desempeño y un nivel requerido. Para una tarea \(z\), sea \(L_{0,z}\) la pérdida de un predictor que no observa la entrada, \(L_{N,z}\) la pérdida de una red con \(N\) parámetros y \(L_{\star,z}\) la pérdida del predictor de referencia que conoce el proceso generador. Las tres cantidades se calcularán sobre el mismo conjunto de prueba reservado. Se definirá el aprovechamiento del desempeño recuperable como

\[
A_z(N)=\frac{L_{0,z}-L_{N,z}}{L_{0,z}-L_{\star,z}}.
\]

El límite inferior del intervalo para \(L_{0,z}-L_{\star,z}\) deberá ser positivo. Cuando el predictor de referencia no mejore de manera demostrable al predictor nulo, la tarea no define un problema de dimensionamiento informativo y se reportará por separado. El cociente no se recortará al intervalo \([0,1]\): los valores exteriores pueden aparecer por variación muestral o porque el modelo evaluado supere la referencia en esa muestra, y conservarlos evita sesgar la comparación.

Para un nivel requerido \(\tau\), y dentro de una familia de arquitectura y un protocolo de entrenamiento fijos, el tamaño suficiente observado será

\[
N_{\min}^{obs}(z,\tau)=
\min\left\{N\in\mathcal N:
\operatorname{LI}\bigl(A_z(N)\bigr)\geq\tau\right\},
\]

donde \(\mathcal N\) es el conjunto de tamaños fijado antes de la evaluación final y \(\operatorname{LI}\) es el límite inferior del intervalo de confianza. El valor primario de \(\tau\) y el nivel de confianza se fijarán con el piloto; también se publicará un análisis de sensibilidad sobre valores vecinos. Esta definición limita la afirmación al conjunto de modelos estudiado y evita presentar \(N_{\min}^{obs}\) como un mínimo universal.

Si ningún tamaño alcanza la meta, la tarea se registrará como \(N_{\min}^{obs}>N_{\max}\), es decir, censurada a la derecha. El estimador deberá reconocer esa categoría. El error en escala logarítmica se calculará donde el tamaño esté identificado, mientras que recomendar un tamaño finito para una tarea censurada contará como subdimensionamiento. Ninguna tarea desaparecerá por exceder el conjunto evaluado.

### 3.2 Capacidad empírica de memorización

La capacidad clásica de una neurona de umbral depende de la familia de funciones, la posición de las entradas y el criterio de aprendizaje [1]-[3]. Las extensiones de Friedland et al. muestran relaciones de escalamiento para redes de perceptrones, pero también distinguen la capacidad geométrica de lo que alcanza un algoritmo concreto [4], [5].

Morris et al. proponen eliminar la posibilidad de generalizar entrenando sobre secuencias aleatorias [8]. En ese escenario, toda reducción de pérdida atribuible al modelo procede de asociaciones particulares. Al aumentar la cantidad de datos, la reducción acumulada de pérdida en entrenamiento, corregida por el desempeño sobre asociaciones nuevas, crece hasta alcanzar una meseta. Esa meseta constituye una estimación empírica de capacidad bajo la arquitectura, precisión y entrenamiento utilizados.

Esta investigación adaptará ese experimento a modelos pequeños. Para cada familia \(F\), tamaño \(N\), algoritmo de entrenamiento \(O\) y precisión numérica \(b\), se estimará \(\widehat C_{mem}(F,N,O,b)\). Su función será caracterizar el sistema de aprendizaje y aportar una referencia previa al estudio de tareas estructuradas. La estabilidad de la meseta y su relación con el número de parámetros se evaluarán en repeticiones independientes.

### 3.3 Ajuste específico de la muestra y mejora fuera de muestra

La pérdida de entrenamiento no distingue por sí sola entre aprender una regularidad y recordar una excepción. Zhang et al. mostraron que las redes profundas pueden ajustar etiquetas aleatorias [6], mientras Arpit et al. observaron una separación temporal entre aprendizaje de patrones y ajuste del ruido [7]. Morris et al. expresan una distinción relacionada mediante conocimiento del proceso generador y memorización no intencional [8].

En las tareas binarias de esta tesis, el generador proporcionará la probabilidad verdadera \(p_\star(y\mid x)\). Sea \(S\) el conjunto de entrenamiento y \(T\) una muestra de referencia no utilizada para ajustar el modelo. El exceso de ajuste específico de la muestra se estimará como

\[
\widehat M_{muestra}(t)=
\sum_{i\in S}\log_2\frac{p_{\theta_t}(y_i\mid x_i)}{p_\star(y_i\mid x_i)}
-\frac{|S|}{|T|}
\sum_{i\in T}\log_2\frac{p_{\theta_t}(y_i\mid x_i)}{p_\star(y_i\mid x_i)}.
\]

La resta descuenta el comportamiento del mismo modelo fuera del entrenamiento y reduce la atribución de fluctuaciones generales a memorización. El contraste conservará su signo y su intervalo de incertidumbre; recortarlo en cero produciría un sesgo ascendente. Esta sigue siendo una definición operacional: mide exceso de ajuste asociado a la muestra, no permite inspeccionar directamente la información almacenada en cada peso.

La mejora predictiva se medirá sobre un conjunto de prueba reservado \(U\), comparando el modelo con un predictor nulo \(q_0\) ajustado sin usar \(U\):

\[
\widehat G_{pred}(t)=\frac{1}{|U|}
\sum_{(x,y)\in U}\log_2
\frac{p_{\theta_t}(y\mid x)}{q_0(y)}.
\]

Un valor positivo representa una reducción de pérdida logarítmica frente a no observar la entrada. Esta medida se relaciona con la información predictiva utilizable de Xu et al. [9], aunque aquí se calcula para un modelo y un punto de entrenamiento concretos. La compresibilidad y la longitud de descripción de modelos se conservarán como referencias complementarias [10], [11]; ninguna de ellas sustituye la evaluación fuera de muestra.

### 3.4 Recuperación e identificabilidad de la regla

Cada tarea principal tendrá una identidad latente \(Z\) elegida de un catálogo finito de \(M\) reglas. El comportamiento de la red se evaluará en entradas de contraste construidas antes del entrenamiento. Si un clasificador puede identificar \(Z\) desde ese comportamiento \(B(W)\) con probabilidad de error \(P_e\), la desigualdad de Fano y el procesamiento de datos permiten escribir [12]:

\[
I(Z;W)\geq I(Z;B(W))
\geq \log_2 M-h_2(P_e)-P_e\log_2(M-1).
\]

La expresión ofrece una cota inferior sobre la información demostrada acerca de la identidad de la regla. Se usará para estudiar identificabilidad y no como conversión automática de información a número de parámetros. El resultado teórico buscado establecerá condiciones bajo las cuales reglas distintas pueden o no distinguirse mediante un conjunto finito de entradas de contraste.

### 3.5 Estimación temprana y bancos temporales controlados

La estimación temprana del desempeño cuenta con antecedentes directos. Domhan et al. y Klein et al. extrapolan curvas parciales para detener configuraciones poco prometedoras [13], [14]. Mellor et al. estudian indicadores calculados sin entrenar la red [15], y NAS-Bench-Suite-Zero muestra que esos indicadores pueden aportar información complementaria y también sesgos dependientes del espacio de búsqueda [16]. Por tanto, la contribución propuesta no será anticipar el desempeño en general, sino determinar si las mediciones que distinguen ajuste específico y generalización agregan valor bajo el mismo presupuesto de observación.

Para la fase temporal, SynTSBench proporciona un antecedente cercano: utiliza configuraciones programables de tendencia, estacionalidad, ruido y otras propiedades para evaluar capacidades de modelos de pronóstico frente a referencias teóricas [17]. El banco de esta tesis adoptará controles compatibles y comprobará si las mediciones de la fase principal siguen siendo útiles en tareas secuenciales. Como extensión opcional, una selección previamente fijada de sistemas dinámicos de `dysts` permitirá comprobar el procedimiento en trayectorias generadas por ecuaciones conocidas [18].

## 4. Hipótesis

**H1 - Capacidad empírica.** En cada combinación de arquitectura, precisión y protocolo de entrenamiento, la memorización observada alcanzará una meseta reproducible al aumentar la cantidad de etiquetas aleatorias. Un modelo de escalamiento que represente las diferencias entre esas combinaciones predecirá la meseta en configuraciones reservadas con menor error que una relación global de la forma \(C=\alpha N\). H1 no se sostendrá si la meseta no puede distinguirse de entrenamiento insuficiente o si su variación entre repeticiones impide ordenarla por tamaño.

**H2 - Separación de trayectorias.** En tareas con realizaciones aleatorias particulares del ruido, existirán intervalos de entrenamiento, definidos previamente en la fase de calibración, en los que el exceso de ajuste específico aumente sin superar la mejora mínima fijada para el desempeño fuera de muestra ni para la recuperación de la regla. H2 no se sostendrá si las trayectorias no pueden distinguirse con la precisión establecida o si el aumento de ajuste específico mejora de forma reproducible la generalización.

**H3 - Estimación del tamaño suficiente.** En tareas finales no usadas para desarrollar el método, un estimador regularizado que calcule la probabilidad de alcanzar la meta para cada tamaño candidato reducirá el error absoluto de \(\log_2 N_{\min}^{obs}\) frente al mejor comparador de igual presupuesto elegido durante la calibración. Además, no aumentará, más allá del margen de no inferioridad, la tasa de recomendar modelos insuficientes ni la de asignar un tamaño finito a tareas cuyo mínimo supera el máximo evaluado. La mejora mínima y el margen se determinarán en el piloto y quedarán registrados antes de la evaluación final. H3 no se sostendrá si el intervalo de la diferencia pareada incluye la ausencia de mejora o si falla la condición de no inferioridad.

## 5. Metodología

### 5.1 Familias de tareas

La familia principal estará formada por funciones representadas mediante diagramas de decisión binarios reducidos y ordenados. Con un orden de variables fijo, esta representación es canónica y permite asignar a cada tarea una identidad y una medida estructural reproducibles. Cada tarea se construirá como

\[
Z\sim\operatorname{Unif}\{1,\ldots,M\},\qquad
S=f_Z(X),\qquad
Y=S\oplus E,\qquad E\sim\operatorname{Bernoulli}(\eta),
\]

donde \(S\) es la etiqueta limpia y \(E\) introduce ruido independiente con tasa conocida. El generador conservará la entrada, la identidad de la regla, la etiqueta limpia, el ruido realizado y la etiqueta observada. Para dimensiones pequeñas se evaluará la tabla de verdad completa; para dimensiones mayores se empleará un conjunto de entradas de contraste fijado antes del entrenamiento.

La fase complementaria empleará señales temporales controladas. Para una tarea \(Z\), la señal limpia tendrá la forma general

\[
s_Z(t)=a_Z+b_Zt+
\sum_{k=1}^{K_Z}A_{Z,k}\sin(2\pi f_{Z,k}t+\phi_{Z,k})+r_Z(t),
\qquad y(t)=s_Z(t)+\epsilon(t),
\]

donde \(r_Z(t)\) representará, según la condición experimental, una frecuencia variable, un cambio de régimen o una dependencia autorregresiva, y \(\epsilon(t)\) seguirá una distribución conocida. Se incluirán condiciones sin ruido y con ruido gaussiano de desviación estándar conocida. El banco avanzará desde componentes aislados hasta combinaciones: tendencia, una frecuencia, varias frecuencias, variación de frecuencia, cambio de régimen y memoria no lineal. Las frecuencias respetarán el límite de Nyquist y se fijará un orden canónico para evitar que dos parámetros describan la misma tarea.

Las particiones temporales se harán por identidad del proceso generador, no mezclando al azar puntos vecinos de una misma trayectoria. La señal limpia y la distribución condicional conocida proporcionarán la referencia de desempeño. En este documento, **pronóstico** designa la estimación de valores futuros \(y_{t+h}\) usando solo información disponible hasta \(t\); **predicción** conserva su sentido general para clasificación y estimación estadística [19].

### 5.2 Modelos y protocolo de entrenamiento

El estudio principal utilizará perceptrones multicapa; la fase temporal empleará redes GRU pequeñas. El piloto fijará un conjunto logarítmico de tamaños, desde modelos deliberadamente insuficientes hasta modelos cuya curva de desempeño se haya estabilizado. Se registrarán el número exacto de parámetros entrenables, inicialización, optimizador, precisión numérica, presupuesto de actualizaciones y criterio de parada.

Todos los métodos de H3 recibirán la misma observación preliminar: dos tamaños pequeños y un número fijo de actualizaciones determinados con el piloto. El estimador propuesto solo podrá usar \(\widehat C_{mem}\), \(\widehat M_{muestra}\), \(\widehat G_{pred}\), recuperación de la regla y descriptores de tarea disponibles dentro de ese presupuesto. Para cada tamaño candidato devolverá una probabilidad de alcanzar la meta; una regla fijada en calibración elegirá el menor tamaño respaldado o responderá \(N_{\min}^{obs}>N_{\max}\) cuando la evidencia no respalde ninguno. El cálculo verdadero de \(N_{\min}^{obs}\) se realizará después, recorriendo el conjunto completo de tamaños de la misma familia, y funcionará exclusivamente como variable de respuesta.

Los modelos producirán distribuciones probabilísticas evaluables mediante pérdida logarítmica dentro de cada tarea. En clasificación se examinará la calibración de las probabilidades; en pronóstico se declarará la distribución de los errores, la estimación de su varianza y la cobertura de los intervalos. Las pérdidas se normalizarán frente a los predictores nulo y de referencia de cada tarea; sus valores brutos no se compararán entre problemas con espacios de salida distintos. Estas serán variables de diagnóstico, no condiciones añadidas después para redefinir el tamaño suficiente. El diseño evita tratar el error cuadrático, la exactitud y la entropía como cantidades intercambiables.

### 5.3 Particiones y unidad de análisis

Las identidades de tarea se dividirán antes de entrenar en cuatro grupos:

1. **piloto**, para comprobar factibilidad y fijar rangos;
2. **desarrollo**, para construir el estimador de tamaño;
3. **calibración**, para fijar umbrales, intervalos y reglas de decisión;
4. **evaluación final**, reservada para decidir las hipótesis confirmatorias.

La tarea generada será la unidad independiente. Tamaños y métodos serán tratamientos pareados dentro de cada tarea; las semillas serán repeticiones anidadas y su variación se informará por separado. El número de tareas se determinará mediante el piloto para alcanzar una precisión objetivo del intervalo alrededor de una diferencia mínima relevante definida por la decisión de recursos, no por el efecto observado en el piloto. Si el presupuesto no permite esa precisión, el estudio se declarará insuficientemente potente para la comparación correspondiente.

### 5.4 Comparadores y atribución

Los comparadores de H3 resolverán la misma decisión y recibirán exactamente los mismos datos tempranos:

- un modelo que use solo número de parámetros y descriptores básicos de la tarea;
- extrapolación de curvas parciales [13], [14];
- indicadores sin entrenamiento compatibles con las arquitecturas estudiadas [15], [16];
- un modelo monótono basado en tamaño del conjunto, tasa de ruido y complejidad declarada de la regla.

Las relaciones de MacKay y Friedland se analizarán como referencias de capacidad bajo sus respectivos supuestos, no como sustitutos automáticos de estos comparadores. Se incluirá una ablación que retire cada medición del estimador propuesto; así se podrá determinar si la mejora, cuando exista, procede de capacidad, ajuste específico o desempeño fuera de muestra.

### 5.5 Análisis estadístico y reproducibilidad

H1 se evaluará mediante curvas de saturación y validación entre configuraciones y repeticiones independientes. H2 usará contrastes pareados en puntos de medición fijados con calibración, sin buscar retrospectivamente el intervalo más favorable. H3 usará como variable principal el error absoluto de \(\log_2 N_{\min}^{obs}\) en tareas con tamaño identificado. La detección de tareas censuradas y la tasa de subdimensionamiento serán condiciones separadas de no inferioridad.

Los intervalos se calcularán a nivel de tarea, respetando la estructura anidada de semillas. Las observaciones independientes permitirán remuestreo ordinario; las señales temporales usarán bloques que conserven su dependencia. Las comparaciones confirmatorias se corregirán por multiplicidad mediante el procedimiento de Holm. Fallos de entrenamiento, tareas no identificables y resultados negativos permanecerán en sus denominadores y se publicarán con razones definidas antes del análisis.

El protocolo, los generadores, las particiones, las semillas y el plan estadístico se registrarán antes de abrir la evaluación final. Los materiales de reproducción incluirán la configuración efectiva, las versiones del código, los registros de entrenamiento y las tablas derivadas de resultados individuales.

### 5.6 Alcance de la interpretación

El experimento medirá comportamiento bajo arquitecturas y protocolos concretos. No pretende medir inteligencia general ni inventariar físicamente la información de cada peso. La capacidad empírica sobre datos aleatorios, el exceso de ajuste de la muestra y la mejora fuera de muestra son variables relacionadas, pero no forman una partición exhaustiva de una memoria común.

Los resultados principales se limitarán a las familias generativas estudiadas. Las señales temporales mostrarán si el procedimiento conserva utilidad en tareas secuenciales; los sistemas caóticos y los datos naturales, si se incluyen, servirán como comprobaciones externas y no como fuentes de una regla verdadera desconocida.

## 6. Aportes esperados

1. **Un protocolo de medición** que distingue capacidad empírica, exceso de ajuste específico y desempeño fuera de muestra durante el entrenamiento.
2. **Una evaluación confirmatoria del dimensionamiento temprano**, con comparadores de igual presupuesto y una definición explícita de tamaño suficiente dependiente de la meta.
3. **Una caracterización de límites**, teórica mediante identificabilidad de reglas y empírica mediante los casos en que las mediciones no mejoran decisiones de tamaño.

La utilidad práctica se evaluará como reducción del entrenamiento necesario para recomendar un tamaño que cumpla el objetivo, descontando el costo de la calibración previa y mostrando cuántas tareas deben reutilizarla para amortizarla. La utilidad científica consiste en establecer cuáles observaciones tempranas distinguen memorización de aprendizaje reutilizable y cuándo esa distinción deja de apoyar una decisión de arquitectura.

## 7. Factibilidad, riesgos y ética

Los datos principales serán sintéticos y no contendrán información personal. El estudio no requiere participantes humanos ni decisiones sobre personas. Los modelos estarán acotados aproximadamente a un millón de parámetros y el piloto medirá tiempo, memoria y número de ejecuciones antes de fijar el diseño confirmatorio.

El principal riesgo es que el estimador no supere una curva parcial de aprendizaje. Ese resultado rechazaría H3 y mostraría que las mediciones adicionales no justifican su costo para dimensionar modelos. Otro riesgo es que la capacidad observada dependa más del optimizador que de la arquitectura; en ese caso se tratará como propiedad del sistema completo de entrenamiento. Finalmente, la fase temporal puede no reproducir los resultados booleanos. Esa divergencia delimitará el alcance del método entre familias y no se ocultará combinando ambas poblaciones.

La fase principal, las tres hipótesis y el análisis de identificabilidad constituyen el alcance mínimo defendible. Si el costo supera el presupuesto, se reducirán primero los sistemas dinámicos opcionales y después las combinaciones temporales, conservando los componentes aislados como comprobación.

## 8. Plan de trabajo

| Periodo | Actividades y entregables |
|---|---|
| Meses 1-6 | Revisión sistemática, formalización de medidas, generadores y estudio piloto. |
| Meses 7-12 | Calibración de capacidad empírica, validación de estimadores y prerregistro confirmatorio. |
| Meses 13-20 | Trayectorias de ajuste específico y generalización en la familia booleana. |
| Meses 21-27 | Estimador de tamaño, comparadores, ablaciones y evaluación final. |
| Meses 28-32 | Comprobación con señales temporales controladas y análisis de alcance entre familias. |
| Meses 33-36 | Integración, publicaciones, divulgación de materiales reproducibles y escritura de tesis. |

## Referencias

[1] D. J. C. MacKay, *Information Theory, Inference, and Learning Algorithms*. Cambridge, Reino Unido: Cambridge University Press, 2003, cap. 40.

[2] T. M. Cover, "Geometrical and statistical properties of systems of linear inequalities with applications in pattern recognition," *IEEE Transactions on Electronic Computers*, vol. EC-14, no. 3, pp. 326-334, 1965.

[3] E. Gardner, "The space of interactions in neural network models," *Journal of Physics A: Mathematical and General*, vol. 21, no. 1, pp. 257-270, 1988.

[4] G. Friedland and M. Krell, "A capacity scaling law for artificial neural networks," arXiv:1708.06019, 2018.

[5] G. Friedland, A. Metere, and M. Krell, "A practical approach to sizing neural networks," arXiv:1810.02328, 2018.

[6] C. Zhang, S. Bengio, M. Hardt, B. Recht, and O. Vinyals, "Understanding deep learning requires rethinking generalization," in *Proc. ICLR*, 2017.

[7] D. Arpit et al., "A closer look at memorization in deep networks," in *Proc. ICML*, pp. 233-242, 2017.

[8] J. X. Morris et al., "How much do language models memorize?" arXiv:2505.24832, 2025.

[9] Y. Xu, S. Zhao, J. Song, R. Stewart, and S. Ermon, "A theory of usable information under computational constraints," in *Proc. ICLR*, 2020.

[10] L. Blier and Y. Ollivier, "The description length of deep learning models," in *Advances in Neural Information Processing Systems*, vol. 31, 2018.

[11] J. Bernstein and Y. Yue, "Computing the information content of trained neural networks," arXiv:2103.01045, 2021.

[12] T. M. Cover and J. A. Thomas, *Elements of Information Theory*, 2nd ed. Hoboken, NJ, EE. UU.: Wiley, 2006.

[13] T. Domhan, J. T. Springenberg, and F. Hutter, "Speeding up automatic hyperparameter optimization of deep neural networks by extrapolation of learning curves," in *Proc. IJCAI*, pp. 3460-3468, 2015.

[14] A. Klein, S. Falkner, J. T. Springenberg, and F. Hutter, "Learning curve prediction with Bayesian neural networks," in *Proc. ICLR*, 2017.

[15] J. Mellor, J. Turner, A. Storkey, and E. J. Crowley, "Neural architecture search without training," in *Proc. ICML*, pp. 7588-7598, 2021.

[16] A. Krishnakumar, C. White, A. Zela, R. Tu, M. Safari, and F. Hutter, "NAS-Bench-Suite-Zero: Accelerating research on zero cost proxies," in *Advances in Neural Information Processing Systems*, vol. 35, 2022.

[17] Q. Tan, Y. Chen, M. Li, R. Gu, Y. Su, and X.-P. Zhang, "SynTSBench: Rethinking temporal pattern learning in deep learning models for time series," in *Advances in Neural Information Processing Systems*, vol. 38, 2025.

[18] W. Gilpin, "Chaos as an interpretable benchmark for forecasting and data-driven modelling," in *Advances in Neural Information Processing Systems*, vol. 34, 2021.

[19] R. J. Hyndman and G. Athanasopoulos, *Forecasting: Principles and Practice*, 3rd ed. Melbourne, Australia: OTexts, 2021.

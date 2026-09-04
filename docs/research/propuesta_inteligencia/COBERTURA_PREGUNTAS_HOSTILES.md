# Preguntas hostiles para la propuesta sobre memorización y dimensionamiento

**Documento asociado:** `PROPUESTA_DOCTORAL_MEMORIZACION_GENERALIZACION_DIMENSIONAMIENTO.md`
**Uso:** preparación de entrevista y revisión metodológica interna.  
**Regla:** responder lo que el diseño permite afirmar. Si una relación no está identificada, no se completa con intuición.

## Núcleo de la propuesta

### 1. ¿Cuál es la tesis en una frase?

Quiero saber si una calibración previa de la capacidad de memorización, combinada con mediciones tempranas de ajuste específico y generalización, ayuda a estimar el menor tamaño de modelo que alcanza una meta de desempeño.

### 2. ¿Dónde está el problema de inteligencia artificial y no solo de compresión?

La decisión estudiada es propia del aprendizaje automático: qué tamaño de red necesita una tarea para generalizar. La compresión aporta medidas, pero el objeto es el comportamiento de sistemas entrenables frente a reglas, ruido, datos nuevos y restricciones de arquitectura.

### 3. ¿Está intentando medir inteligencia?

No. La inteligencia incluye adaptación, transferencia y muchas capacidades que este protocolo no cubre. La tesis estudia capacidad empírica, ajuste específico, generalización y reconocimiento de reglas en familias controladas.

### 4. ¿Por qué importa dimensionar modelos si hoy se usan redes sobredimensionadas?

Porque sobredimensionar aumenta costo de entrenamiento, inferencia y búsqueda, y puede ser inviable en dispositivos limitados. Además, entender cuándo los parámetros se usan para memorizar ejemplos o para representar una regla es una pregunta científica aunque el cómputo fuera gratuito.

### 5. ¿Cuál es la contribución nueva?

Es comprobar, en tareas finales reservadas, si combinar una calibración reutilizable con esas mediciones tempranas estima el tamaño suficiente mejor que curvas parciales e indicadores sin entrenamiento con el mismo presupuesto. El costo de calibrar se incluirá en la comparación. También se caracterizarán las condiciones en las que el método deja de ayudar.

## Fundamentos y mediciones

### 6. MacKay dice dos bits por peso. ¿Por qué no usa directamente esa regla?

Porque el resultado corresponde a una neurona de umbral, entradas en posición general y etiquetas binarias aleatorias. Es una transición geométrica de capacidad, no una ley de almacenamiento útil para cualquier red profunda. En esta tesis es un control y un antecedente, no la respuesta.

### 7. Morris ya encontró aproximadamente 3,6 bits por parámetro. ¿La tesis es una réplica pequeña?

La calibración de capacidad sí adapta parte de ese protocolo a modelos pequeños y se reconoce como calibración. La pregunta doctoral aparece después: si las medidas obtenidas durante corridas piloto ayudan a estimar el tamaño suficiente en tareas estructuradas nuevas. Morris no evalúa esa decisión.

### 8. ¿Por qué no puede dividir información útil por capacidad y llamarla ocupación?

Porque los dos números describen objetos distintos. La capacidad aleatoria mide ejemplos particulares recuperables. La información predictiva mide reducción de incertidumbre en datos no vistos gracias a una regla reutilizable. Una regla corta puede ahorrar bits en millones de predicciones sin ocupar millones de bits en los pesos. Compartir la unidad “bit” no autoriza el cociente.

### 9. Entonces, ¿el término ocupación desaparece por completo?

Sí. El diseño no identifica una fracción física de pesos ocupados. La capacidad empírica, el exceso de ajuste y la mejora fuera de muestra se publican por separado porque responden preguntas distintas.

### 10. ¿Qué significa ganancia predictiva?

Es la reducción de pérdida logarítmica que logra el modelo al observar la entrada, comparada con un predictor nulo que no la observa. Se mide en datos no usados para entrenar. Cuando se usa logaritmo en base dos, su unidad es bits por ejemplo. También se compara con el predictor de referencia que conoce el proceso generador para definir la meta de desempeño de cada tarea.

### 11. ¿Cómo distingue memorización de generalización?

El generador sintético conserva la regla, la etiqueta limpia y el ruido realizado. El ajuste específico se estima mediante la diferencia entre el contraste del entrenamiento y el de una muestra de referencia no usada para ajustar el modelo. El contraste conserva su signo y su incertidumbre. La generalización se evalúa en otro conjunto reservado y la regla se comprueba con entradas de contraste definidas antes de entrenar.

### 12. ¿Por qué una longitud comprimida no da una cota inferior de tamaño?

Porque cualquier compresor produce una descripción posible y, por tanto, una cota **superior** de la complejidad de Kolmogorov. No prueba que una descripción más corta sea imposible. La propuesta observa el menor tamaño dentro de un conjunto fijado y trata de estimarlo; no lo deriva de un compresor.

### 13. ¿Qué aporta Fano?

Para una regla elegida entre un catálogo finito, Fano relaciona el error de identificarla con una cantidad mínima de información sobre su identidad. Es una cota válida en bits. No entrega por sí sola el número de neuronas; para eso haría falta limitar cuántos bits puede representar cada estado del modelo.

### 14. ¿No contradice esto la afirmación de MacKay?

No. MacKay ofrece una caracterización mucho más específica y más fuerte para clasificadores de umbral. Fano ofrece una condición general de identificabilidad. Una es un comparador bajo supuestos geométricos; la otra es un piso informativo para un catálogo finito.

## Tareas y medición

### 15. ¿Qué es exactamente una regla booleana canónica?

Será una función representada por un diagrama de decisión binario reducido y ordenado. Al fijar el orden de variables, la representación es canónica y evita contar dos veces la misma función. El número de nodos es un factor estructural del experimento, no complejidad de Kolmogorov.

### 16. ¿Por qué no usar simplemente longitud de código del generador?

Porque programas largos pueden calcular funciones simples y programas diferentes pueden calcular la misma función. La longitud sintáctica puede ser un factor experimental, pero no se llamará complejidad mínima. La identidad finita de la regla y su representación canónica son las autoridades del experimento.

### 17. ¿Cómo sabe que el modelo reconoció la regla y no memorizó las sondas?

Las sondas no se usan en entrenamiento ni en selección de parada. Se construyen para separar reglas del catálogo e incluyen intervenciones poco probables bajo el muestreo de entrenamiento. Una respuesta correcta exige reproducir el comportamiento de la regla en casos nuevos.

### 18. ¿Cómo se medirá la recuperación de la regla?

El acuerdo entre el modelo y la regla verdadera en entradas contrafactuales. Para tareas pequeñas se puede evaluar la tabla de verdad completa; para tareas mayores se usa un conjunto separador fijado antes del entrenamiento. No es una medida general de inteligencia.

### 19. ¿Por qué usar señales temporales como confirmación?

Porque permiten comprobar si las mediciones siguen siendo útiles fuera de MLP y reglas estáticas, manteniendo conocida la señal limpia, el ruido y la ley generadora. El banco separa tendencia, periodicidad, frecuencia variable, cambios de régimen y memoria no lineal antes de combinarlos. Si el resultado no se mantiene en una red recurrente pequeña, esa diferencia limita el alcance del protocolo.

### 20. ¿Por qué no usar datos reales?

En datos reales no conocemos la regla verdadera, la cantidad de ruido ni el conjunto completo de intervenciones que distingue explicaciones. Serían útiles como comprobación externa después de validar las medidas, pero no para establecer qué significa cada una.

## Diseño experimental

### 21. ¿Cuál es la unidad estadística?

Una tarea generada de manera independiente, identificada por su regla y condiciones. Arquitecturas y tamaños son tratamientos pareados dentro de esa tarea. Las semillas son repeticiones del algoritmo y no aumentan artificialmente el número de tareas.

### 22. ¿Cuántas tareas harán falta?

El piloto estimará la variación entre tareas y fijará una precisión objetivo del intervalo y una diferencia mínima relevante. De allí saldrá el número de tareas antes de la evaluación final. Si el presupuesto no alcanza esa precisión, la comparación se declarará insuficientemente potente.

### 23. ¿Cómo evita escoger la mejor definición después de ver resultados?

Piloto, desarrollo, calibración y evaluación final usan identidades de regla separadas. El piloto fija rangos; desarrollo construye el estimador; calibración fija umbrales; la evaluación final se usa una sola vez para decidir las hipótesis. Los cambios posteriores se informan como análisis exploratorios.

### 24. ¿La parada temprana sigue siendo parte central?

Es una parte del protocolo de entrenamiento, no la definición de la variable principal. La regla de parada se fija durante la calibración. Las hipótesis se deciden con tareas y ejemplos reservados que no participaron en esa elección.

### 25. ¿Qué es \(N_{min}^{obs}\)?

Para una tarea, una familia de arquitectura y una meta \(\tau\), es el modelo más pequeño del conjunto previamente fijado cuya cota inferior alcanza la fracción requerida del desempeño recuperable. Es un mínimo observado dentro de ese conjunto, no el menor modelo matemáticamente posible.

### 26. ¿Cómo podría ganar de manera engañosa el predictor propuesto?

Consultando casi todos los tamaños antes de emitir la estimación o recomendando redes demasiado pequeñas. Todos los métodos observarán los mismos dos modelos iniciales durante el mismo número de actualizaciones. La reducción de error tampoco basta: la tasa de subdimensionamiento es una condición separada. El método no gana si ahorra parámetros a costa de fallar la tarea.

### 27. ¿Contra qué se compara?

Contra un modelo basado en descriptores básicos, extrapoladores de curvas parciales, indicadores sin entrenamiento y un modelo monotónico que use tamaño de datos, ruido y complejidad declarada. MacKay y Friedland son referencias de capacidad, no competidores automáticos para esta decisión. Si una curva sencilla gana, H3 no se sostiene.

### 28. ¿Cinco semillas son suficientes?

El número se fijará en el piloto según la inestabilidad del entrenamiento. En cualquier caso, las semillas se reportan como variación anidada; no se tratan como la población científica ni sustituyen tareas independientes.

### 29. ¿Qué ocurre con entrenamientos fallidos?

Se conservan y se clasifican. Un tamaño que no entrena bajo el presupuesto fijado no puede desaparecer del denominador. Se hará análisis de sensibilidad para distinguir insuficiencia de representación y dificultad de optimización, sin cambiar el veredicto principal después de verlo.

## Novedad, alcance y viabilidad

### 30. Arpit ya mostró que primero se aprenden patrones y luego ruido. ¿Qué queda para H2?

H2 no reclama descubrir esa secuencia por primera vez. Verifica que las mediciones elegidas la distinguen con suficiente precisión y examina si aportan información a H3. Si no la distinguen, esa parte del estimador no queda justificada.

### 31. Friedland ya propuso dimensionar redes. ¿Qué queda para H3?

Precisamente la comparación conceptual. Friedland relaciona capacidad y tamaño bajo supuestos propios. H3 pregunta si observar ajuste y generalización durante un presupuesto inicial mejora una decisión sobre tareas nuevas. Si no mejora los comparadores prácticos, el método propuesto no queda respaldado.

### 32. NAS ya predice el desempeño sin entrenar. ¿Qué añade esta tesis?

La búsqueda de arquitecturas sin entrenamiento y la extrapolación de curvas son los vecinos directos. Por eso entran como comparadores con el mismo presupuesto. La afirmación nueva es más estrecha: que distinguir ajuste específico y generalización mejora la estimación del **tamaño suficiente en tareas nuevas**. Si los métodos existentes igualan o superan al propuesto, H3 no se sostiene.

### 33. ¿No está prometiendo dos tesis, una para MLP y otra para GRU?

No. La familia booleana con MLP contiene el estudio principal. El banco temporal con GRU es una confirmación recortable. El resultado doctoral no depende de que ambas familias produzcan la misma constante ni del éxito de la confirmación.

### 34. ¿Es viable en tres años?

Los modelos están acotados aproximadamente a un millón de parámetros y los datos son sintéticos. El piloto medirá costo antes de fijar el conjunto de tamaños. El alcance mínimo termina con la familia booleana. En la fase temporal se recortan primero los sistemas caóticos y después las combinaciones de componentes; las señales aisladas quedan como comprobación mínima.

### 35. ¿Cuál sería un resultado negativo valioso?

Que las mediciones no mejoren una curva parcial, que dependan demasiado del optimizador o que sus resultados no se mantengan en señales temporales. Cada resultado establecería un límite concreto para el dimensionamiento temprano.

### 36. ¿Qué afirmación no debe hacerse durante la entrevista?

Debe afirmarse exactamente el alcance: se medirá el comportamiento de modelos pequeños bajo procesos y protocolos conocidos, y se observará el menor tamaño que cumple una meta dentro del conjunto evaluado. El diseño no autoriza generalizaciones sobre inteligencia, ocupación física de pesos ni tamaños mínimos universales.

## Señales temporales y terminología

### 37. ¿Por qué no basta una suma de senos con ruido gaussiano?

Porque es un control necesario, pero demasiado simple como confirmación completa. Una frecuencia fija puede resolverse con una representación compacta y ofrece atajos espectrales. El banco empieza allí y añade varias frecuencias, frecuencia variable, cambios de régimen y memoria no lineal. Cada componente se evalúa aislado antes de estudiar interacciones.

### 38. ¿Cómo evita fuga entre entrenamiento y prueba temporal?

La partición se hace por identidad de la ley generadora: combinaciones distintas de parámetros latentes pertenecen a piloto, ajuste, calibración y prueba. No se mezclan al azar puntos vecinos de una misma trayectoria. Además, toda ventana de pronóstico usa únicamente información disponible hasta su origen temporal.

### 39. ¿La fórmula \(\tfrac12\log_2(1+\mathrm{SNR})\) dice cuánta memoria usa la red?

No. Esa expresión corresponde a información o capacidad de un canal gaussiano bajo supuestos concretos de distribución, independencia y potencia. No describe automáticamente una señal sinusoidal correlacionada ni el contenido de los pesos. En esta tesis solo será una referencia cuando sus supuestos se satisfagan.

### 40. ¿Por qué no usar MNIST o el archivo UCR como fuente ideal?

Porque no conocemos exactamente su proceso generador, el nivel de ruido ni una regla verdadera que permita separar memoria y estructura. Pueden servir como comprobación externa, pero no como referencia para atribuir cada error a señal o ruido conocido.

### 41. SynTSBench ya genera tendencia, estacionalidad y ruido. ¿Dónde está la novedad?

No está en generar esas señales. SynTSBench es un antecedente y comparador directo. La pregunta propia es si las mediciones tempranas de memorización y generalización mejoran la estimación del tamaño suficiente. Si no mejoran los comparadores existentes, H3 no se sostiene.

### 42. ¿Por qué el documento dice “predicción” y no siempre “pronóstico”?

Porque no son sustitutos universales. Se usa **pronóstico** para estimar un valor futuro \(y_{t+h}\) con información disponible hasta \(t\). Se usa **predicción** en el sentido más amplio de aprendizaje automático: clasificación, predictor nulo, información predictiva o estimación del tamaño suficiente. En español ambos términos son válidos; la distinción evita llamar pronóstico a una clasificación sin eje futuro.

### 43. ¿Qué ocurre si ni el modelo más grande alcanza la meta?

La tarea queda censurada a la derecha: solo sabemos que su tamaño suficiente supera el máximo evaluado. No se elimina ni recibe un tamaño inventado. El estimador debe reconocer la categoría “mayor que el máximo”; recomendar un tamaño finito en ese caso cuenta como subdimensionamiento.

### 44. ¿Por qué todavía aparece la base dos en las ecuaciones?

Porque una diferencia de pérdida logarítmica necesita una base, y la base dos ofrece una unidad interpretable. Eso no convierte la tesis en un “estudio en bits” ni hace que capacidad, ajuste y generalización sean porciones de una misma memoria. La decisión científica no depende de cambiar el logaritmo natural por el logaritmo en base dos.

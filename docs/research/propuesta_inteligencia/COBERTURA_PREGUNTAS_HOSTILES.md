# Preguntas hostiles para la propuesta sobre memorización y ganancia predictiva

**Documento asociado:** `PROPUESTA_DOCTORAL_CAPACIDAD_OCUPACION_BITS.md`  
**Uso:** preparación de entrevista y revisión metodológica interna.  
**Regla:** responder lo que el diseño permite afirmar. Si una relación no está identificada, no se completa con intuición.

## Núcleo de la propuesta

### 1. ¿Cuál es la tesis en una frase?

Quiero saber si separar, en bits, la memorización de ejemplos particulares y la ganancia al predecir casos nuevos ayuda a anticipar el menor tamaño de modelo que basta para una tarea.

### 2. ¿Dónde está el problema de inteligencia artificial y no solo de compresión?

La decisión estudiada es propia del aprendizaje automático: qué tamaño de red necesita una tarea para generalizar. La compresión aporta medidas, pero el objeto es el comportamiento de sistemas entrenables frente a reglas, ruido, datos nuevos y restricciones de arquitectura.

### 3. ¿Está intentando medir inteligencia?

No. La inteligencia incluye adaptación, transferencia y muchas capacidades que este protocolo no cubre. La tesis mide memoria específica, ganancia predictiva y reconocimiento de reglas en familias controladas. Es una base cuantitativa para estudiar aprendizaje, no un cociente universal de inteligencia.

### 4. ¿Por qué importa dimensionar modelos si hoy se usan redes sobredimensionadas?

Porque sobredimensionar aumenta costo de entrenamiento, inferencia y búsqueda, y puede ser inviable en dispositivos limitados. Además, entender cuándo los parámetros se usan para memorizar ejemplos o para representar una regla es una pregunta científica aunque el cómputo fuera gratuito.

### 5. ¿Cuál es la contribución nueva?

No es una nueva cifra de bits por parámetro. Es comprobar, en prueba intacta, si un perfil que mantiene separadas capacidad, memorización específica y ganancia predictiva predice el tamaño suficiente mejor que reglas y curvas de aprendizaje ya disponibles. La región donde no funciona también queda identificada.

## Fundamentos en bits

### 6. MacKay dice dos bits por peso. ¿Por qué no usa directamente esa regla?

Porque el resultado corresponde a una neurona de umbral, entradas en posición general y etiquetas binarias aleatorias. Es una transición geométrica de capacidad, no una ley de almacenamiento útil para cualquier red profunda. En esta tesis es un control y un antecedente, no la respuesta.

### 7. Morris ya encontró aproximadamente 3,6 bits por parámetro. ¿La tesis es una réplica pequeña?

La calibración de capacidad sí adapta parte de ese protocolo a modelos pequeños y se reconoce como calibración. La pregunta doctoral aparece después: si las medidas obtenidas durante corridas piloto ayudan a estimar el tamaño suficiente en tareas estructuradas nuevas. Morris no evalúa esa decisión.

### 8. ¿Por qué no puede dividir información útil por capacidad y llamarla ocupación?

Porque los dos números describen objetos distintos. La capacidad aleatoria mide ejemplos particulares recuperables. La información predictiva mide reducción de incertidumbre en datos no vistos gracias a una regla reutilizable. Una regla corta puede ahorrar bits en millones de predicciones sin ocupar millones de bits en los pesos. Compartir la unidad “bit” no autoriza el cociente.

### 9. Entonces, ¿el término ocupación desaparece por completo?

Desaparece como variable principal. Solo puede reportarse, de manera secundaria, la memorización específica dividida por una capacidad empírica obtenida con el mismo protocolo. Aun así se llama carga relativa, no ocupación útil, y no se supone limitada a uno porque la capacidad observada es una estimación. La cota sobre la regla aprendida y el aprovechamiento predictivo se publican aparte: no se suman como si fueran compartimentos exhaustivos de los pesos.

### 10. ¿Qué significa ganancia predictiva?

Es la reducción de pérdida logarítmica que logra el modelo al observar la entrada, comparada con un predictor nulo que no la observa. Se mide en datos no usados para entrenar y se expresa en bits por ejemplo. La información predictiva \(\mathcal V\) es el antecedente formal, pero la variable de esta tesis es una ganancia del punto de entrenamiento, no contenido físico de los pesos. También se compara con el oráculo para saber qué fracción de la ganancia recuperable alcanzó el modelo; esa fracción tampoco es memoria ocupada.

### 11. ¿Cómo distingue memorización de generalización?

El generador sintético es conocido. Permite calcular qué parte de la predicción explica la regla y qué parte corresponde a realizaciones particulares, como ruido de etiqueta. El contraste de longitudes de código se suma sobre la muestra completa; no se conservan solo los ejemplos favorables. Además se repite sobre datos intactos: un exceso que también aparezca allí se clasifica como descalibración o diferencia respecto del oráculo, no como memoria específica. La generalización se evalúa sobre entradas nuevas y sondas contrafactuales.

### 12. ¿Por qué una longitud comprimida no da una cota inferior de tamaño?

Porque cualquier compresor produce una descripción posible y, por tanto, una cota **superior** de la complejidad de Kolmogorov. No prueba que una descripción más corta sea imposible. La versión anterior invertía esa desigualdad. La propuesta revisada observa el menor tamaño en una grilla y trata de predecirlo; no finge derivarlo de un compresor.

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

### 18. ¿Qué representa \(R\)?

El acuerdo entre el modelo y la regla verdadera en entradas contrafactuales. Para tareas pequeñas se puede evaluar la tabla de verdad completa; para tareas mayores se usa un conjunto separador fijado antes del entrenamiento. No es una medida general de inteligencia.

### 19. ¿Por qué usar señales temporales como confirmación?

Porque permiten comprobar si el perfil sobrevive fuera de MLP y reglas estáticas, manteniendo conocida la señal limpia, el ruido y la ley generadora. El banco separa tendencia, periodicidad, frecuencia variable, cambios de régimen y memoria no lineal antes de combinarlos. Si el resultado no se transporta a una red recurrente pequeña, esa diferencia limita el alcance del protocolo.

### 20. ¿Por qué no usar datos reales?

En datos reales no conocemos la regla verdadera, la cantidad de ruido ni el conjunto completo de intervenciones que distingue explicaciones. Serían útiles como comprobación externa después de validar las medidas, pero no para establecer qué significa cada una.

## Diseño experimental

### 21. ¿Cuál es la unidad estadística?

Una tarea generada de manera independiente, identificada por su regla y condiciones. Arquitecturas y tamaños son tratamientos pareados dentro de esa tarea. Las semillas son repeticiones del algoritmo y no aumentan artificialmente el número de tareas.

### 22. ¿Por qué al menos 30 tareas por estrato? ¿No es otro número arbitrario?

Es un piso operativo, no una justificación de potencia. El número final se calculará en el piloto con la varianza observada y el efecto mínimo relevante. Si 30 no alcanzan precisión, se aumentará antes de congelar el estudio; no después de ver el resultado confirmatorio.

### 23. ¿Cómo evita escoger la mejor definición después de ver resultados?

Piloto, ajuste, calibración y prueba usan identidades de regla separadas. El piloto fija definiciones; ajuste entrena el predictor; calibración fija umbrales; la prueba se abre una sola vez. Cambios posteriores se marcan como exploratorios.

### 24. ¿El early stopping sigue siendo parte central?

Es parte del protocolo, no la definición de información útil. La parada se elige en calibración. La información y el reconocimiento que deciden las hipótesis se calculan en datos intactos. Así se evita definir el punto con validación y luego “descubrir” que coincide con validación.

### 25. ¿Qué es \(N_{min}^{obs}\)?

El modelo más pequeño de una grilla predeclarada cuya cota inferior de desempeño supera el umbral y cuya calibración cumple el criterio fijado. Es un mínimo observado dentro de la grilla, no el menor modelo matemáticamente posible.

### 26. ¿Cómo podría ganar de manera engañosa el predictor propuesto?

Mirando casi toda la grilla antes de “predecir” o recomendando redes demasiado pequeñas. Todos los métodos tendrán el mismo sondeo de dos modelos pequeños y un presupuesto fijo de actualizaciones. La reducción de error tampoco basta: la tasa de subdimensionamiento es una puerta separada. El método no gana si ahorra parámetros a costa de fallar la tarea.

### 27. ¿Contra qué se compara?

Contra la cuenta de parámetros y dimensión de entrada, reglas de MacKay/Friedland donde sean aplicables, una curva de aprendizaje piloto y un predictor monotónico basado solo en tamaño de datos y ruido. Si una curva sencilla gana, la hipótesis principal falla.

### 28. ¿Cinco semillas son suficientes?

El número se fijará en el piloto según la inestabilidad del entrenamiento. En cualquier caso, las semillas se reportan como variación anidada; no se tratan como la población científica ni sustituyen tareas independientes.

### 29. ¿Qué ocurre con entrenamientos fallidos?

Se conservan y se clasifican. Un tamaño que no entrena bajo el presupuesto fijado no puede desaparecer del denominador. Se hará análisis de sensibilidad para distinguir insuficiencia de representación y dificultad de optimización, sin cambiar el veredicto principal después de verlo.

## Novedad, alcance y viabilidad

### 30. Arpit ya mostró que primero se aprenden patrones y luego ruido. ¿Qué queda para H2?

H2 no reclama descubrir esa secuencia por primera vez. Verifica que las tres medidas elegidas efectivamente la separan y que pueden alimentar H3. Si no la separan, el perfil queda invalidado antes de usarlo para dimensionar.

### 31. Friedland ya propuso dimensionar redes. ¿Qué queda para H3?

Precisamente la comparación. Friedland usa reglas derivadas de capacidad y una heurística sobre datos. H3 pregunta si observar memorización y ganancia predictiva en un sondeo acotado añade poder fuera de muestra. Si no añade, el perfil no sustituye ese antecedente.

### 32. NAS ya predice el desempeño sin entrenar. ¿Qué añade esta tesis?

La búsqueda de arquitecturas sin entrenamiento y la extrapolación de curvas son los vecinos directos. Por eso entran como comparadores con el mismo presupuesto. La afirmación nueva es más estrecha: que separar memorización, ganancia predictiva y recuperación de una regla mejora la estimación del **tamaño suficiente en tareas nuevas**. Si los proxies existentes igualan o superan el perfil, H3 falla.

### 33. ¿No está prometiendo dos tesis, una para MLP y otra para GRU?

No. La familia booleana con MLP contiene el estudio principal. El banco temporal con GRU es una confirmación recortable. El resultado doctoral no depende de que ambas familias produzcan la misma constante ni del éxito de la confirmación.

### 34. ¿Es viable en tres años?

Los modelos están acotados aproximadamente a un millón de parámetros y los datos son sintéticos. El piloto medirá costo antes de congelar la grilla. El alcance mínimo termina con la familia booleana. En la confirmación temporal se recortan primero los sistemas caóticos y después las combinaciones de componentes; las señales aisladas quedan como mínimo.

### 35. ¿Cuál sería un resultado negativo valioso?

Que las medidas en bits no mejoren una curva de aprendizaje barata, que dependan demasiado del optimizador o que no se transporten a señales temporales. Cada resultado establecería un límite concreto y evitaría presentar capacidad, memorización y ganancia predictiva como una receta de tamaño.

### 36. ¿Qué afirmación no debe hacerse durante la entrevista?

No debe decirse que se medirá inteligencia, que dos bits por peso valen para toda red, que la información útil ocupa una fracción literal de la capacidad, que el cociente señal-ruido mide memoria del modelo, que un compresor produce una cota inferior ni que se encontrará el tamaño mínimo universal. Ninguna de esas afirmaciones pertenece a la propuesta revisada.

## Señales temporales y terminología

### 37. ¿Por qué no basta una suma de senos con ruido gaussiano?

Porque es un control necesario, pero demasiado simple como confirmación completa. Una frecuencia fija puede resolverse con una representación compacta y ofrece atajos espectrales. El banco empieza allí y añade varias frecuencias, frecuencia variable, cambios de régimen y memoria no lineal. Cada componente se evalúa aislado antes de estudiar interacciones.

### 38. ¿Cómo evita fuga entre entrenamiento y prueba temporal?

La partición se hace por identidad de la ley generadora: combinaciones distintas de parámetros latentes pertenecen a piloto, ajuste, calibración y prueba. No se mezclan al azar puntos vecinos de una misma trayectoria. Además, toda ventana de pronóstico usa únicamente información disponible hasta su origen temporal.

### 39. ¿La fórmula \(\tfrac12\log_2(1+\mathrm{SNR})\) dice cuánta memoria usa la red?

No. Esa expresión corresponde a información o capacidad de un canal gaussiano bajo supuestos concretos de distribución, independencia y potencia. No describe automáticamente una señal sinusoidal correlacionada ni el contenido de los pesos. En esta tesis solo será una referencia cuando sus supuestos se satisfagan.

### 40. ¿Por qué no usar MNIST o el archivo UCR como fuente ideal?

Porque no conocemos exactamente su proceso generador, el nivel de ruido ni una regla verdadera que permita separar memoria y estructura. Pueden servir como comprobación externa, pero no como autoridad para medir capacidad en bits o aprovechamiento de información recuperable.

### 41. SynTSBench ya genera tendencia, estacionalidad y ruido. ¿Dónde está la novedad?

No está en generar esas señales. SynTSBench es un antecedente y comparador directo. La pregunta propia es si un perfil que separa capacidad de memorización, memoria específica, regla recuperada y ganancia predictiva permite anticipar el tamaño suficiente del modelo. Si no mejora los comparadores existentes, H3 falla.

### 42. ¿Por qué el documento dice “predicción” y no siempre “pronóstico”?

Porque no son sustitutos universales. Se usa **pronóstico** para estimar un valor futuro \(y_{t+h}\) con información disponible hasta \(t\). Se usa **predicción** en el sentido más amplio de aprendizaje automático: clasificación, predictor nulo, información predictiva o estimación del tamaño suficiente. En español ambos términos son válidos; la distinción evita llamar pronóstico a una clasificación sin eje futuro.

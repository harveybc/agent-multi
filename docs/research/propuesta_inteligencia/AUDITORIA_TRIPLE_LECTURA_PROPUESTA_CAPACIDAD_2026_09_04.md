# Auditoría de triple lectura de la propuesta sobre memorización y dimensionamiento

**Fecha:** 2026-09-04<br>
**Documento revisado:** `PROPUESTA_DOCTORAL_CAPACIDAD_OCUPACION_BITS.md` en `agent-multi@ac001fb8`<br>
**Veredicto de entrada:** `REVISE`<br>
**Alcance:** contenido científico, claridad para admisión y consistencia terminológica.

## Dictamen ejecutivo

La pregunta de fondo es válida: una cuenta de parámetros no permite decidir por sí sola cuál es el modelo más pequeño que alcanza un nivel de generalización solicitado. La propuesta también acierta al separar memorización de generalización y al usar procesos generadores conocidos.

La versión revisada, sin embargo, seguía presentando las unidades en bits como si fueran el objeto doctoral. El subtítulo “Estudio en bits”, la expresión “perfil en bits” y la repetición de esa unidad producían una promesa más fuerte que el método. Además, la ecuación de ganancia predictiva tenía el signo invertido, el tamaño “suficiente” carecía de un umbral explícito y el texto acumulaba siete símbolos antes de explicar la decisión práctica que debían apoyar.

La disposición es centrar la propuesta en **dimensionamiento temprano de redes bajo tareas controladas**. Las reducciones de pérdida se expresan en base dos cuando corresponde, pero compartir unidad no convierte capacidad, ajuste específico y generalización en partes de un mismo depósito.

## Tribunal 1: aprendizaje automático y estadística

### A1. La ganancia predictiva tenía el signo invertido

La fórmula publicada usaba `log(q0)-log(ptheta)` y luego afirmaba que un valor positivo representaba ahorro frente al predictor nulo. La reducción correcta de pérdida logarítmica es `log(ptheta)-log(q0)`, equivalente a `L0-Ltheta`.

**Disposición:** corregir la ecuación y exigir modelos probabilísticos evaluables bajo la misma regla de puntuación.

### A2. La memorización específica no descontaba la fluctuación fuera de muestra

Aplicar la parte positiva al contraste agregado evita el sesgo por seleccionar ejemplos favorables, pero todavía atribuía a memorización cualquier exceso finito sobre el generador. La comparación con prueba se mencionaba en prosa y no formaba parte del estimador.

**Disposición:** definir un contraste de diferencia entre entrenamiento y una muestra de referencia del mismo tamaño efectivo. Conservar su signo e interpretarlo como **exceso de ajuste específico de la muestra**, una medida operacional que debe validarse, no como lectura directa de los pesos.

### A3. “Tamaño suficiente” no estaba definido

No existe un tamaño suficiente sin un nivel de desempeño requerido. La expresión permitía cambiar el objetivo después de observar la curva.

**Disposición:** definir `N_min_obs(tau)` como el menor tamaño de un conjunto preestablecido cuya cota de desempeño alcanza una fracción `tau` de la mejora recuperable. Fijar el valor primario de `tau` antes de la evaluación final y publicar sensibilidad sobre valores vecinos. Las tareas que superen el tamaño máximo deben tratarse como censuradas, no desaparecer.

### A4. Algunos comparadores no resolvían la misma tarea

MacKay y Friedland son referencias sobre capacidad, no necesariamente predictores tempranos del tamaño requerido por una tarea nueva. Tratarlos como competidores directos confundía antecedentes teóricos con métodos de selección.

**Disposición:** reservarlos para el marco teórico. Los comparadores de H3 serán cuenta de parámetros y descriptores de tarea, extrapolación de curvas parciales y criterios sin entrenamiento, todos con igual presupuesto de observación.

### A5. Las hipótesis permitían decisiones posteriores al resultado

“Aproximadamente lineal”, “una región de entrenamiento” y “menor error” no fijaban precisión, ventana ni magnitud relevante. H2 podía buscar retrospectivamente el intervalo más favorable.

**Disposición:** usar puntos de medición fijados en calibración, diferencias pareadas por tarea, una mejora mínima definida con el piloto y corrección por comparaciones múltiples. Las semillas son repeticiones anidadas, no unidades independientes.

### A6. El costo de la calibración podía desaparecer de la comparación

Una calibración de capacidad por familia, precisión y entrenamiento puede ser más costosa que las corridas que pretende evitar. Presentar solo el ahorro posterior habría favorecido al método propuesto.

**Disposición:** contabilizar la calibración como inversión reutilizable, informar su costo amortizado y calcular cuántas tareas hacen falta para recuperar esa inversión.

### A7. H1 mezclaba una condición fija con una comparación entre arquitecturas

La primera frase fijaba arquitectura, precisión y protocolo, pero la segunda pedía que el mismo modelo explicara la arquitectura. Eran dos niveles de análisis sin distinguir.

**Disposición:** estimar la meseta dentro de cada combinación y evaluar después un modelo de escalamiento que represente diferencias entre combinaciones reservadas.

### A8. El estimador de H3 seguía siendo una caja vacía

“Un estimador sencillo” no definía qué devuelve ante cada tamaño ni cómo trata las tareas cuyo tamaño suficiente supera el máximo probado.

**Disposición:** exigir un estimador regularizado de la probabilidad de alcanzar la meta por tamaño candidato, incluida una salida explícita por encima del máximo evaluado.

### A9. La regla de puntuación podía confundirse con comparabilidad entre tareas

Usar pérdida logarítmica en clasificación y pronóstico no vuelve comparables sus valores brutos cuando cambian el espacio de salida y la distribución del objetivo.

**Disposición:** hacer comparaciones dentro de cada tarea y normalizar frente a sus propios predictores nulo y de referencia; no comparar pérdidas brutas entre familias.

## Tribunal 2: teoría de la información y capacidad neuronal

### B1. La unidad común no establece una partición común

Capacidad de memorizar etiquetas aleatorias, exceso de ajuste en entrenamiento y reducción de pérdida en prueba pueden expresarse con logaritmos en base dos. Eso no permite sumarlas ni interpretarlas como porcentajes de una misma memoria física.

**Disposición:** retirar “estudio en bits” y “perfil en bits” del título, resumen, pregunta y contribuciones. Explicar una sola vez por qué algunas medidas usan esa unidad.

### B2. La capacidad “usada” no es identificable con estos experimentos

El comportamiento de entrada y salida no revela de manera única cómo una red distribuye información entre sus parámetros. Dos parametrizaciones pueden implementar la misma función y una misma red puede reutilizar una representación en muchos casos.

**Disposición:** no proponer un escalar de ocupación. Reportar por separado capacidad empírica, exceso de ajuste y desempeño fuera de muestra.

### B3. Los resultados de MacKay y Morris estaban demasiado cerca de una regla general

El resultado de MacKay pertenece a una neurona binaria de umbral bajo entradas y etiquetas específicas. La cifra de Morris es una estimación empírica para transformadores tipo GPT y su protocolo de entrenamiento, no una constante de redes neuronales.

**Disposición:** describir ambos como antecedentes acotados y convertir la variación entre familias, precisión y entrenamiento en parte de la pregunta experimental.

### B4. Fano aporta una cota parcial, no el contenido del modelo

Identificar una regla latente a partir del comportamiento del modelo permite acotar la información demostrada sobre esa identidad. No cuantifica toda la información de los pesos ni entrega por sí sola el número de parámetros.

**Disposición:** mantener Fano como análisis teórico secundario y sacar su cota del conjunto primario que estima tamaño.

### B5. El canal gaussiano no mide memoria de una red

La capacidad de un canal con ruido gaussiano depende de supuestos de potencia, distribución y uso del canal. Una serie sintética con ruido conocido permite calcular una referencia probabilística, pero no convierte la fórmula de Shannon en ocupación de pesos.

**Disposición:** usar la verosimilitud condicional conocida del generador y reservar la capacidad de canal como antecedente, no como variable de respuesta.

## Tribunal 3: admisión doctoral interdisciplinaria

### C1. El título y el resumen sonaban promocionales

“Estudio en bits” y “perfil en bits” colocaban una unidad antes que el problema. Un lector podía interpretar que se prometía medir inteligencia o inventariar físicamente los pesos.

**Disposición:** titular desde los fenómenos y la decisión: memorización, generalización y dimensionamiento de redes.

### C2. Había lenguaje interno o poco natural

“Prueba intacta”, “grilla”, “oráculo”, “preflight”, etiquetas en mayúsculas y expresiones como “comparadores que puedan derrotar la propuesta” suenan a bitácora de laboratorio, no a documento de admisión.

**Disposición:** usar “conjunto de prueba reservado”, “conjunto de tamaños”, “predictor de referencia”, “estudio piloto” y “comparadores competitivos”.

### C3. La defensa ocupaba demasiado espacio

El texto repetía lo que cada medida no era. Las cautelas son necesarias, pero distribuidas en cada párrafo producen ansiedad argumentativa.

**Disposición:** explicar afirmativamente cada variable y concentrar sus límites en una subsección breve sobre alcance de interpretación.

### C4. El número de objetos ocultaba la tesis

Siete símbolos, cinco aportes y dos familias experimentales hacían parecer que había varios proyectos. La decisión central, estimar temprano el menor tamaño que cumple un objetivo, aparecía tarde.

**Disposición:** tres objetivos, tres hipótesis y tres aportes. La familia booleana es principal; las series controladas comprueban alcance y son recortables.

### C5. La pertinencia práctica era demasiado abstracta

“Comprender la generalización” no basta para justificar tres años. La propuesta debe mostrar la decisión que mejora y el costo que evita.

**Disposición:** presentar desde el inicio el costo de entrenar modelos demasiado pequeños o innecesariamente grandes y evaluar ahorro computacional como variable secundaria.

## Criterios aplicados a la nueva versión

1. El título no promete una medición de inteligencia ni una contabilidad física de bits.
2. La pregunta nombra una decisión, un momento de observación y comparadores de la misma clase.
3. Toda magnitud tiene población, denominador y conjunto de datos declarados.
4. `N_min_obs` depende explícitamente de un umbral de desempeño y conserva casos censurados.
5. La ecuación de ganancia predictiva tiene dirección correcta.
6. El exceso de ajuste descuenta un control fuera de muestra y conserva su signo.
7. Fano y la normalización frente al predictor de referencia son análisis secundarios.
8. Las hipótesis se deciden en tareas reservadas y no mediante búsqueda retrospectiva de ventanas.
9. La terminología de series temporales reserva “pronóstico” para valores futuros.
10. Las limitaciones se declaran una vez y no dominan la prosa.

**Veredicto de salida tras aplicar estas disposiciones:** `ACCEPT FOR AUTHOR REVIEW`.

# Auditoria critica de la propuesta UFM sobre capacidad en bits

**Fecha:** 2026-09-04  
**Documento evaluado:** `PROPUESTA_DOCTORAL_CAPACIDAD_OCUPACION_BITS.md`  
**Veredicto:** **REVISE CON REFORMULACION CONCEPTUAL MAYOR**

## Dictamen ejecutivo

La intuicion que origina la propuesta es valiosa: una cuenta de parametros no dice cuanto memoriza una red, cuanto aprende de una regla reutilizable ni cual es el menor tamano que basta para una tarea. Es un problema genuino de inteligencia artificial, admite experimentos controlados y conecta teoria de la informacion, aprendizaje y seleccion de modelos.

La version recibida, sin embargo, no debe enviarse. Sus dos formulas centrales no sostienen la interpretacion que se les atribuye:

1. `U*/C` divide informacion predictiva en datos estructurados por capacidad de memorizacion medida sobre ruido uniforme. Ambas cantidades estan expresadas en bits, pero no representan el mismo objeto. Compartir unidad no vuelve comparables dos magnitudes.
2. `Khat/alpha` no es una cota inferior valida del numero de parametros. Un compresor computable ofrece una **cota superior** de la complejidad de Kolmogorov; ademas, la capacidad observada por entrenamiento es una cota inferior de la capacidad alcanzable. Sustituir ambas en la direccion propuesta no puede producir una cota inferior.

El documento tambien hace circular la relacion entre `U*` y validacion, deja indefinido que cuenta como patron reconocido, atribuye novedad a fenomenos ya estudiados y contiene prosa interna impropia de una postulacion. No son detalles de estilo: afectan la pregunta, las hipotesis y el metodo.

La disposicion recomendada es conservar el problema y cambiar el contrato cientifico. La propuesta revisada debe separar:

- **capacidad empirica de memorizacion**, calibrada con datos aleatorios;
- **memorizacion especifica de la muestra**, estimada durante el entrenamiento;
- **informacion predictiva utilizable**, medida fuera de muestra respecto de un predictor que no observa la entrada;
- **reconocimiento de reglas conocidas**, evaluado con pruebas contrafactuales;
- **tamano suficiente observado**, definido por un criterio reproducible y predicho sin llamarlo cota universal.

## Hallazgos P0

### P0.1 — La ocupacion `U*/C` mezcla dos poblaciones de informacion

La capacidad de Morris se estima eliminando la generalizacion: los datos uniformes no contienen una regla reutilizable y la reduccion de perdida representa memorizacion de ejemplos. En cambio, `U*` se propone sobre datos estructurados y se interpreta como informacion util para generalizar. Un programa corto puede explicar un numero arbitrariamente grande de predicciones correctas; los bits ahorrados en un conjunto de prueba pueden superar la informacion almacenada en los pesos. Por eso ese ahorro predictivo no es una porcion literal del almacenamiento del modelo.

**Disposicion:** retirar “ocupacion util”. Si se conserva una normalizacion, solo puede comparar memorizacion especifica con capacidad de memorizacion bajo el mismo protocolo y debe llamarse indice empirico de carga, sin prometer que queda entre cero y uno. La informacion util se reporta en un eje separado.

### P0.2 — La supuesta cota minima invierte dos desigualdades

Para cualquier descripcion computable de un objeto `x`, la complejidad de Kolmogorov satisface `K(x) <= longitud(descripcion) + constante`. La longitud de un compresor no prueba que `K(x)` sea al menos ese valor. Por otro lado, Morris advierte que la capacidad alcanzada por descenso de gradiente es una cota inferior empirica: un entrenamiento mejor podria almacenar mas.

Por tanto, dividir una cota superior de descripcion por una cota inferior empirica de capacidad no produce una cota inferior del tamano de red.

**Disposicion:** eliminar la afirmacion. El menor tamano pasa a ser `N_min_obs`, el menor modelo de una grilla predeclarada que cumple un criterio fuera de muestra. El perfil informativo intenta **predecir** ese valor y se compara con reglas existentes. Como resultado teorico separado puede usarse Fano: identificar una regla latente entre `M` alternativas con error pequeno exige una cantidad minima de informacion sobre su identidad. Ese resultado no debe venderse como una cuenta exacta de parametros.

### P0.3 — H2 selecciona y explica con la misma variable

`U*` se toma en el paso elegido por perdida de validacion y luego se afirma que predice esa perdida mejor que otros puntos. Si `U` se calcula tambien como reduccion de NLL, la relacion es parcialmente definicional.

**Disposicion:** separar seleccion y evaluacion. El punto de parada se elige en calibracion; la informacion predictiva y el reconocimiento se estiman en tareas y ejemplos intactos. La prueba primaria no vuelve a usar las observaciones que eligieron el punto.

### P0.4 — “Patron”, `m` y longitud de programa no estan operacionalizados

Dos programas de distinta longitud pueden calcular la misma funcion. Una plantilla puede aparecer muchas veces, puede solaparse con otra o reconocerse por azar. Sin una familia generativa canonica, `m/n` no tiene unidad estable.

**Disposicion:** sustituir “programas cortos” por familias finitas con complejidad conocida por construccion. Se propone una familia primaria de reglas booleanas canonicas y una confirmacion secuencial con automatas finitos minimos. La regla latente se elige de un catalogo finito; su entropia es conocida. El reconocimiento se evalua en sondas contrafactuales no usadas para entrenar.

## Hallazgos P1

### P1.1 — La novedad estaba sobredimensionada

MacKay y Cover caracterizan la capacidad de una neurona de umbral bajo supuestos concretos. Friedland y colaboradores ya derivaron reglas lineales de capacidad y metodos practicos de dimensionamiento. Arpit y colaboradores mostraron que las redes aprenden patrones simples antes de memorizar ruido. Morris y colaboradores separaron formalmente generalizacion y memorizacion no intencional. Xu y colaboradores definieron informacion predictiva utilizable bajo una familia de predictores.

**Disposicion:** H1 se presenta como calibracion, no como aporte principal. La novedad candidata es evaluar conjuntamente esas magnitudes y comprobar si el perfil resultante anticipa el tamano suficiente en tareas nuevas.

### P1.2 — La unidad estadistica era incorrecta

`arquitectura x tamano x fuente` es una celda experimental, no una observacion independiente. Las semillas tampoco son la poblacion cientifica.

**Disposicion:** la unidad primaria es una tarea generada de manera independiente. Arquitecturas y tamanos son tratamientos pareados dentro de cada tarea; las semillas son repeticiones anidadas. Las familias y niveles de complejidad se separan en ajuste, calibracion y prueba intacta.

### P1.3 — Faltaban comparadores capaces de derrotar la propuesta

Comparar solo con `C` o con el ultimo paso haria facil ganar.

**Disposicion:** incluir al menos: cuenta de parametros, reglas de MacKay/Friedland aplicables, error de validacion de un piloto pequeno y una curva de escalamiento sin variables informativas. La metrica primaria es error absoluto en `log2(N_min_obs)`; la tasa de subdimensionamiento es una puerta de seguridad separada.

### P1.4 — El alcance entre MLP y recurrente no era coherente

Una MLP y una recurrente no consumen la misma estructura ni resuelven la misma familia de tareas. Empatarlas por parametros no crea una comparacion causal.

**Disposicion:** MLP para reglas booleanas como estudio principal; red recurrente pequena para automatas como confirmacion de transportabilidad del protocolo. No se comparan como si una fuera tratamiento de la otra.

### P1.5 — El texto de admision contenia lenguaje de bitacora

Referencias a Satoshi, Musashi, commits, otra postulacion, precio del programa, “un nulo es tesis”, “no es matricula” y largas listas de lo que el trabajo no es distraen y parecen defensivas.

**Disposicion:** mover toda genealogia y defensa hostil a documentos internos. El texto de admision solo explica el problema, lo que se propone, como puede fallar y por que importa.

### P1.6 — Faltaba el vecino mas cercano: prediccion de arquitecturas

La version recibida discute capacidad y compresion, pero omite la literatura que ya intenta anticipar el desempeno de redes con curvas parciales y proxies de costo cero. Domhan, Klein, Mellor y NAS-Bench-Suite-Zero hacen indefendible una novedad amplia basada en “predecir antes de entrenar”.

**Disposicion:** reposicionar H3 como una prueba incremental: determinar si la descomposicion informativa aporta valor sobre extrapolacion de curvas y proxies compatibles, bajo el mismo presupuesto. Esos metodos deben ser comparadores obligatorios, no menciones de contexto.

## Encaje con UFM

La UFM publica Ciencias de la Computacion como area del programa doctoral y describe la investigacion doctoral como original, objetiva, sistematica y verificable. La propuesta revisada encaja por su objeto: aprendizaje, memorizacion, generalizacion y seleccion del tamano de redes. No necesita injertar a Hayek, teoria de la conciencia ni economia para justificar la universidad. El vinculo con libertad y responsabilidad puede aparecer, si el formulario lo pide, en la eleccion de una investigacion reproducible y abierta; no debe convertirse en una hipotesis artificial.

## Pregunta y contribucion recomendadas

**Pregunta madre:**

> ¿Puede un perfil en bits que separa memorizacion especifica de la muestra y ganancia predictiva fuera de muestra anticipar, en tareas sinteticas de estructura conocida, el menor tamano de una red que generaliza, mejor que la cuenta de parametros y las reglas de capacidad existentes?

**Contribucion doctoral candidata:** un protocolo reproducible y una prueba fuera de muestra de su valor para dimensionar redes; una caracterizacion de donde deja de funcionar; y un resultado teorico de identificabilidad para tareas con regla latente finita.

Esta formulacion es menos grandiosa que “medir inteligencia”, pero mucho mas dificil de derribar. Si el predictor no supera los baselines, el resultado negativo sigue delimitando que las magnitudes en bits no bastan para dimensionar una red. Eso si es falsable.

## Condiciones para aprobar el nuevo borrador

1. No volver a llamar `U/C` ocupacion util.
2. No volver a llamar cota inferior a una longitud obtenida por compresion.
3. Definir una sola pregunta, tres objetivos y tres hipotesis como maximo.
4. Especificar familia generativa, unidad independiente, particiones y metrica primaria.
5. Citar la literatura que ya ocupa memorizacion, informacion utilizable y dimensionamiento.
6. Separar el resultado teorico de Fano de la estimacion empirica de parametros.
7. Retirar toda prosa interna y toda comparacion con otras postulaciones.
8. Usar referencias numeradas en el cuerpo y una bibliografia comprobable.

Con esas correcciones, el proyecto deja de depender de dos formulas defectuosas y conserva lo mejor de la idea original: tratar los bits como magnitudes experimentales, no como metafora.

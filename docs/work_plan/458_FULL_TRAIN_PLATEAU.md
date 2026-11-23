# 458. Entrenamiento completo y recuperacion en plateau

Requisitos y pruebas anteriores al lanzamiento:

|Requisito|Prueba de aceptacion|
|---|---|
|No estimar capacidad con una muestra corta|Cada epoca de cada rama usa una permutacion completa sin reemplazo de las 18.899 ventanas etiquetadas de 2020-2023; validacion 2024, 4.899 ventanas; sin TEST. No hay curriculum semestral en este contraste.|
|Transferencia real|Verificar manifiesto del bundle 61, columnas y arquitectura; copiar estrictamente solo encoder, comparar sus 208 salidas en todas las ventanas TRAIN en CPU/eval, diferencia maxima cero. Cabezal supervisado nuevo e identico al del control aleatorio de igual semilla.|
|No confundir mejoras|Factorial por semilla: encoder aleatorio/transferido y restauracion simple/doble trayectoria. Misma maquina dentro de semilla; mismos ejemplos, orden, mascara de dropout y optimizador para el mismo numero logico de epoca.|
|Contrato efectivo|AdamW LR 3e-4, wd 0,01; dropout 0,3 exclusivamente en los bloques del nucleo TCN (decision de este ensayo, no propiedad del actor SAC fuente). Lotes de 32 con ultimo parcial; no es una estimacion del batch optimo. El factorial no separa estos efectos comunes de los del volumen de datos.|
|Rollback|Gracia LR 20, paciencia 20, factor 1/5, piso 4,8e-7; campeon incluye pesos, momentos AdamW y tabla de calibracion. En un descenso ambos puntos arrancan al mismo LR reducido. Mantener aparte el campeon global.|
|Alternativa acotada|Solo al bajar LR: durante 5 epocas completas comparar A desde el campeon y B desde el ultimo estado antes del descenso, cada uno con sus propios momentos. Continuar desde el mejor estado realmente observado en esas cinco epocas (empate favorece A). La rama adicional no reinicia paciencia ni equivale a una muestra independiente.|
|Seleccion|Actividad >2 ventas/semana en validacion tiene prioridad; entre igualmente elegibles, mayor retorno neto con tolerancia 1e-8 puntos. Antes de encontrar candidato activo el mejor inactivo solo es un ancla tecnica. Nunca promocion automatica. Parada: desde epoca 100 contar 80 sin nuevo campeon; cap 2.000 epocas logicas.|
|Recuperacion e inspeccion|Guardar estado atomico por epoca, incluyendo ramas vivas, progreso de paciencia, LR, configuracion, semillas, metricas de cuentas TRAIN por ano y validacion, y elegibilidad. Registrar costo real: una epoca con dos ramas ejecuta dos pasadas.|

Analisis de desarrollo sobre validacion ya usada; el encoder fuente fue elegido en esa misma validacion. No es una prueba independiente de generalizacion ni un modelo para operar. Se evalua el especialista OPEN_SELL, no un actor SAC que seleccione tres acciones. Los resultados previos no se sobrescriben.

## Evidencia inicial y lectura final

Transferencia CPU/eval: hash del modelo fuente
`5b14e9f64f2fd296df2d0df6c6fdeffe0e16cb2e6e9715233caf400d44447c25`;
error maximo 0 sobre 18.899 x 208 salidas. La regularizacion 0,1 del bundle
fuente era de los criticos; no es el dropout del nuevo especialista. Se
comprobaron cuatro Dropout activos (dos por bloque de nucleo) de p=0,3.
Pruebas de independencia de momentos y pesos entre ramas, prioridad del
piso de actividad, tolerancia y frontera del scheduler: 2 tests aprobados.

Ejecucion: Gamma 5070 Ti (7), Gamma 5090 (19), Omega 4070 (43). Por GPU:
transferido/restauracion, aleatorio/restauracion, transferido/doble,
aleatorio/doble. Cada epoca de cada rama procesa 590 lotes de 32 y 1 de
19; el entrenamiento no usa datos sinteticos, estado de posicion como
entrada al encoder ni ventanas de 2024 para gradientes. La escala de
etiqueta se fija en todo TRAIN; no interpretar este ensayo como
entrenamiento incremental punto-en-el-tiempo.

Persistencia: `/var/tmp/role458-full-train/seed-*/<brazo>/` contiene
CONFIG, INIT, PROGRESS y checkpoint de recuperacion en la maquina ejecutora.
Al terminar: HISTORY, curva de retornos, DONE y campeon reevaluado
(retorno y tabla exactos). El coordinador recoge los 12 resultados y
los pesos ganadores; ante parada del worker reintenta como maximo dos
veces desde el checkpoint y deja evidencia de fallo repetido. No es un
servicio de arranque tras apagar la maquina. Reinvocar el coordinador
con el mismo directorio y codigo reanuda las colas.

Comparar por semilla e inicializacion: actividad y retorno de validacion,
Sharpe, drawdown, retorno de cada ano TRAIN, puntos de bifurcacion,
origen/epoca ganadora, pasos adicionales y tiempo. Sin bifurcacion los
brazos equivalentes no son replicas independientes; la doble trayectoria
tiene mas computo por evento y no prueba superioridad a igual presupuesto.
No decidir por una epoca temprana ni por el mejor de los doce aislado.

# Plan maestro CRISP-DM para el programa data-centric

**Fecha de corte:** 2026-09-10

**Alcance:** pronostico supervisado, seleccion de representaciones para RL y optimizacion con DOIN.
**Regla principal:** ningun modelo compensa una entrada mal definida. La unidad de trabajo inicial es la variable con su procedencia y disponibilidad temporal, no la arquitectura neuronal.

## 1. Que significa "cada variable de entrada"

No significa que toda columna almacenada deba llegar al modelo. Significa que toda variable candidata debe tener una identidad estable y pasar por una secuencia explicita de controles antes de competir.

El universo se construye con tres bancos separados:

1. **Banco financiero.** Variables crudas y derivadas del repositorio `financial-data`, incluidas las vistas materializadas que hoy alimentan a `predictor`, `gym-fx` y `agent-multi`.
2. **Banco publico de pronostico.** Series no financieras con procedencia y licencia comprobables. Sirve para decidir si un metodo se generaliza fuera del dominio en que fue concebido.
3. **Banco sintetico de mecanismos conocidos.** Senales cuyo componente limpio, ruido, eventos y cambios de regimen son conocidos por construccion. Sirve para calibrar diagnosticos; no reemplaza la confirmacion en datos publicos o financieros.

La misma columna repetida en dos cortes temporales no se cuenta como dos conceptos distintos. Se conserva una identidad logica de variable y se registran por separado sus apariciones fisicas, periodos y digests.

## 2. Inventario: reutilizar y reconciliar, no empezar de cero

El inventario historico de `financial-data` se conserva como antecedente, pero no es suficiente como autoridad actual: cubre cinco raices historicas y deja por fuera la mayor parte del arbol `features`.

La reconciliacion ejecutada en este ciclo encontro:

| Capa | Resultado actual | Interpretacion |
|---|---:|---|
| Manifiesto `financial-data/features` | 1,680 cortes fisicos presentes | 200 cortes de trading y 1,480 cortes por fuente/frecuencia |
| Columnas declaradas en esos cortes | 7,860 apariciones | No son 7,860 variables unicas; falta resolver equivalencias e identidad logica |
| Volumen fisico declarado | 14,436,534,039 bytes | Se verifico existencia y tamano, no se releyo todo el contenido |
| Perfil inmediato en `predictor` | 2 datasets, 26,541 filas, 99 columnas | ETHUSDT H4 y EURUSD 1h disponibles localmente |
| Estado de metadatos | incompleto | Faltan, segun fuente, unidades, licencia, contrato de disponibilidad o diccionario especifico |

La estrategia es incremental:

1. Reusar `MANIFEST.json`, archivos de procedencia, diccionarios y metadatos de entrenamiento.
2. Comparar digest, tamano y esquema contra el ultimo censo.
3. Reperfilar valores solo para archivos nuevos, modificados o seleccionados para una fase experimental.
4. Registrar de forma explicita `UNAVAILABLE` o `UNKNOWN`; nunca inferir unidades, licencia o disponibilidad a partir del nombre.
5. Emitir un recibo de cada reconciliacion y conservar el inventario anterior.

## 3. CRISP-DM aplicado al programa

### 3.1 Comprension del problema

Cada experimento debe declarar antes de mirar resultados:

- decision que pretende mejorar: pronostico, accion de trading, seleccion de representacion o asignacion de presupuesto;
- consumidor del resultado y horizonte de uso;
- costo de errores, abstenciones, latencia y computo;
- baselines sencillos que hacen inutil una mejora aparente;
- condiciones bajo las cuales un resultado negativo es informativo;
- dominio de uso y dominios en los que no se permite extrapolar.

Para pronostico, la pregunta no es solo "que modelo reduce el error", sino si una preparacion de datos reduce error fuera de muestra sin borrar extremos, anticipar el futuro o aumentar el costo de forma desproporcionada. Para RL, la pregunta es si una representacion mejora aprendizaje y decision bajo el mismo presupuesto, no si luce informativa en una prueba supervisada auxiliar.

### 3.2 Comprension de los datos

Por dataset y por variable se registran:

- fuente, licencia, unidad, frecuencia y zona horaria;
- tiempo de evento y tiempo de disponibilidad;
- cobertura, ausencias, duplicados, irregularidad temporal y valores no finitos;
- cardinalidad, constantes, rango y estadisticos robustos;
- cambios de distribucion y estabilidad por particion temporal;
- posible relacion con el objetivo, medida solo dentro de la particion permitida;
- digest de bytes, contrato de columnas y version de codigo usada para el perfil.

Las metricas de informacion y compresion son descriptores, no verdades sobre inteligencia o informacion libre de ruido. Se podran registrar longitud comprimida bajo un compresor fijado, entropia discreta bajo una cuantizacion declarada, complejidad de permutacion, rango efectivo correctamente definido, redundancia y estabilidad. Su utilidad se decide fuera de muestra.

### 3.3 Preparacion de datos

La preparacion ocurre en tres rejas distintas:

1. **Admisibilidad de variable cruda.** Procedencia, disponibilidad causal, unidad, calidad minima y rol conocidos.
2. **Elegibilidad de transformacion.** El operador respeta `fit/transform`, usa solo pasado, tiene paridad batch/incremental, costo medido y utilidad publica demostrada o estado experimental explicito.
3. **Seleccion de variables.** Solo entre salidas admisibles y elegibles, ajustada exclusivamente en entrenamiento y validada con divisiones temporales anidadas.

No se permite que un selector rescate una variable que fallo la primera reja ni que un gen de DOIN active una transformacion no licenciada.

### 3.4 Modelado

El modelado se ordena de barato a caro:

1. baselines de persistencia, estacionales, lineales y aleatorios;
2. seleccion univariada y multivariada con estabilidad temporal;
3. modelos supervisados pequenos en `predictor` para pronostico;
4. seleccion de representaciones y extractores bajo presupuesto;
5. optimizacion L2/DOIN;
6. RL y evaluacion economica solo cuando las rejas anteriores permiten atribuir el resultado.

`predictor` se reactiva como banco supervisado reproducible, no como generador de features ni como sistema live. Sus plugins permiten comparar el mismo contrato de datos entre familias de modelos, pero se deben corregir o aislar los preprocessors historicos que ajustan informacion fuera de entrenamiento.

### 3.5 Evaluacion

La unidad estadistica debe ser una tarea, serie, origen o ventana causal declarada; una semilla no se presenta como unidad independiente. Cada conclusion debe incluir:

- estimando y contraste primario;
- intervalo de incertidumbre y tratamiento de multiplicidad;
- costos completos de perfilado, ajuste, seleccion y evaluacion;
- controles de capacidad y dimension cuando una transformacion agrega columnas;
- desempeno en extremos, cambios de regimen y datos faltantes;
- abstencion o resultado inconcluso cuando la evidencia no identifica una mejora;
- replica publica no financiera antes de una afirmacion general.

### 3.6 Puesta en uso

En este programa, desplegar primero significa publicar un artefacto reproducible y consumible por los demas repositorios. No significa activar trading.

Un resultado apto para consumo incluye:

- manifest de variables y operadores elegibles;
- digests de datos, codigo, particiones y resultados;
- contrato de entrada/salida y disponibilidad temporal;
- recibo de revision independiente;
- adaptador determinista para `predictor`, `agent-multi` o `doin-plugins`;
- nueva validacion financiera y live antes de conceder autoridad operativa.

## 4. Orden cientifico y operativo

| Paso | Trabajo | Salida que abre el siguiente paso |
|---|---|---|
| I0 | Congelar decisiones, objetivos y roles temporales | Contrato CRISP-DM por experimento |
| I1 | Reconciliar los tres inventarios | Censo de datasets y variables con huecos nombrados |
| I2 | Perfilar calidad, temporalidad e informacion | Ledger por variable y por particion |
| I3 | Calibrar diagnosticos en sinteticamente conocido | Metricas interpretables y limites de deteccion |
| I4 | Evaluar transformaciones en banco publico | Lista revisada de operadores elegibles |
| I5 | Seleccionar variables con validacion temporal anidada | Manifest de variables congeladas y controles |
| I6 | Ejecutar pronostico supervisado en `predictor` | Evidencia de transferencia entre tareas y modelos |
| I7 | Construir grupos y representaciones modulares | Universo L1 congelado |
| I8 | Optimizar L2 con DOIN | Candidatos bajo presupuesto y procedencia completa |
| I9 | Evaluar RL y trading offline | Resultado mecanico/economico atribuible |
| I10 | Revalidar en dominio financiero y live | Elegibilidad operativa separada de la cientifica |

La seleccion de variables ocurre en **I5**, despues de caracterizar los datos y licenciar transformaciones. Puede haber un filtro mecanico previo para retirar columnas imposibles o causales invalidas, pero ese filtro no es seleccion por rendimiento.

## 5. Relacion con el trabajo ya realizado

- **T0/T1:** se conservan como evidencia de contrato causal y calibracion sintetica. No se reinterpretan retroactivamente.
- **T2:** su banco publico y sus resultados pertenecen a la reja I4. La adjudicacion existente se revisa con su identidad original; no se mezcla con el nuevo inventario.
- **M3/M4:** las mediciones de capacidad, complejidad y aprendizaje se incorporan como diagnosticos experimentales en I2-I3. Un descriptor que no demuestre utilidad incremental se retira.
- **B4 y campañas RL:** se adjudican bajo el contrato con que fueron ejecutadas. Los hallazgos nuevos solo gobiernan campañas sucesoras.
- **Planes de 13 pasos:** se conservan como catalogo de familias de operadores. Cada paso debe demostrar causalidad, utilidad y costo antes de entrar al flujo; el numero del paso no confiere elegibilidad.

## 6. OLAP: memoria del programa, no deposito indiscriminado

El cubo se amplia de forma aditiva con seis granos nuevos:

- dataset, serie de panel y variable;
- perfil de inventario por dataset;
- perfil por variable;
- metricas de informacion por particion;
- trayectoria por epoca o checkpoint;
- recibos de ingesta; los recibos de adjudicacion se agregan al conectar las campanas externas.

Los resultados viejos no se borran. Se marcan con cobertura de metadatos y, cuando no sea posible reconstruir un campo, queda `UNAVAILABLE`. La migracion correcta es: base desechable, validacion de esquema e ingesta, respaldo de la base real, migracion aditiva, backfill comprobable y recibo final. `reset_olap.py` no forma parte de este plan.

## 7. Entregables doctorales acumulativos

Cada tarea del programa deja material reutilizable para la propuesta de seleccion de representaciones:

- mapa de tareas, variables, operadores y costos;
- definicion reproducible de fidelidad como presupuesto parcial de entrenamiento o evaluacion;
- curvas parciales y decisiones de continuar, detener o abstenerse;
- comparadores ASHA/Hyperband y BOHB/SMAC bajo el mismo espacio;
- analisis de transferencia entre tareas y cambio temporal;
- resultados negativos y limites de identificabilidad;
- artefactos y protocolos suficientes para reproducir cada figura o tabla.

La tesis no gobierna el work plan completo: consume evidencia del programa. El programa puede explorar mas operadores y dominios, pero solo los resultados que respeten el protocolo doctoral entran como evidencia academica.

## 8. Estado de esta iteracion

Completado en `predictor`:

- registro inicial de los dos datasets locales actualmente utilizables;
- perfil reproducible de 99 columnas;
- reconciliacion estructural de las 1,680 vistas del manifiesto de `financial-data`;
- esquema OLAP aditivo para inventario, variables, particiones y epocas;
- correccion del bootstrap de una base OLAP nueva, que antes exigia un `ALTER TABLE` manual;
- pruebas focales y validacion end-to-end en PostgreSQL desechable.

Pendiente por diseno, no por olvido:

- resolver identidades logicas y metadatos de todo el banco financiero;
- completar contratos de disponibilidad y unidades donde la fuente no los declara;
- reconciliar el banco publico T2 y registrar el banco sintetico E0 bajo el mismo contrato;
- implementar la reja `PUBLICLY_ELIGIBLE` antes de los selectores y genes;
- migrar la base OLAP real solo despues de respaldo y revision del recibo de base desechable.

# Orden a Satoshi: censo completo, reja de variables y continuidad OLAP

**Autoridad:** decision del owner del 2026-09-10 de incorporar el programa de transformaciones al work plan data-centric, reactivar `predictor` para la evidencia supervisada y conservar entregables doctorales desde el inicio.

**Base revisada por Musashi:** `predictor@aef4dc22e7a6039eb349b75caf5e9b5823607d5e`.

**Documentos gobernantes:**

1. `docs/integracion_workplan_2026_09_10/00_PLAN_MAESTRO_CRISP_DM_DATA_CENTRIC.md`
2. `docs/integracion_workplan_2026_09_10/01_CONTRATO_INVENTARIO_Y_ELEGIBILIDAD.md`
3. `docs/integracion_workplan_2026_09_10/02_EVIDENCIA_IMPLEMENTACION_Y_OLAP.md`

## 0. Disposicion

Ejecutar P0-P6 de punta a punta en CPU. No abrir una campana GPU, no modificar resultados cerrados y no alterar procesos en curso. El trabajo es estructural, de inventario, contratos, pruebas y ETL. Donde una licencia, unidad o disponibilidad no pueda demostrarse, registrar el hueco y continuar con las demas unidades; no inferirlo ni bloquear el censo entero.

Congelar antes de editar los tips y estados de `financial-data`, `preprocessor`, `predictor`, `agent-multi`, `doin-domains` y `doin-plugins`. Usar ramas separadas por repositorio y publicar un mapa de commits.

## P0. Reproducir la base de Musashi

1. Reproducir los 8 tests focales del commit revisado.
2. Regenerar byte a byte `crispdm_dataset_inventory.v1.json`.
3. Regenerar la reconciliacion de 1,680 cortes y comprobar 0 ausentes.
4. Crear una base PostgreSQL desechable, cargar dos veces el inventario y comprobar que permanecen 2 datasets, 2 series agregadas, 99 variables y 1 recibo.
5. Verificar que el ETL historico funciona en una base nueva sin el `ALTER TABLE` manual.

Si una cifra cambia por bytes nuevos, detener solo la adjudicacion de esa cifra, publicar el delta por identidad y continuar con los demas P.

## P1. Censo incremental canonico de `financial-data`

Construir en `financial-data` un censo versionado que una, sin releer indiscriminadamente todo el lago:

- `features/MANIFEST.json`;
- metadatos de procedencia existentes;
- diccionarios de datos especificos y genericos;
- metadata de entrenamiento y contratos de disponibilidad existentes;
- digest fisico calculado para entradas nuevas, cambiadas o seleccionadas.

Entregar dos granos separados:

1. **Aparicion fisica:** archivo/corte, periodo, fuente, frecuencia, esquema, bytes y digest.
2. **Variable conceptual:** id estable, semantica, unidad, rol, disponibilidad, linaje y lista de apariciones.

Requisitos:

- no contar apariciones como variables unicas;
- no usar rutas fisicas como identidad logica;
- declarar equivalencias y conflictos en vez de resolverlos por nombre;
- separar `event_time` de `available_time`;
- listar de forma exacta los huecos de licencia, unidad, procedencia y disponibilidad;
- emitir `ADDED`, `UNCHANGED`, `CHANGED` y `MISSING` contra la version anterior;
- dejar un recibo reproducible y un resumen pequeno apto para Git; los artefactos grandes pueden quedar content-addressed fuera del repo.

El barrido de valores se hace en dos niveles: perfil completo de las vistas model-ready activas y perfil por muestreo/seleccion para el resto. Ningun muestreo se presenta como censo fisico completo.

## P2. Unificar los tres bancos sin mezclar autoridad

### P2.1 Banco publico

Reconciliar los manifests y resultados ya producidos por T2. No volver a descargar ni puntuar. Registrar por familia, panel y serie: fuente, licencia, periodo, frecuencia, longitud, missingness, particiones y estado de exposicion.

### P2.2 Banco sintetico

Registrar los generadores E0/T1/M4 con parametros de mecanismo conocido: senal limpia, tipo y nivel de ruido, eventos, cambios, retardos, seed tape y reconstruccion. Distinguir `CALIBRATION_ONLY` de cualquier banco confirmatorio.

### P2.3 Banco financiero

Usar el censo de P1. Las vistas de ETH H4 ya expuestas son desarrollo, nunca sustituto del banco publico.

Entregar un indice comun de datasets y variables, pero conservar la clase de autoridad de cada banco. Un join no promueve evidencia sintetica o financiera a confirmacion publica.

## P3. Implementar la reja `PUBLICLY_ELIGIBLE`

Crear un manifest revisable y un consumidor unico con estos bindings minimos:

- `variable_id` o `operator_id` y version;
- esquema de entrada/salida;
- unidad y disponibilidad temporal;
- `fit_scope` y politica de estado incremental;
- parametros;
- digests de datos, codigo, particiones y evidencia;
- costo medido;
- decision, alcance y razon del revisor.

Integrarlo en este orden:

1. `preprocessor`: antes de materializar una transformacion experimental.
2. `predictor`: antes de construir ventanas o ajustar un modelo.
3. `agent-multi`: antes de formar el universo de variables, grupos o extractores.
4. `doin-domains`: antes de publicar genes L2.
5. `doin-plugins`: verificar ids y digests; no seleccionar ni promover.

Regresiones obligatorias:

- manifest ausente, rancio o con digest distinto rehusa;
- variable no elegible no reaparece por fallback;
- grupo vacio no reactiva todas las variables;
- operador no elegible no entra por nombre de plugin;
- reemplazar evidencia manteniendo una etiqueta positiva rehusa;
- mismo manifest produce el mismo universo ordenado en proceso fresco.

## P4. Corregir fronteras de seleccion existentes

Auditar con pruebas ejecutables, no solo lectura:

- preprocessors aprendidos que usan CV no temporal;
- filtros por grupos que fallan abiertos;
- `predictor` phase 2.6, donde el scaler historico se ajusta por separado sobre train/validation/test;
- screens de `agent-multi` que calculan relevancia o redundancia fuera del split permitido;
- busquedas que reutilizan validacion repetidamente;
- genomas que incluyen todas las columnas numericas sin pasar la reja.

Corregir o etiquetar `LEGACY_NON_AUTHORITATIVE`. Una prueba debe cambiar la cola futura y demostrar que el estado/seleccion del pasado permanece identico.

## P5. Diseno de seleccion de variables

Sellar el diseno antes de puntuar. La unidad exterior sera tarea/origen/serie, no semilla. Comparadores minimos:

- todas las variables mecanicamente admisibles;
- filtro de estabilidad y redundancia;
- informacion mutua train-only;
- modelo lineal regularizado;
- selector actual de `agent-multi`, solo si supera P4;
- control aleatorio del mismo tamano.

El split exterior permanece intocado. Dentro de entrenamiento se ajustan imputacion, escalas, transformaciones y selector. Congelar:

- objetivo y baseline por tarea;
- numero de variables o regla de presupuesto;
- coste de seleccion;
- estabilidad entre origenes;
- no-inferioridad global y preservacion de extremos;
- tratamiento de multiplicidad;
- regla `INCONCLUSIVE` y criterio de retiro.

No ejecutar todavia una confirmacion grande. Si el banco y las unidades permiten un preflight mecanico pequeno, ejecutarlo sin conclusiones y detenerse ante la reja de revision.

## P6. Conectar campanas al OLAP

Extender de forma aditiva el cubo de `predictor@aef4dc2` para recibir un sobre comun de:

- run/campana/celda/candidato;
- dataset, variables y operadores consumidos;
- particiones y exposicion;
- presupuesto, costos y dispositivo;
- checkpoints/epocas cuando existan;
- estado terminal, metricas, incertidumbre y adjudicacion;
- digests de artefactos y recibo de ingesta.

Productores iniciales para backfill: T1/T2, B4 y M3/M4. Reglas:

- no borrar ni resetear el cubo;
- no inventar campos faltantes: usar `UNAVAILABLE`;
- separar resultados no gobernantes, mecanicos, development, calibration y confirmation;
- preservar toda identidad original;
- probar primero en PostgreSQL desechable;
- comparar conteos antes/despues;
- hacer el backfill real solo tras respaldo y recibo;
- demostrar idempotencia y rechazo de un artefacto mutado.

El estado actual del cubo debe publicarse honestamente: 39 experimentos historicos, ultima carga de performance 2026-04-16, mas el inventario CRISP-DM cargado el 2026-09-10. La orden no autoriza limpiar esos datos ni iniciar un cubo nuevo.

## 7. Entregables academicos obligatorios

Cada paquete debe dejar, desde ahora:

- pregunta y decision CRISP-DM que motiva la tarea;
- unidad estadistica y particiones;
- tabla de datasets/variables/operadores y costos;
- diagrama del flujo causal de datos;
- comparadores y regla de abstencion;
- ledger de intentos, incluidos negativos e inconclusos;
- tabla/figura reproducible con script y digests;
- texto de alcance que pueda reutilizarse en la propuesta doctoral sin lingo interno.

La palabra "fidelidad" debe acompañarse de su definicion operacional: cantidad parcial de entrenamiento o evaluacion usada para estimar, a menor costo, el resultado que se obtendria con el presupuesto completo.

## 8. Informe de retorno

Entregar un unico parte con:

1. tips PRE y POST por repositorio;
2. inventario y delta cuantificados;
3. huecos de metadata agrupados por responsable real;
4. pruebas adversariales y conteos tomados del tip final;
5. estado del manifest de elegibilidad;
6. estado de cada integracion consumidora;
7. esquema OLAP, backfill e idempotencia;
8. efectos ejecutados y no ejecutados;
9. preguntas que requieran exclusivamente al owner.

**Stop obligatorio:** no abrir GPU, confirmacion, live ni optimizacion DOIN a partir de estos artefactos hasta que Musashi revise el censo, la reja y el diseno de seleccion. El trabajo CPU, las pruebas, la migracion OLAP respaldada y el backfill verificable si estan autorizados.

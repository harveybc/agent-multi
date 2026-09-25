# Evidencia de implementacion y estado OLAP

**Fecha:** 2026-09-10

**Alcance:** primera base ejecutable del plan CRISP-DM. No incluye seleccion de variables ni una conclusion cientifica.

## 1. Artefactos derivados

| Artefacto | Resultado |
|---|---|
| Inventario local CRISP-DM | 2 datasets, 26,541 filas, 99 variables |
| Digest del inventario | `03be76a4c545eb60319053caf693560383b2c9f3119b2b2dcd70c19b7018a2a0` |
| Reconciliacion `financial-data/features` | 1,680 cortes presentes, 0 ausentes |
| Apariciones de columnas declaradas | 7,860, antes de resolver equivalencias conceptuales |
| Bytes fisicos declarados | 14,436,534,039 |
| Digest de reconciliacion | `833e6afebaa85bf2bf3af73a843963ae85651412f152ca113c6473436a7c9947` |

Los dos datasets locales quedaron `PROFILED_WITH_METADATA_GAPS`. Esto es un resultado correcto: el perfil fisico existe, pero no se fabricaron licencia, unidad o disponibilidad donde la fuente actual no las declara.

## 2. Pruebas de codigo

- compilacion de los modulos Python modificados: PASS;
- bateria focal de inventario y reconciliacion: 8 PASS;
- regeneracion del inventario local: igualdad byte a byte;
- inicializacion PostgreSQL en base desechable: PASS;
- ingesta repetida del mismo inventario: idempotente;
- ETL historico en base nueva, sin `ALTER TABLE` manual: 72 hechos escritos, 0 omitidos;
- conteo final desechable: 2 datasets, 2 series agregadas, 99 variables, 2 perfiles de dataset, 99 perfiles de variable, 1 recibo y 72 hechos historicos.

La primera ejecucion aislada detecto una referencia residual a `series_key` despues de separar series y variables. Se corrigio antes de migrar el cubo real y se repitio la prueba completa.

## 3. Cubo real

Hechos observados antes de la migracion:

- PostgreSQL y Metabase estaban ejecutandose;
- 39 experimentos historicos registrados;
- ultima carga de `fact_performance`: 2026-04-16 UTC.

Por tanto, el cubo estaba disponible pero no estaba recibiendo las campanas recientes. No se borro ni se reinterpreto ningun resultado historico.

Antes de escribir se creo un respaldo privado en formato PostgreSQL custom:

- nombre logico: `predictor_olap_pre_crispdm_20260910.dump`;
- tamano: 67,304 bytes;
- SHA-256: `a69ef96b7da8ce590b5b9c9d4c4692707f070f982746727140d1bbd01701c503`;
- modo: solo propietario.

La migracion real fue exclusivamente aditiva. Estado posterior:

| Tabla nueva | Filas |
|---|---:|
| `dim_dataset` | 2 |
| `dim_series` | 2 |
| `dim_variable` | 99 |
| `fact_dataset_inventory` | 2 |
| `fact_variable_profile` | 99 |
| `fact_ingestion_receipt` | 1 |

Recibo de ingesta:

- receipt: `4a1ba7b16a8a968c0f0e0145ce54752980af2e9d3e1276a6ad30068097cdd88c`;
- artifact: `03be76a4c545eb60319053caf693560383b2c9f3119b2b2dcd70c19b7018a2a0`;
- 2 datasets y 99 variables.

Los 39 experimentos y la fecha maxima de los hechos historicos permanecieron sin cambio observable. La base desechable se elimino al terminar.

## 4. Frontera honesta

El cubo ya recibe inventario y perfiles, pero aun no recibe automaticamente todas las campanas de `agent-multi`, `preprocessor`, `custodia` o DOIN. Esa conexion es la siguiente tarea: cada productor debe emitir un sobre comun de run/candidato/checkpoint y el ETL debe rehusar una ejecucion sin identidad, particiones, costos y recibo.

Hasta que esa conexion exista, no se afirmara que el OLAP contiene "todos los experimentos". Los resultados externos ya cerrados se incorporaran por backfill verificable, sin inventar campos ausentes y sin borrar la historia anterior.

# Contrato de inventario, caracterizacion y elegibilidad

## 1. Proposito

Este contrato evita tres errores recurrentes: llamar inventario a una lista de archivos, seleccionar variables antes de conocer su disponibilidad temporal y tratar una transformacion prometedora como si ya fuera apta para todos los modelos.

La autoridad se separa en cuatro objetos:

1. **Registro de dataset:** que existe, de donde viene y para que puede usarse.
2. **Perfil de variable:** que significa, cuando esta disponible y como se comporta.
3. **Registro de operador:** que transformacion se aplico, con que estado y sobre que particion se ajusto.
4. **Manifest de elegibilidad:** que variables y operadores superaron una revision concreta para un alcance concreto.

## 2. Campos minimos por dataset

| Grupo | Campos |
|---|---|
| Identidad | `dataset_id`, `dataset_family`, `source_id`, version, digest fisico |
| Uso | problema, objetivo permitido, consumidor, dominio, estado experimental |
| Tiempo | columna temporal, zona horaria, frecuencia declarada/observada, evento y disponibilidad |
| Procedencia | proveedor, URL/DOI cuando aplique, licencia, fecha de adquisicion, transformaciones previas |
| Esquema | columnas, tipos, roles, unidad de cada variable, politica de valores faltantes |
| Cobertura | inicio, fin, filas, series, duplicados, intervalos irregulares |
| Particiones | desarrollo, calibracion, confirmacion y embargo causal |
| Custodia | ruta logica, digest, codigo de perfilado y recibo de revision |

Un campo desconocido se registra como desconocido. No se completa por semejanza con otro dataset.

## 3. Campos minimos por variable

| Grupo | Campos |
|---|---|
| Identidad | `variable_id`, dataset, nombre fisico, nombre conceptual, version |
| Semantica | descripcion, unidad, tipo, rango fisico cuando exista |
| Tiempo | instante representado, instante disponible, rezago operativo |
| Rol | entrada, objetivo, control, identificador, metadata o excluida |
| Calidad | faltantes, no finitos, duplicacion, constantes, extremos y cobertura |
| Distribucion | cuantiles, escala robusta, estabilidad por particion y deriva |
| Informacion | entropia/compresion bajo contrato, redundancia, asociacion train-only |
| Dependencia | variables de origen, operador, parametros, fit scope y codigo |
| Decision | estado, razon, alcance, evidencia publica y revision |

Las metricas de asociacion con el objetivo se calculan dentro de entrenamiento o calibracion. Nunca se incorporan al perfil global usando confirmacion o test.

## 4. Estados

Los estados son explicitos y no se deducen de que un archivo exista:

- `DISCOVERED`: localizado, aun sin contrato suficiente.
- `PROFILED_WITH_METADATA_GAPS`: perfil fisico disponible, metadata incompleta.
- `MECHANICALLY_ADMISSIBLE`: esquema, tiempo, unidad y rol permiten evaluarlo.
- `LAB_CALIBRATED`: diagnostico u operador calibrado en datos de verdad conocida.
- `PUBLICLY_EVALUATED`: evaluado en banco publico bajo protocolo congelado.
- `PUBLICLY_ELIGIBLE`: revision externa permite su uso en el alcance nombrado.
- `DOMAIN_REVALIDATED`: la utilidad fue revalidada en un dominio de aplicacion.
- `LIVE_ELIGIBLE`: contrato operativo independiente satisfecho.
- `REJECTED`, `INCONCLUSIVE` o `UNAVAILABLE`: estado terminal o informativo con razon tipada.

Ningun estado superior se obtiene por nombre, etiqueta del productor o pertenencia a un repositorio.

## 5. Orden de las rejas

```text
dataset registrado
  -> variable mecanicamente admisible
  -> operador causal y reproducible
  -> calibracion sintetica
  -> utilidad publica
  -> revision externa
  -> seleccion train-only
  -> congelamiento del conjunto
  -> modelado / L2 / DOIN
  -> revalidacion financiera
  -> live, si corresponde
```

La seleccion de variables no decide procedencia, causalidad ni licencias. Recibe un universo que ya paso esas rejas.

## 6. Seleccion de variables

La seleccion debe:

- usar unicamente entrenamiento dentro de cada split exterior;
- ajustar escalas, imputacion y estadisticos dentro del split;
- comparar filtros simples, informacion mutua, estabilidad, redundancia y metodos embebidos;
- incluir un control de igual dimension cuando agregue variables;
- medir estabilidad de la seleccion entre origenes y cambios de regimen;
- congelar el conjunto antes de una unica confirmacion;
- registrar costo de seleccion y todos los intentos, no solo el ganador;
- impedir que la evaluacion repetida sobre test se convierta en optimizacion manual.

Los selectores historicos de `preprocessor`, `predictor` y `agent-multi` son comparadores o codigo a corregir; no constituyen por si mismos una autoridad cientifica.

## 7. Integracion por repositorio

| Repositorio | Responsabilidad |
|---|---|
| `financial-data` | fuente, procedencia, diccionarios, disponibilidad y manifest de vistas |
| `preprocessor` | contratos causales `fit/transform`, operadores y paridad batch/incremental |
| `predictor` | pronostico supervisado offline y comparacion entre familias de modelos |
| `agent-multi` | seleccion de representaciones/RL y consumo del manifest elegible |
| `doin-domains` | definicion del espacio L2 a partir de ids ya elegibles |
| `doin-plugins` | adaptadores deterministas y verificacion de identidades |
| OLAP | inventario, perfiles, costos, curvas, resultados y recibos |

`doin-domains` y `doin-plugins` no vuelven a decidir la elegibilidad: consumen y verifican la decision revisada.

## 8. Salidas versionadas de esta primera implementacion

- `examples/research/crispdm_dataset_registry.v1.json`: registro humano y ejecutable de datasets locales.
- `examples/research/crispdm_dataset_inventory.v1.json`: perfil derivado y ligado a bytes.
- `examples/research/financial_data_manifest_reconciliation.v1.json`: reconciliacion estructural del manifiesto de features.
- `olap/information_schema.py`: esquema e ingesta atomica del inventario.

El siguiente censo no reemplaza estos archivos en silencio: emite una version nueva y declara que entradas se conservaron, cambiaron, aparecieron o desaparecieron.

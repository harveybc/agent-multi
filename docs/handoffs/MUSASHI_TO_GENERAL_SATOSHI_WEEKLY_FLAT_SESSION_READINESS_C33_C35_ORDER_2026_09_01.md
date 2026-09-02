# Orden C33-C35: fuente inmutable y aislamiento de fixtures

**Fecha:** 2026-09-01  
**Base:** `gym-fx@bf561df`, evidencia `agent-multi@a772de9b`  
**Prioridad:** P0 acotada  
**Ambito:** CPU, offline, sin efectos

## PRE obligatorio

Congelar antes de editar:

1. constructor directo de `VerifiedSource` acepta componentes fabricados;
2. mutacion in-place de `source.frame` cambia el analisis sin cambiar source
   digest;
3. la puerta TEST_ONLY produce `collector_active=True` bajo schema productivo y
   sin marca de fixture.

## C33: fuente construible por un solo camino

La ruta de produccion debe aceptar bytes o un descriptor verificado y parsear
internamente. `VerifiedSource` no puede tener constructor publico por campos:

- usar `init=False` y fabrica privada validada, o eliminar el tipo de la API;
- campos de bytes, digest, roles e identidad no modificables;
- `source_digest` siempre recalculado, nunca aceptado;
- identidad logica y roles revalidados en el ultimo punto de uso.

El adversario que llama directamente al tipo debe rehusar antes de construir
paquete alguno.

## C34: contenido inmutable o reverificado

No almacenar un DataFrame mutable como autoridad reutilizable. Opciones
admisibles:

- parsear una copia fresca desde los bytes privados dentro de cada build; o
- conservar una representacion inmutable canonica y reconstruir una copia; o
- calcular un digest canonico completo del frame y volver a verificarlo justo
  antes de consumirlo, sin exponer la referencia interna.

Una propiedad publica `frame` solo puede devolver una copia. Mutar la copia no
debe alterar ejecuciones posteriores. El package digest debe cambiar cuando
cambian los bytes y permanecer determinista cuando no cambian.

## C35: fixtures fuera de la autoridad productiva

Mover loaders/builders de trust inyectable al arbol `tests/` o a un modulo que
no se distribuya. No basta el sufijo TEST_ONLY. Ninguna funcion embarcada de
produccion debe aceptar `ResolvedTrust` arbitrario y devolver un paquete
productivo autoritativo.

Para probar el futuro modo provisionado, el harness puede invocar una costura
privada bajo pytest, pero su resultado debe usar schema y estado de fixture
distintos. Ademas, todo consumidor futuro de readiness debe verificar antes de
usar suficiencia:

- schema productivo;
- `trust_manifest_digest` igual al pin esperado;
- `trust_status=PROVISIONED_AUTHORIZING`;
- ausencia de cualquier marca fixture/test;
- digest integral del paquete.

Añadir desde ahora el verificador de consumo, aunque la grilla siga bloqueada.

## Aceptacion

1. constructor directo con bytes/digest/frame rehusa;
2. digest no canonico e identidad de ruta rehusan tambien en build;
3. mutar una copia del frame no cambia el siguiente paquete;
4. no existe referencia mutable al frame interno;
5. busqueda estructural en codigo distribuido no encuentra builders/loaders
   productivos que acepten trust arbitrario;
6. un paquete fixture no pasa el verificador productivo;
7. paquete con manifest digest distinto, status no provisionado o schema de
   fixture rehusa en consumo;
8. la ruta productiva actual sigue no autoritativa frente a cualquier bundle;
9. las 30 pruebas historicas bajo fixture aislado siguen ejercitando pairing y
   firma, sin producir artefacto productivo;
10. focalizada, suite completa y sanitizacion verdes.

## Fronteras

Sin provisionar clave, tocar almacenamiento restringido, colector, live,
servicios, venue, GPU, SAC, grilla ni promocion. C33-C35 son exclusivamente
fronteras de objetos y consumo offline.

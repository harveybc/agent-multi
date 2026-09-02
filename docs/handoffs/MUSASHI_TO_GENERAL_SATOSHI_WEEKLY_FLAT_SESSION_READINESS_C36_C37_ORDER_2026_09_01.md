# Orden C36-C37: consumo derivado y una sola fabrica productiva

**Fecha:** 2026-09-01  
**Base:** `gym-fx@36caaef`, evidencia `agent-multi@4a58dc4d`  
**Prioridad:** P0 acotada  
**Ambito:** CPU, offline, sin efectos

## PRE obligatorio

Congelar antes de editar:

1. diccionario minimo auto-digerido pasa `verify_consumable_readiness`;
2. `_build_package` con `ResolvedTrust` construido por el llamador, digest
   igual al pin y `fixture=False` produce schema productivo autoritativo;
3. ese paquete pasa el verificador.

## C36: el consumidor no acepta afirmaciones autoemitidas

Mientras el manifiesto productivo este `NOT_PROVISIONED`, el verificador debe
rehusar todo paquete sin excepcion.

Cuando se provisione, el consumidor no puede limitarse al SHA del diccionario.
Debe cargar internamente el manifiesto pinneado, sin parametro de digest del
llamador, y verificar una de estas dos formas:

1. rederivar el paquete desde los bytes de source, export y receipt firmados; o
2. verificar una atestacion firmada del paquete completo bajo una identidad de
   evaluador fijada en el manifiesto revisado.

Hasta que una de esas rutas exista, conservar el verificador en estado
`NOT_PROVISIONED_NON_CONSUMABLE`.

Ademas, validar schema exacto y consistencia completa: autoridad no nula,
collector activo, estado suficiente, al menos 30 records soportados, digests de
pairing/source/export presentes y concordantes, `economic_grid_authorized`
siempre false.

## C37: una sola fabrica productiva real

Ninguna funcion del modulo distribuido puede aceptar `ResolvedTrust` arbitrario
y un selector capaz de emitir schema productivo. Refactorizar:

- la fabrica productiva carga por si misma el manifiesto pinneado;
- el nucleo comun no recibe `fixture=False` desde el llamador;
- la costura de tests construye exclusivamente schema fixture dentro de
  `tests/`, usando funciones puras que no pueden seleccionar schema productivo;
- `ResolvedTrust` no tiene constructor publico util para produccion, o la
  fabrica productiva ignora completamente cualquier instancia externa.

Una busqueda estructural debe probar que no queda funcion distribuida con la
combinacion `ResolvedTrust` + selector de fixture/schema.

## Aceptacion

1. los tres PRE rehusan;
2. paquete minimo o re-digerido no es consumible;
3. el verificador no acepta `expected_manifest_digest` del llamador;
4. bajo `NOT_PROVISIONED`, todo paquete rehusa consumo;
5. no existe costura distribuida que emita schema productivo con trust externo;
6. fixture conserva su schema y nunca pasa consumo;
7. produccion sigue generando diagnostico spot no autoritativo;
8. conteo de suite publicado coincide con la salida real;
9. focalizada, suite completa y sanitizacion verdes.

## Fronteras

Sin clave, almacenamiento restringido, colector, live, servicio, venue, GPU,
SAC, grilla ni promocion. Solo integridad de fabricas y consumo offline.

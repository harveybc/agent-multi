# Auditoria C28-C32: trust fail-closed y fuente unica

**Fecha:** 2026-09-01  
**Auditor:** General Musashi  
**Codigo revisado:** `gym-fx@bf561df`  
**Evidencia revisada:** `agent-multi@301f5f50`, saneo `a772de9b`  
**Veredicto:** `REVISE` acotado

## Resumen

La correccion principal de C28 es valida: la ruta productiva carga un
manifiesto committeado, verifica su digest contra el pin del codigo y el estado
actual `NOT_PROVISIONED_NON_AUTHORIZING` impide que cualquier bundle active el
colector. El digest de la orden aprobatoria tambien coincide con el documento
publicado. Las correcciones de as-of, intervalo `[close_at, reopen_at)` y
spacing booleano son ejecutantes.

Queda abierta la frontera C29. `VerifiedSource` es una dataclass frozen, pero su
constructor publico todavia acepta por separado bytes, digest y DataFrame, y el
DataFrame interno sigue siendo mutable. Ademas, el modulo productivo embarca
una segunda puerta que acepta un trust arbitrario y produce paquetes
autoritativos con el mismo schema que produccion, sin marca test-only.

## Reproduccion independiente

```text
30 pruebas focalizadas pasan
542 pruebas completas pasan

DIRECT_CONSTRUCTOR_ACCEPTED
  source.logical_id=<absolute-private-token>
  source_digest=d

MUTABLE_AFTER_HASH True 0 1
SHIPPED_TEST_DOOR True gymfx.wp4.session_readiness.v4 False
```

El segundo resultado significa: despues de calcular el hash, se altero el
timestamp de una fila dentro de `source.frame`; el source digest permanecio
igual, pero el inventario cambio de cero a un hueco observado.

## Hallazgos

### 1. Critico: el constructor directo conserva el bypass de C29

`VerifiedSource(...)` es publico y acepta exactamente los cinco componentes
que debian ser inseparables. El test publicado no prueba que eso rehuse: añade
un sexto argumento `extra=1`, por lo que cualquier dataclass ordinaria lanzaria
`TypeError`. Sin ese argumento, bytes no relacionados, digest no canonico,
identidad de ruta y DataFrame arbitrario son aceptados por produccion.

`_build_package()` tampoco recalcula el digest ni vuelve a validar la identidad
logica del objeto recibido.

### 2. Critico: frozen no hace inmutable un DataFrame

`@dataclass(frozen=True)` impide reasignar `source.frame`, pero no impide
`source.frame.loc[...] = ...`. La ruta de construccion reutiliza esa referencia
mutable. Por tanto, una fuente correctamente creada por `from_csv_bytes()` se
puede modificar despues del hash y antes del analisis.

La propiedad requerida es inmutabilidad de contenido o reverificacion en el
ultimo punto de uso, no inmutabilidad superficial del atributo.

### 3. Alto: la puerta TEST_ONLY produce autoridad indistinguible

`build_readiness_package_with_trust_TEST_ONLY()` y
`load_trust_manifest_TEST_ONLY()` viven en el modulo embarcado. El nombre es una
advertencia, no una frontera. La puerta acepta un `ResolvedTrust` construido por
el llamador y genera:

```text
collector_active=True
schema=gymfx.wp4.session_readiness.v4
sin marca test_only
```

Hoy ninguna grilla consume el paquete, por lo que no hay promocion efectiva.
Pero un futuro consumidor que confie solo en schema/veredicto no puede
distinguir fixture de autoridad productiva pinneada.

## Parte aceptada

- Manifiesto productivo pinneado y `NOT_PROVISIONED` fail-closed.
- Ningun parametro `trust` en `build_readiness_package()`.
- Firma Ed25519 y binding export-receipt.
- Contrato temporal as-of y rechazo de evidencia futura.
- Cierre `[close_at, reopen_at)`.
- Spacing no numerico o booleano queda `UNAVAILABLE`.
- Excepciones pasan por schema, rango e identidad comunes.
- Saneo del literal de topologia en la evidencia posterior.
- Suites de 30 y 542 reproducidas.

La conclusion spot negativa sigue aceptada. No se reabren C23-C28, C30-C32.

## Disposicion

Ejecutar C33-C35. Mantener el manifiesto productivo no provisionado. Sin
ceremonia de clave, live, grilla ni entrenamiento.

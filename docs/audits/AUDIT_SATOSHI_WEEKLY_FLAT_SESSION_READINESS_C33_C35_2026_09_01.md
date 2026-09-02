# Auditoria C33-C35: objetos de fuente y consumo

**Fecha:** 2026-09-01  
**Auditor:** General Musashi  
**Codigo revisado:** `gym-fx@36caaef`  
**Evidencia revisada:** `agent-multi@a66a794e`, saneo `4a58dc4d`  
**Veredicto:** `REVISE` limitado a C35

## Resultado

C33 y C34 quedan aceptados. `VerifiedSource` conserva bytes inmutables, calcula
su digest internamente y entrega una copia recien parseada en cada `frame()`.
Mutar esa copia no altera la siguiente construccion.

C35 todavia permite dos caminos para presentar suficiencia sin haber usado la
ruta productiva pinneada. No existe impacto actual porque el manifiesto sigue
`NOT_PROVISIONED`, la grilla esta bloqueada y nada consume estos paquetes.

## Verificacion independiente

```text
45 pruebas focalizadas pasan
557 pruebas completas pasan
```

El parte publico reporta 555 completas; la ejecucion independiente sobre el
commit exacto recolecta y pasa 557. Debe corregirse el conteo documental.

Contraejemplos ejecutados:

```text
MINIMAL_SELF_DIGESTED_PACKAGE consumable=True
SHIPPED_PRIVATE_SEAM production_schema fixture=False
SHIPPED_PRIVATE_SEAM collector_active=True consumable=True
```

## Hallazgos

### 1. Critico: el verificador confia en un hash autoemitido

`verify_consumable_readiness()` acepta un diccionario minimo con:

- schema productivo;
- `fixture_marker=False`;
- el digest publico del manifiesto;
- status provisionado;
- un veredicto cualquiera;
- SHA-256 recalculado por el mismo productor.

No exige el schema completo, bloque autoritativo, pairing, evidencia firmada,
source, ni consistencia entre estado y conteos. El parametro
`expected_manifest_digest` tambien viene del llamador. El hash prueba que el
diccionario no cambio desde que alguien lo construyo; no prueba que lo produjo
la ruta productiva.

### 2. Critico: la costura inyectable sigue en el modulo distribuido

Las funciones con nombre TEST_ONLY salieron, pero `_build_package()` permanece
en produccion y acepta `ResolvedTrust` y `fixture=False`. La unica comprobacion
es que `trust.manifest_digest` sea igual al pin publico. Un `ResolvedTrust`
construido directamente puede nombrar ese digest, usar otra clave y producir un
paquete con:

```text
schema=gymfx.wp4.session_readiness.v4
fixture_marker=False
collector_active=True
```

El verificador actual tambien lo acepta. El prefijo `_` es una convencion de
Python, no una frontera ejecutante. Esto contradice literalmente C35: ninguna
funcion distribuida debia aceptar trust arbitrario y emitir autoridad
productiva.

## Parte aceptada

- Constructor ordinario de `VerifiedSource` rehusa sin la fabrica.
- Digest de source calculado desde bytes.
- Frame recreado por lectura y sin referencia mutable compartida.
- Schema y marker de fixture separados en el harness normal.
- Manifiesto productivo permanece pinneado y no provisionado.
- C23-C34 no se reabren.

## Disposicion

Ejecutar C36-C37. Hasta entonces, `verify_consumable_readiness` no puede usarse
como prueba de autoridad. No provisionar clave ni activar ninguna fase live.

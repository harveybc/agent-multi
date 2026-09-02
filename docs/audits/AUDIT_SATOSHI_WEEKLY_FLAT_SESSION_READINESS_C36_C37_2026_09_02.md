# Auditoria C36-C37: consumo derivado y fabrica productiva

**Fecha:** 2026-09-02  
**Entrega:** `gym-fx@9353e0a`, evidencia `agent-multi@66fdbb7b`  
**Base:** `gym-fx@36caaef`  
**Veredicto:** **REVISE ACOTADO — C36/C37 CIERRAN LOS BYPASSES DE
AUTORIDAD; C38 BLOQUEA EL APROVISIONAMIENTO Y LA ACTIVACION**

## 1. Reproduccion independiente

Se inspecciono el diff completo `36caaef..9353e0a` y se ejecutaron las
baterias desde el checkout limpio de la entrega:

```text
WP4_TIER_A_ROOT=<raiz-logica> pytest tests/test_wp4_session_readiness.py -q
57 passed

WP4_TIER_A_ROOT=<raiz-logica> pytest -q
569 passed, 68 warnings
```

Sin `WP4_TIER_A_ROOT`, la prueba Tier-A rehusa como esta declarado: 56 pasan
y una falla cerrada. Ningun entrenamiento, servicio, venue, clave o artefacto
privado fue tocado.

Los tres PRE de la orden quedaron muertos en la frontera actual:

1. todo consumo rehusa mientras el manifiesto pinneado esta
   `NOT_PROVISIONED`;
2. la fabrica productiva no acepta trust externo y estampa el schema
   productivo en un solo sitio;
3. el nucleo comun no sella paquetes y la costura de tests solo puede sellar
   schema fixture.

La rederivacion desde source, export y receipt firmados es la direccion
correcta. Un SHA autoemitido ya no confiere autoridad.

## 2. C38-A (S2, alto): el schema anidado no es exacto y produce excepcion cruda

El verificador puro acepta como consistente un paquete re-digerido cuyo bloque
`authoritative` carece de `required_pre_bars`. El consumidor provisionado usa
despues acceso directo por clave y cae con `KeyError`:

```text
CONSISTENCY_ACCEPTED_MISSING_REQUIRED_PRE_BARS
CONSUMER_ACCESS KeyError 'required_pre_bars'
```

El contraejemplo usa el mismo cuerpo suficiente de 30 semanas de la bateria,
elimina solo esa clave y recalcula el digest canonico. No es un bypass de
autoridad bajo el manifiesto actual: hoy todo rehusa antes. Si se aprovisiona,
sin embargo, una entrada externa malformada puede derribar el consumidor en
vez de producir un refusal tipado. Ademas contradice la aceptacion de schema
exacto y consistencia completa.

La correccion debe validar, antes de cualquier acceso, las formas exactas de
los bloques anidados y cada valor que el consumidor reutiliza para rederivar.
No se acepta envolver el cuerpo entero en un `except Exception`: la frontera
debe nombrar el campo invalido.

## 3. C38-B (S2, alto): el camino provisionado completo nunca se ejecuto

La bateria prueba por separado:

- `_check_consumable_consistency` con un pin de fixture;
- `_require_rederivation_match` con dos diccionarios minimos; y
- el rechazo incondicional de `verify_consumable_readiness` bajo el pin real
  no provisionado.

No existe una prueba que, bajo un manifiesto provisionado aislado, ejecute la
misma secuencia que usara produccion:

```text
build_readiness_package -> verify_consumable_readiness -> consumable=True
```

Por tanto las dos mitades verdes no demuestran que la composicion vaya a
funcionar durante la ceremonia real. C38 debe ejercer esa ruta completa con
manifiesto y clave efimeros de test, sin introducir parametros publicos de
trust ni modificar el manifiesto embarcado.

## 4. Disposicion

C36 y C37 quedan **aceptados en mecanica**. La entrega completa permanece
`REVISE` hasta C38 porque no se debe descubrir el primer fallo del camino
provisionado durante una ventana que reinicia el EA.

Hasta la aceptacion de C38:

- no generar clave productiva;
- no cambiar el manifiesto pinneado;
- no activar el colector;
- no reiniciar EA, bridge o runner;
- no lanzar SAC, GPU ni grilla economica.

La posicion MT5 observada durante esta auditoria estaba plana y sin ordenes,
pero ese hecho operativo no sustituye la puerta de software.


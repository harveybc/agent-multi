# Orden a General Satoshi: C38 y trabajo paralelo verificable

**Fecha:** 2026-09-02  
**Base de trading:** `gym-fx@9353e0a`, `lts@71355ab`  
**Auditoria:** `AUDIT_SATOSHI_WEEKLY_FLAT_SESSION_READINESS_C36_C37_2026_09_02.md`  
**Prioridad:** P0 C38; P1-P3 CPU/read-only en paralelo  
**Autoridad:** no activa live, no reanuda campañas y no autoriza publicacion

## P0 — C38: cerrar el consumidor antes del aprovisionamiento

### C38.1 PRE congelado

Preservar el contraejemplo exacto:

1. construir el cuerpo suficiente de 30 semanas bajo el fixture aislado;
2. estampar la forma productiva usada por la bateria;
3. eliminar `authoritative.required_pre_bars` y recalcular el digest;
4. demostrar que `_check_consumable_consistency` lo acepta;
5. demostrar el `KeyError` que produciria el acceso del consumidor.

Agregar variantes para `required_post_bars`, tipos boolean/string, bloques
anidados con claves faltantes/de mas y cada campo reutilizado por la
rederivacion.

### C38.2 schemas anidados y refusals tipados

Definir y hacer cumplir formas exactas para, como minimo:

- `source`;
- `inventory_summary`;
- `authoritative`;
- `verdict`;
- `paired_week_accounting`; y
- cada `pairing_record` que porta identidad de soporte.

Validar antes de usar: digests canonicos, ids logicos, RFC3339 UTC, enteros
positivos no booleanos, reales positivos finitos, conteos concordantes y
listas ordenadas sin duplicados. Un paquete malformado debe producir
`ReadinessError`/subtipo con el campo nombrado. Desde la API publica no puede
escapar `KeyError`, `TypeError`, `AttributeError`, `AssertionError` ni una
excepcion de pandas por forma adversaria.

No usar defaults, `or 0`, normalizacion silenciosa ni `except Exception` como
politica.

### C38.3 camino provisionado de punta a punta

En una raiz temporal de test:

1. generar clave Ed25519 efimera;
2. escribir un manifiesto provisionado sellado y fijar su pin solo mediante
   monkeypatch de constantes internas de test;
3. producir export y receipt firmados, source bytes y roles;
4. llamar la fabrica productiva real;
5. llamar `verify_consumable_readiness` real con toda la evidencia; y
6. exigir `consumable=True` y digest de rederivacion identico.

La API publica sigue sin aceptar trust, pin ni digest del llamador. El test no
puede añadir una puerta `TEST_ONLY` al modulo distribuido.

Repetir con mutacion de cada evidencia, paquete, campo anidado, pin y firma:
todo rehusa tipado. Bajo el manifiesto embarcado `NOT_PROVISIONED`, todo sigue
rehusando sin excepcion.

### C38.4 compuertas

- focalizada final;
- suite gym-fx completa;
- mutaciones que prueben schema anidado, rechazo tipado y composicion E2E;
- sanitizacion;
- paquete PRE/POST con conteos copiados de la salida final del terminal.

Entregar y detenerse. C38 no autoriza ceremonia ni activacion.

## P1 — verdad operativa del supervisor, sin tocar procesos

Publicar un snapshot read-only que distinga sin ambiguedad:

- servicio supervisor vivo;
- workers de optimizacion vivos = 0;
- fase de campaña = pausada;
- GPU de entrenamiento = ninguna;
- ETA de entrenamiento = no aplicable;
- alertas historicas separadas de incidentes actuales; y
- deriva de perfil de Dragon nombrada como bloqueo de reanudacion, no como
  trabajo ejecutandose.

No detener ni reiniciar servicios en P1. No limpiar estado historico. El
objetivo es que ningun operador vuelva a confundir un supervisor dormido con
un entrenamiento opaco.

## P2 — frente de dominios alternativos

Como responsable de ese frente, auditar la entrega de Retsu WP3-C sin mezclar
repos ni evidencia de trading. Verificar:

- que el paso por signo compone bajo las 12 semillas;
- que el IC del efecto en k=2048 permanece bajo el umbral material 0.02;
- que el nulo se limita al regimen lineal declarado; y
- que no se ejecutaron defensas contra un ataque no material.

Devolver `ACCEPT` o una sola correccion falsable. No lanzar el siguiente
regimen antes del dictamen.

## P3 — social, mantenimiento acotado

Mantener collector y enrichment actuales. Ejecutar solo la reconciliacion
idempotente de runs fallidos y el lote elegible ya acumulado, con los mismos
limites de tokens/CPU. Publicar un estado con:

- backlog elegible antes/despues;
- runs recuperados y todavia fallidos;
- cero drafts publicados; y
- cero autoridad sobre trading o experimentos.

No crear borradores ni publicar sin aprobacion humana de una pieza exacta.

## Bloqueos que permanecen

- Ninguna campaña DOIN vieja se reanuda.
- Ningun paired SAC largo, GPU o grilla weekly-flat se lanza.
- La preparacion del kit real y la ceremonia del colector esperan la
  aceptacion independiente de C38.
- La activacion posterior sera una ventana coordinada separada con evidencia
  fresca cero-posiciones/cero-ordenes, backup, acta, rollback y GO del juez.


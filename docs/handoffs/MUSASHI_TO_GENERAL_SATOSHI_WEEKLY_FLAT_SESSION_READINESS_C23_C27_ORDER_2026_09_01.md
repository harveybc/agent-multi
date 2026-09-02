# Orden C23-C27: cerrar la autoridad de session readiness

**Fecha:** 2026-09-01  
**Base auditada:** `gym-fx@d08fa5f`, evidencia `agent-multi@aa4dad1b`  
**Prioridad:** P0  
**Ambito:** CPU, offline, sin efectos

## Objetivo

Hacer que `AUTHORITATIVE_SUPPORT_SUFFICIENT_FOR_CALIBRATION` solo pueda nacer
de evidencia MT5 autenticamente ligada, treinta intervalos con ventanas
pre/post locales y un paquete que conserve toda esa identidad. No repetir el
analisis spot ni ejecutar modelos.

## PRE obligatorio

Congelar antes de editar los cinco contraejemplos del dictamen:

1. exportacion y recibo fabricados con `seal()` producen treinta intervalos;
2. `authoritative={collector_active: true}` mas un conteo 30 produce suficiente;
3. cuatro barras remotas antes y cuatro despues soportan treinta intervalos;
4. dos ledgers autoritativos distintos con igual conteo producen igual digest;
5. ruta absoluta y digest no canonico entran al paquete.

La reproduccion debe invocar las APIs publicas ejecutantes, no replicas de su
logica.

## C23: raiz de confianza externa

Un SHA autoemitido no es autoridad. El consumidor debe exigir una raiz de
confianza que el mismo payload no pueda inventar. Son admisibles:

- firma desprendida sobre recibo y exportacion con clave publica fijada por la
  orden revisada, manteniendo la clave privada fuera de repos publicos; o
- verificacion descriptor-bound contra un ledger restringido aceptado, cuya
  identidad y digest esperado provengan de un manifiesto externo revisado.

No es admisible comparar dos campos que vienen del mismo bundle ni confiar en
etiquetas de exporter/parser escritas por el productor. Verificar de forma
ejecutante identidad de exporter, parser y codigo, activacion, acquisition
range, venue, cuenta y simbolo. Una exportacion sintetica autoconsistente debe
rehusar.

## C24: una sola ruta de derivacion

Eliminar de las APIs publicas toda entrada de `authoritative`, `paired`,
`collector_active` o conteos ya derivados. La construccion del paquete debe
recibir bytes de evidencia mas el contrato de confianza y derivar internamente:

1. activacion;
2. intervalos;
3. ventanas pareadas;
4. conteo;
5. veredicto.

El juez no debe aceptar diccionarios fabricados por el llamador. El conteo se
recalcula desde records verificados en el ultimo punto de uso; nunca se confia
en `supported_paired_weeks` como escalar transportado.

## C25: pairing local y causal

Por cada intervalo, ligar exactamente las barras requeridas:

- ventana pre-cierre adyacente sobre la grilla declarada;
- ausencia de barras dentro del cierre autoritativo;
- ventana post-reapertura adyacente sobre la misma grilla;
- timestamps, roles y digests de las filas seleccionadas;
- limites dentro del acquisition range de la evidencia.

Una barra remota no satisface una ventana local. Ocho barras no pueden
certificar treinta semanas. Cualquier contradiccion temporal rehusa en vez de
reducirse a un booleano `supported=False` silencioso.

## C26: paquete completamente ligado

El digest final debe cubrir, en forma canonica y ordenada:

- trust contract y prueba de activacion;
- exportacion de sesiones y rango adquirido;
- intervalos autoritativos;
- pairing ledger con timestamps y digests de barras por lado;
- ledger observado separado y explicitamente no autoritativo;
- source bytes/digest verificado, contrato de columnas, timezone, bar width y
  ventanas de metricas;
- veredicto derivado.

Cambiar cualquier intervalo o barra pareada debe cambiar el digest aunque el
conteo siga siendo 30. Nombrar por separado `authoritative_pairing_digest` y
`observed_gap_ledger_digest`.

## C27: schemas y fronteras estrictas

- Parsear bytes JSON rechazando claves duplicadas, constantes no finitas,
  campos desconocidos y faltantes.
- Exigir digests canonicos de 64 hex minusculas y timestamps RFC3339 UTC.
- Validar identidad y frescura/rango de activacion y adquisicion.
- Atar excepciones de operador a la misma venue/cuenta/simbolo y a una raiz de
  confianza; una excepcion autoconsistente no puede acuñar autoridad.
- Rechazar precios no positivos, bools, strings numericos y datos no finitos en
  roles economicos.
- Si se declara quote continuity, consumir timestamps de quote reales,
  ordenados y suficientes bajo un contrato de spacing ligado; en otro caso
  publicar `UNAVAILABLE`.
- Rechazar rutas absolutas, traversal, host u operador en identidades logicas;
  verificar source digest desde bytes/descriptor, no aceptarlo como prosa.

## Bateria de aceptacion

Ademas de todo lo existente:

1. los cinco PRE rehusan o dejan estado no autoritativo;
2. bundle auto-sellado con 30 intervalos no activa el colector;
3. sustitucion de trust root, exporter, parser, codigo, receipt o export rehusa;
4. JSON con claves duplicadas y campos extra/faltantes rehusa;
5. 29 ventanas locales validas da deficit 1; 30 da soporte suficiente;
6. retirar una barra local reduce soporte; agregar barras remotas no lo restaura;
7. una barra dentro del cierre rehusa;
8. mutar cualquiera de dos poblaciones autoritativas de igual cardinalidad
   cambia el digest;
9. quote evidence ausente sigue `UNAVAILABLE`; desordenada no puede dar true;
10. una raiz Tier-A falsa falla, nunca skip.

Ejecutar focalizada y suite completa con las dos raices logicas declaradas.
Publicar comandos, entorno, conteos exactos y digests del paquete.

## Fronteras

Sin preflight live, despliegue, reinicio, conexion a venue, comandos, posicion,
GPU, SAC, grilla economica ni promocion. La conclusion spot negativa permanece;
P0-P2 siguen `COORDINATED_WINDOW_REQUIRED` hasta que el kit real del operador
produzca un GO separado.

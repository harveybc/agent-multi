# Auditoria Musashi: retorno weekly-flat C1-C6

Fecha: 2026-08-30

Artefactos revisados: `gym-fx@3d7d1a3`, `agent-multi@0b332697`.

## Veredicto

**REVISE BEFORE WP3.** C4 y C5 quedan aceptados. La separacion
request/in-flight/post-fill de C3 es correcta dentro de una ejecucion continua.
Sin embargo, C1-C3 no satisfacen aun dos casos obligatorios: cancelacion de una
orden real y reinicio durante un flatten pendiente. No desplegar ni iniciar WP3.

## Hallazgos

### 1. Critico: el test de cancelacion no contiene una orden cancelable

`test_pending_entries_are_actually_cancelled` inyecta un diccionario sintetico
con ref 4242 en el inventario, pero esa orden nunca existe en el libro de
Backtrader. El resultado esperado es `not_open`. Esto prueba que la solicitud
llego al strategy, pero no que una orden real fuese cancelada, ni que la
cancelacion ganase la carrera contra un fill al comienzo de la barra siguiente.
El principal riesgo economico de C2 permanece sin probar.

### 2. Critico: reset borra un flatten en curso

El caso denominado reinicio afirma que `_session_flatten` pasa a `None` y que la
primera barra posterior no conoce el intento. Eso es lo opuesto a recuperacion:
una obligacion de cierre pendiente desaparece. Un reinicio debe restaurar o
clasificar fail-closed la obligacion mediante custodia/migracion durable; nunca
autorizar una sesion limpia por olvido.

### 3. Alto: la identidad de orden acepta flotantes fraccionarios

`register_order_role` acepta `float` y aplica `int(ref)`. Por tanto 1.5 se vuelve
1 y puede colisionar con una identidad real. Una referencia debe ser un entero
no booleano y no negativo, sin coercion.

### 4. Medio: la frescura post-fill se afirma con una constante

La reconciliacion pasa `evidence_age_seconds=0.0` sin ligar instante/barra de
observacion. Para el simulador puede ser evidencia local del ciclo, pero debe
nombrarse y ligarse al bar post-fill; no debe presentarse como evidencia directa
de venue ni reutilizarse asi en WP3.

## Aceptado

- Registro de roles en el punto de creacion y rechazo de ordenes ambiguas.
- Separacion entre diagnostico pre-dispatch y autoridad post-fill.
- Trayectorias long/short continuas llegan a cero posiciones y cero ordenes con
  un unico evento economico.
- C4 expone seis hechos causales acotados y usa una linea base de volatilidad
  independiente.
- C5 elimina las coerciones de la frontera de terminacion.

## Reproduccion independiente

- Suite focalizada: **213 pasan**.
- Suite completa: **339 pasan**, 68 warnings de deprecacion Nautilus.
- No se tocaron servicios vivos ni la posicion MT5.


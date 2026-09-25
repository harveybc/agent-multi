# Orden Musashi a Satoshi: cierre ejecutante weekly-flat G4/C5

Fecha: 2026-08-30

Prioridad unica: cerrar la semantica ejecutante semanal antes de WP3, WP4,
despliegue o entrenamiento largo. Reproducir cada hallazgo antes de editar.

## C1. Registro autoritativo de roles de orden

Registrar al crear cada orden su rol por identidad estable del broker:
`entry`, `protective_stop`, `protective_take_profit` o `close`. El inventario
debe derivar de ese registro. Prohibido inferir `reduce_only` solo por lado,
tamano o precio. Probar una entrada/reversion opuesta del mismo tamano que la
posicion y demostrar que no se confunde con proteccion.

## C2. Cancelacion ejecutante

En WIND_DOWN y FORCED_FLATTEN cancelar realmente todas las entradas pendientes
identificadas por el registro, esperar/observar su estado terminal y conservar
los brackets protectores hasta que la posicion este cerrada. Un rechazo, timeout
o inventario ambiguo produce incidente tipado y bloquea el exito.

## C3. Cierre y reconciliacion posteriores al fill

Separar `flatten_requested`, `flatten_in_flight` y `flatten_confirmed`. La
confirmacion solo puede ocurrir DESPUES de ejecutar CLOSE y obtener evidencia
fresca del mismo camino real con cero posiciones y cero ordenes. La comprobacion
pre-dispatch puede ser diagnostica, nunca autoridad de exito.

Pruebas obligatorias por `GymFxEnv` real:

- largo y corto;
- posicion con dos brackets;
- entrada pendiente independiente mas brackets;
- fill retrasado un bar;
- cierre rechazado;
- cancelacion rechazada;
- reinicio durante `in_flight`;
- resultado final exacto: cero posiciones, cero ordenes y un solo evento
  economico de cierre con costes.

## C4. Observacion de reapertura y metrica de volatilidad

Agregar a la observacion tipada, con espacio declarado coincidente: progreso de
barras cerradas, racha estable, spread relativo, gap en sigmas, ratio de
volatilidad y continuidad. Definir una linea base pasada de volatilidad realizada
independiente de la sigma del gap. Todo debe ser causal, finito, acotado y
fail-closed ante ausencia.

## C5. Fronteras estrictas de terminacion

Eliminar `or 0` y coerciones equivalentes de `_session_termination_record` y de
toda evidencia semanal autoritativa. Tipos ausentes o invalidos rehusan con
razon tipada.

## C6. Paquete de retorno

Entregar PRE/POST, ruta de llamada, inventario de ordenes antes/despues,
trayectoria completa request-to-confirmation y suites. No desplegar, no tocar la
posicion MT5 y no reclamar paridad live. La aceptacion independiente de este
paquete sera la unica compuerta para iniciar WP3.


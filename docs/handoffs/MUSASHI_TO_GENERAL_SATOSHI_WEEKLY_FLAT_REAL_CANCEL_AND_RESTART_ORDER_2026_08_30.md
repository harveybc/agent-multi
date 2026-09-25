# Orden Musashi: cancelacion real y recuperacion de flatten

Fecha: 2026-08-30

Correccion final y acotada antes de WP3. C4 y C5 no se reabren.

## R1. Referencias estrictas

`register_order_role` acepta solo enteros no booleanos y no negativos. Congelar
regresiones para 1.5, NaN, infinito, strings y booleanos; ninguna coercion.

## R2. Orden pendiente real

Construir por el camino ejecutante una orden de entrada Stop o Limit REAL,
registrada como `entry`, coexistiendo con una posicion y sus brackets. Debe
seguir visible en `broker.get_orders_open()` al entrar en WIND_DOWN.

Demostrar sobre el broker real del entorno:

1. la solicitud nombra su ref real;
2. el strategy llama a `cancel()` antes de aplicar la accion de esa barra;
3. el veredicto terminal es `Canceled/Cancelled/Expired`;
4. la orden no se llena;
5. los dos brackets permanecen vivos;
6. rechazo, fill-before-cancel y timeout nunca cuentan como exito.

El fixture sintetico `not_open` puede conservarse como prueba de incidente, pero
no como evidencia de cancelacion ejecutada.

## R3. Reinicio fail-closed

Un flatten `requested` o `in_flight` no puede desaparecer con `reset()` ni con
reinicio de proceso. Atarlo a una custodia durable o al mecanismo de migracion
ya aceptado. Tras reinicio solo son legales:

- reanudar/verificar el cierre pendiente; o
- estado tipado de recuperacion que bloquea nuevas entradas hasta evidencia
  fresca de cero posiciones y cero ordenes.

Probar reinicio de objeto y proceso. El test actual que exige “inherits nothing”
debe invertirse; hoy codifica el defecto.

## R4. Procedencia de la reconciliacion

Ligar la comprobacion post-fill a bar/instante monotono y clasificarla
explicitamente como evidencia del simulador. WP3 debera sustituirla por evidencia
directa tipada del venue; queda prohibido heredar el literal de edad cero.

## R5. Retorno y compuerta

Entregar PRE/POST y las trayectorias completas con inventario real. Sin
despliegue ni entrenamiento. La aceptacion independiente de R1-R4 desbloqueara
WP3; WP4 continua detras de la paridad live de WP3.


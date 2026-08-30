# Orden Musashi: fsync final y carreras reales

Fecha: 2026-08-30

Alcance cerrado: dos defectos mecanicos. No reabrir D1 ni la estrategia.

## F1. Reconocimiento durable

Corregir la frontera posterior a eliminar el marcador. Un fallo en CUALQUIER
fsync, incluido el fsync del directorio que hace durable la eliminacion, debe
dejar una indicacion durable que obligue a `read()` a rehusar. Soluciones
aceptables:

- intencion permanente + ACK append-only durable; o
- restaurar el marcador y fsync del padre antes de propagar el fallo.

Congelar el reproducer exacto: falla solo la tercera llamada a `_fsync_dir`;
despues del error, una instancia nueva debe rehusar la lectura.

Aplicar la misma propiedad a creacion y todas las transiciones.

## F2. Carreras concurrentes reales

Lanzar ambos procesos con `Popen`, detenerlos en una barrera comun y liberarlos
simultaneamente. Probar por separado:

- creacion de la misma obligacion;
- requested a in-flight;
- confirm contra fail;
- interrupt contra confirm.

Exactamente un ganador; el perdedor observa el estado del ganador y nunca lo
sobrescribe. Repetir tras instancia fresca.

## F3. Retorno

Entregar PRE/POST, reproducer y suite completa. Sin despliegue. La aceptacion
independiente de F1-F2 desbloqueara WP3 para implementacion, no para activacion
live; la activacion seguira requiriendo evidencia directa del venue.


# Auditoria Musashi: retorno weekly-flat D1-D5

Fecha: 2026-08-30

Artefactos: `gym-fx@1f55875`, `agent-multi@d63a5b1e`.

## Veredicto

**REVISE TWO MECHANICAL DEFECTS.** D1, integridad por digest, permisos,
symlinks, multiplicidad y reinicio entre procesos quedan aceptados. WP3 sigue
bloqueado por una frontera de fsync aun incorrecta y porque las supuestas
carreras de procesos son secuenciales.

## Hallazgos

### 1. Critico: fallo del fsync final deja una transicion reconocida

`_durable_write` elimina el marcador `.unacknowledged` y despues ejecuta el
ultimo fsync del directorio. Si ese fsync falla, la funcion lanza error pero el
marcador ya no existe y `read()` acepta el contenido nuevo.

Reproduccion independiente, fallando solo la tercera llamada a `_fsync_dir`:

- `mark_in_flight` lanza `OSError`;
- el marcador queda ausente;
- `read()` devuelve `flatten_in_flight` como valido.

Es el mismo tipo de reconocimiento parcial que el marcador pretendia impedir.
La eliminacion del marcador necesita protocolo append-only/ACK o restauracion
durable fail-closed si el fsync posterior falla.

### 2. Alto: las pruebas de carrera no son concurrentes

`test_two_processes_racing_each_transition` llama `_run(...)` para el primer
proceso y espera su terminacion; solo despues lanza el segundo. Prueba dos
procesos distintos, pero no una carrera. No demuestra eleccion atomica ante dos
transiciones simultaneas.

## Aceptado

- Un broker vacio de episodio nuevo ya no certifica el cierre antiguo.
- `interrupted_unresolved` es terminal sin claim de cierre y mantiene bloqueo.
- Digest verificado en lecturas y estados desconocidos fail-closed.
- Raiz 0700 y archivos 0600 bajo umask 000; symlinks rechazados.
- Varias obligaciones abiertas exigen disposicion del operador.
- Escritor y recuperador se ejecutaron en procesos realmente distintos.

## Verificacion

- Suite completa: **414 pasan**, 68 warnings Nautilus.
- Reproducer fsync-final: **defecto reproducido**.
- Servicios vivos y posicion MT5 intocados.


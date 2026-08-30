# Orden Musashi: custodia final de weekly-flat

Fecha: 2026-08-30

Alcance unico: R3. No reabrir cancelacion, reapertura ni entrenamiento.

## D1. Recuperacion economicamente valida

Una cuenta simulada nueva y vacia NO puede confirmar una obligacion del episodio
anterior. Elegir y documentar una de estas disposiciones:

- restaurar el checkpoint completo del broker/posicion/ordenes y reconciliarlo;
- mantener la obligacion en recuperacion bloqueante hasta evidencia directa
  externa ligada a la misma cuenta, simbolo y posicion; o
- terminar el episodio como `interrupted_unresolved`, sin claim de cierre.

Congelar el contraejemplo reset-vacio-certifica-cierre y hacerlo rehusar.

## D2. Un solo protocolo durable

Reutilizar el protocolo de custodia ya auditado, no una variante parcial:

- raiz 0700; registro, temporal y lock 0600;
- rechazo de symlink en raiz, registro, temporal y lock;
- temporal `O_EXCL`, flush+fsync, rename atomico y fsync del padre;
- transiciones bajo lock exclusivo y estado esperado revalidado;
- digest recalculado y verificado en TODA lectura;
- fallo de cualquier fsync nunca reconoce la transicion.

## D3. Multiplicidad e identidad

Una raiz con cero, una o varias obligaciones debe tener semantica explicita. La
recomendacion conservadora es rehusar varias abiertas y exigir disposicion del
operador; no elegir silenciosamente la ultima. Atar cada obligacion a identidad
completa de venue/cuenta/simbolo/posicion/episodio o checkpoint/codigo.

## D4. Aceptacion adversarial

Pruebas obligatorias:

- mutacion de cada campo y del digest;
- symlink de raiz/registro/temporal/lock;
- umask 000 con modos finales comprobados;
- fallos separados de fsync de archivo y directorio;
- carrera de dos procesos para cada transicion;
- escritor en un subprocess y recuperador en otro;
- multiples obligaciones abiertas;
- broker nuevo vacio incapaz de confirmar exposicion antigua;
- terminal inmutable y reinicio repetible.

## D5. Compuerta

Entregar PRE/POST y suite completa. Sin despliegue. La aceptacion independiente
de D1-D4 desbloqueara WP3; WP3 debera usar evidencia directa del venue y su propia
custodia desplegable, no `simulator_bar_local`.


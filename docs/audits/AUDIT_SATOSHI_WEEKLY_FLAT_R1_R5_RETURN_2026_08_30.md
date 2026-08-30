# Auditoria Musashi: retorno weekly-flat R1-R5

Fecha: 2026-08-30

Artefactos: `gym-fx@22cf7d8`, `agent-multi@2a056317`.

## Veredicto

**REVISE CUSTODY ONLY.** R1, R2 y la separacion de procedencia de R4 se
aceptan. La cancelacion de una Limit real esta demostrada. WP3 sigue bloqueado
porque la custodia R3 no cumple la durabilidad/integridad que declara y porque
la recuperacion puede confirmar una obligacion antigua usando el broker vacio de
un episodio simulado nuevo.

## Hallazgos

### 1. Critico: una cuenta simulada reinicializada certifica el cierre antiguo

`reset()` crea un bridge y broker nuevos con posicion cero, recupera la
obligacion anterior y luego permite confirmarla con ese cero/cero. Esa evidencia
no demuestra que la exposicion anterior fue cerrada; demuestra que el episodio
nuevo nacio vacio. La obligacion debe permanecer bloqueada hasta restaurar el
estado economico/checkpoint correspondiente o recibir evidencia externa directa
de la misma cuenta y posicion. En simulacion, abandonar el episodio debe quedar
como interrupcion no confirmada, no como cierre economico exitoso.

### 2. Critico: los registros no verifican su digest

`read()` y `outstanding()` hacen `json.loads` sin recalcular
`record_digest`. Reproduccion independiente: tras cambiar
`signed_exposure_at_request` de 1 a 999 sin actualizar el digest, `read()`
devuelve 999 como valido. Una alteracion puede cambiar identidad, estado o
evidencia de la obligacion sin rechazo.

### 3. Alto: las transiciones abiertas no son durables ni exclusivas

`_overwrite_open` usa `write_text` y `replace`: sin temporal `O_EXCL`, sin
`fchmod 0600`, sin fsync de archivo y directorio, sin verificacion bajo lock del
estado esperado. Dos procesos pueden competir durante `requested → in_flight`,
y un corte puede dejar una transicion no durable.

### 4. Alto: permisos y symlinks no cumplen el contrato declarado

La raiz se crea sin forzar 0700; se reprodujo modo **0775**. `read()` sigue un
symlink de registro y `outstanding()` lo recorre. El protocolo debe rehusar
symlinks en cada lectura/escritura y fijar raiz 0700, archivos y temporales 0600.

### 5. Alto: multiples obligaciones quedan en estado irresoluble

La recuperacion elige solo `outstanding[-1]`. Al confirmar esa obligacion, las
anteriores quedan en la lista de recuperacion, pero `_session_flatten` permanece
terminal y nunca cambia a la siguiente. Debe rehusar multiples obligaciones
pendientes para disposicion del operador o procesarlas secuencialmente con una
regla probada.

### 6. Medio: no hubo reinicio de proceso real

La prueba llamada process restart crea otro objeto Python dentro del mismo
proceso. Falta un `subprocess` independiente que escriba, termine y otro que
recupere usando la misma raiz.

## Aceptado

- Referencias de orden estrictamente enteras.
- Limit real visible en el libro, cancelada con veredicto `Canceled`, sin fill y
  preservando los brackets.
- Rechazo/fill/timeout/rol desconocido no cuentan como exito.
- Procedencia `simulator_bar_local`, `venue_direct=false`; WP3 no puede heredarla.

## Verificacion

- Suite completa: **370 pasan**, 68 warnings Nautilus.
- Reproducer de integridad: registro alterado aceptado.
- Reproducer de permisos: raiz 0775.
- Servicios vivos y posicion MT5 intocados.


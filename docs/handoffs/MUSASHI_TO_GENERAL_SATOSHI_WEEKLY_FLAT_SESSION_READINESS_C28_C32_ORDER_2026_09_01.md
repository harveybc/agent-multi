# Orden C28-C32: fijacion real, fuente unica y causalidad as-of

**Fecha:** 2026-09-01  
**Base:** `gym-fx@30db3f5`, evidencia `agent-multi@a14508a7`  
**Prioridad:** P0  
**Ambito:** CPU, offline, sin efectos

## Objetivo

Cerrar las cuatro sustituciones que aun permiten acuñar autoridad: trust root
elegido por el llamador, DataFrame independiente de los bytes, sesiones
futuras y barra situada exactamente en la frontera de cierre.

## PRE obligatorio

Congelar antes de editar:

1. clave atacante + bundle atacante + `TrustContract` atacante alcanza 30;
2. dos DataFrames distintos con los mismos `source_bytes` producen igual
   paquete;
3. intervalos posteriores a `now` cuentan como soporte;
4. barra exactamente en `close_at` es aceptada;
5. spacing booleano produce continuidad verdadera.

## C28: trust root fijado por la ruta ejecutante

La API de produccion no puede aceptar `TrustContract` del llamador. Debe cargar
un manifiesto de confianza revisado con:

- clave publica Ed25519;
- venue, cuenta y simbolo;
- digests canonicos de exporter, parser y codigo;
- politica de frescura;
- schema y digest propio;
- referencia y digest de la orden que lo aprobo.

El digest esperado del manifiesto debe estar fijado fuera del bundle de
evidencia y verificado antes de parsear sesiones. La inyeccion de trust usada
por tests debe vivir en una API explicitamente test-only que la ruta de
produccion no pueda llamar.

Como aun no existe una clave operativa revisada, materializar el manifiesto en
estado `NOT_PROVISIONED_NON_AUTHORIZING`. No inventar una clave productiva en
fixtures. La ceremonia posterior sera:

1. operador genera la clave privada en almacenamiento restringido;
2. solo la clave publica y los digests de codigo se publican para revision;
3. Musashi fija el manifiesto y su digest;
4. el exporter live firma con la privada sin exponerla.

Hasta ese acto, ningun bundle puede producir `collector_active=True`.

## C29: una sola poblacion de datos

Eliminar la pareja independiente `source_bytes` + `frame` de produccion. La
ruta debe abrir/recibir bytes una sola vez, calcular su digest y construir el
DataFrame mediante el parser fijado sobre esos mismos bytes. Alternativamente,
un objeto verificado debe conservar bytes, digest, identidad de parser y frame,
pero su constructor publico no puede aceptar esos componentes por separado.

Dos fuentes con filas distintas deben producir digests distintos incluso sin
huecos observados. Sustituir el DataFrame despues del hash debe ser imposible.

## C30: contrato temporal as-of

El export firmado debe declarar `exported_at` y `observed_through`. Verificar:

```text
activated_at <= observed_through <= exported_at <= evaluation_as_of
acquisition_start <= acquisition_end <= observed_through
cada reopen_at <= observed_through
cada barra usada <= evaluation_as_of
```

`evaluation_as_of` tampoco puede ser un reloj arbitrario que el mismo bundle
controle: debe provenir de la invocacion revisada y quedar ligado al paquete.
Toda evidencia futura rehusa; no se limita a quedar unsupported.

## C31: semantica exacta del intervalo

Declarar y probar el cierre como `[close_at, reopen_at)`. Rehusar cualquier
barra con `close_at <= timestamp < reopen_at`. La barra en `reopen_at` es la
primera barra post-reapertura; la ultima barra pre-cierre permanece en
`close_at - bar_width`.

Aplicar la misma semantica a sesiones y excepciones de operador.

## C32: fronteras restantes y evidencia publica

- Validar `expected_spacing_seconds` como real positivo, finito y no booleano.
- Validar todos los campos de `TrustContract` y del manifiesto; identidades de
  codigo/exporter/parser deben ser digests canonicos, no etiquetas libres.
- Validar cada intervalo de excepcion con schema exacto, `close < reopen`,
  acquisition range y binding completo.
- Eliminar rutas privadas literales de toda evidencia publica; conservar el
  contraejemplo con un token logico sanitizado.
- Escanear contenido y nombres de archivo bajo el paquete completo.

## Aceptacion

1. sustituir signer y trust juntos rehusa en la API de produccion;
2. manifiesto ausente, no provisionado, re-digerido o de otra orden rehusa;
3. no existe parametro publico de produccion para inyectar una clave;
4. bytes distintos o filas distintas cambian identidad; bytes y frame no se
   pueden desacoplar;
5. intervalos o barras posteriores al as-of rehusan;
6. barra en `close_at` rehusa y barra en `reopen_at` es post valida;
7. spacing bool/string/cero/NaN/inf rehusa;
8. excepciones invertidas, futuras, fuera de rango o de schema distinto
   rehusan;
9. 29 semanas historicas locales dan deficit 1 y 30 dan suficiente solo bajo
   un trust manifest productivo de fixture aislado;
10. suite focalizada, completa y escaneo de sanitizacion verdes.

## Fronteras

Sin generar clave privada productiva, tocar almacenamiento restringido,
activar colector, preflight live, servicio, venue, comandos, posiciones, GPU,
SAC ni grilla. La ceremonia de clave y P0-P2 requieren operador y autorizacion
separada; este ciclo solo endurece y deja fail-closed la implementacion.

# Auditoria C23-C27: autoridad de session readiness

**Fecha:** 2026-09-01  
**Auditor:** General Musashi  
**Codigo revisado:** `gym-fx@30db3f5`  
**Evidencia revisada:** `agent-multi@a14508a7`  
**Veredicto:** `REVISE`

## Resumen

La implementacion mejora de forma sustancial: la verificacion Ed25519 es
criptograficamente real, el schema firmado es cerrado, el pairing busca barras
adyacentes y el digest final incluye el bloque autoritativo. La conclusion
`SPOT_HISTORY_NOT_MT5_SESSION_AUTHORITY` tambien permanece correcta.

La raiz de confianza, sin embargo, no esta fijada por ninguna orden ejecutante.
El mismo llamador entrega la clave publica, las identidades y la politica de
frescura mediante `TrustContract`. Un atacante puede generar un par de claves,
firmar treinta intervalos y suministrar simultaneamente su propio contrato. El
paquete lo acepta como autoridad. La bateria solo prueba sustituir el firmante
sin sustituir el contrato; no prueba el ataque real, que sustituye ambos.

Ademas, los bytes declarados como fuente no generan el DataFrame consumido, se
aceptan intervalos futuros respecto al instante de evaluacion y una barra en la
frontera exacta `close_at` no se considera dentro del cierre.

## Verificacion independiente

En un worktree desprendido y limpio de `gym-fx@30db3f5`:

```text
pytest tests/test_wp4_session_readiness.py -q
30 passed in 0.86s

WP4_TIER_A_ROOT=<workspace> WP4_DATA_ROOT=<financial-data> pytest -q
542 passed in 12.75s
```

Los siguientes adversarios se ejecutaron sobre las APIs publicas:

```text
CALLER_SUPPLIED_TRUST AUTHORITATIVE_SUPPORT_SUFFICIENT_FOR_CALIBRATION 30
FUTURE_INTERVAL_ACCEPTED 2024-07-29T00:00:00Z > 2024-06-01
BAR_AT_CLOSE_BOUNDARY_ACCEPTED True
DIFFERENT_FRAMES_SAME_PACKAGE True True
BOOL_SPACING {'value': True, 'expected_spacing_seconds': True}
```

## Hallazgos

### 1. Critico: el llamador elige la raiz de confianza

`build_readiness_package(..., trust=TrustContract(...))` admite cualquier clave
publica. No existe manifiesto revisado cargado por la ruta ejecutante, digest de
orden fijado, allowlist de clave ni pin en codigo. Los tests generan una clave
aleatoria, firman con ella y pasan el contrato correspondiente al consumidor.

Esto demuestra firma, pero no confianza externa. El ataque que C23 pretendia
matar sigue vivo: el falsificador controla evidencia y trust root a la vez.
Las identidades `exp-1`, `par-1` y `code-1` de la bateria tampoco son digests de
artefactos ejecutantes.

La orden anterior exigia una clave publica fijada por una orden revisada. No se
publico una clave concreta ni se implemento una ceremonia operativa; ante esa
ausencia se debio dejar la autoridad deshabilitada, no convertir el contrato
del llamador en la orden.

### 2. Critico: `source_bytes` y `frame` son poblaciones independientes

El constructor recibe los bytes y el DataFrame por parametros separados. Solo
calcula SHA-256 de los primeros; nunca deriva ni verifica el segundo desde esos
bytes. Con los mismos bytes arbitrarios, un DataFrame de precios 1 y otro de
precios 999 producen exactamente el mismo paquete cuando no contienen huecos.

Por tanto, `source_digest_verified_from_bytes` es literalmente cierto pero
probatoriamente inutil: esos bytes no son la fuente demostrada de las filas que
alimentaron el analisis.

### 3. Critico: no existe frontera as-of

La bateria de treinta semanas usa `now=2024-06-01`, pero incluye intervalos y
barras hasta finales de julio. La adquisicion declarada termina en diciembre y
tambien se acepta. Se valida que el rango sea ordenado, pero no que haya sido
observado antes del instante de evaluacion.

El export no porta `exported_at` ni `observed_through`; por ello una firma valida
puede certificar sesiones futuras. Esto introduce look-ahead precisamente en
el ledger que se usaria para decidir suficiencia.

### 4. Alto: la frontera de cierre es abierta por el lado equivocado

El codigo busca barras con `timestamp > close_at` y `< reopen_at`. La semantica
del pairing usa como ultima barra previa `close_at - bar`, de modo que el cierre
fisico es `[close_at, reopen_at)`. Una barra exactamente en `close_at` debe ser
contradiccion; hoy se acepta y el intervalo queda soportado.

### 5. Alto: quedan fronteras estrictas y sanitizacion pendientes

- `expected_spacing_seconds=True` autoriza quote continuity.
- `TrustContract` no valida identidades como digests canonicos ni edad maxima
  como real positivo no booleano.
- Las excepciones de operador no validan por registro campos exactos, orden ni
  rango de adquisicion antes de construir intervalos.
- La evidencia publica C23-C27 reproduce una ruta privada absoluta de ejemplo,
  contradiciendo su afirmacion de cero topologia.

## Parte aceptada

- La verificacion Ed25519 sobre el cuerpo canonico funciona cuando la clave ya
  es confiable.
- Export y receipt quedan ligados por digest.
- Claves JSON duplicadas, campos extra y constantes no finitas rehusan.
- Las ventanas locales ordinarias y sus row digests estan implementados.
- El paquete separa `authoritative_pairing_digest` de
  `observed_gap_ledger_digest`.
- La conclusion spot negativa y el deficit diagnostico de 30 se mantienen.
- La suite completa reproduce 542 verdes con las raices Tier-A declaradas.

Nada de esto autoriza calibracion, grilla economica, entrenamiento, colector,
despliegue ni accion de venue.

## Disposicion

Ejecutar C28-C32. La implementacion offline puede avanzar inmediatamente, pero
la autoridad real debe permanecer deshabilitada hasta una ceremonia separada de
provision de clave por el operador. P0-P2 siguen
`COORDINATED_WINDOW_REQUIRED`.

# Auditoria Musashi: retorno weekly-flat WP3

Fecha: 2026-08-30

Artefactos: `lts@eb47ee0`, `agent-multi@7f83de29`.

## Veredicto

**REVISE BEFORE WP3 ACCEPTANCE.** La arquitectura read-only y la traduccion sin
segunda maquina de estados son buenas, pero hay tres bypasses de autoridad, una
costura MT5 rota y WP3.3 ausente. WP4 y toda activacion siguen bloqueados.

## Hallazgos

### 1. Critico: un payload viejo puede reenvolverse como fresco

`VenueDirectEvidence.verify` calcula edad con `self.observed_at`, suministrado
fuera de los bytes, mientras los parsers extraen otro `observed_at` interno sin
atar ambos. Reproducer: payload interno de 2020 + timestamp externo de 2026 fue
aceptado bajo una politica de 120 segundos. La frescura debe derivar unicamente
del timestamp firmado/publicado dentro de los bytes o de metadata de transporte
autenticada e inseparable del payload.

### 2. Critico: cuenta y simbolo externos no se atan a los hechos

El sobre puede declarar USDCAD aunque las posiciones internas sean BTCUSD; la
verificacion solo contrasta el sobre con la politica. `build_exposure_facts`
suma todas las posiciones y ordenes. Reproducer: posicion BTCUSD aceptada como
exposicion USDCAD. En account-session, el fingerprint derivado/interno tampoco
se compara con el fingerprint del sobre.

Cada parser debe devolver y ligar identidad interna; toda fila debe coincidir
con venue/cuenta/simbolo de la politica. Mezcla de simbolos rehusa, nunca filtra
ni suma silenciosamente.

### 3. Alto: coercion previa anula los validadores estrictos

Los parsers llaman `float(value)` antes de `require_real`; `True` se convierte
en 1.0 y strings se aceptan indiscriminadamente. Reproducer MT5:
`volume: true` fue aceptado como una posicion de 1.0. MT5 debe exigir numeros
JSON no booleanos. Alpaca puede admitir sus strings numericos documentados, pero
mediante parser lexical explicito, no `float()` generico.

### 4. Alto: la identidad revisada de autoridad es opcional

`load_authority(... expected_code_identity=None)` acepta cualquier checkout y
el dry-run lo invoca sin digest esperado. Esto contradice la afirmacion de que
rehusa una autoridad distinta. El digest revisado debe ser obligatorio y parte
del config/manifiesto; ausencia rehusa.

### 5. Alto: WP3.3 no esta implementado

No existe custodia live en el commit LTS ni adaptador que ate obligaciones a
evidencia directa por venue/cuenta/simbolo. Cargar `flatten_custody.py` como
modulo de autoridad no materializa ni usa una custodia desplegable. WP3.3 sigue
pendiente.

### 6. Alto: el payload MT5 real no atraviesa el bridge

El EA emite `time_open_unix`, el parser WP3 lo exige y el runner lo consume,
pero `PositionSnapshot(extra='forbid')` no declara el campo. El endpoint rechaza
el snapshot completo con 422. Mientras esto siga asi no existe evidencia MT5
ejecutante para WP3. Corregir contrato y probar EA bytes → FastAPI → store →
parser, sin desplegar.

### 7. Medio: falta evidencia real de Alpaca

El unico dry-run publicado es MT5; las estructuras Alpaca son fixtures. Se
necesita al menos una captura read-only sanitizada del camino real para account,
positions, clock y open orders, con ausencias tipadas cuando no haya brackets.

## Aceptado

- Modulos WP3 no contienen cliente venue ni credenciales y el dry-run no escribe.
- Parser desde bytes detecta duplicados y constantes JSON no finitas.
- Roles se derivan de semantica declarada, no geometria.
- Adaptador carga la autoridad compartida y no define otra maquina de estados.
- Watchdog solo suprime stale-bar durante cierre esperado.
- Suites focalizadas: **100 pasan**.


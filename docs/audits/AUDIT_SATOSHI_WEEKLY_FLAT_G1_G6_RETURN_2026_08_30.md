# Auditoria Musashi: retorno weekly-flat G1-G6

Fecha: 2026-08-30

Artefactos revisados:

- `gym-fx@23e36d1`
- `agent-multi@00932865`

## Veredicto

**REVISE BEFORE DISPATCH.** G1, G3, G5 y las reparaciones fundacionales G6 se
aceptan. G2 se acepta como mecanismo causal fail-closed, pero necesita completar
su superficie observable y corregir la referencia de volatilidad. G4 y C5 se
rechazan: el camino real todavia no cancela entradas pendientes ni confirma el
cierre despues de ejecutarlo. WP3, WP4, despliegue y computo largo permanecen
bloqueados.

## Hallazgos

### 1. Critico: la cancelacion de entradas pendientes no se ejecuta

`overlay_action` produce `cancel_pending=true`, pero
`GymFxEnv._apply_session_exposure_overlay` solo publica ese booleano en `info`.
No llama al broker para cancelar las referencias de entrada. Por tanto una
entrada pendiente puede sobrevivir al wind-down o al forced-flatten y llenarse
durante la ventana prohibida. Los brackets protectores deben permanecer vivos.

### 2. Critico: la reconciliacion ocurre antes del cierre

La compuerta se evalua dentro de `_apply_session_exposure_overlay`, antes de que
el comando CLOSE llegue al plugin de estrategia. Usa la exposicion y las ordenes
anteriores al cierre, y no existe una segunda comprobacion posterior al fill.
Reproduccion independiente: la trayectoria real solo emitio
`flat_confirmed=false`, `positions=1`, `orders=2`; nunca aparecio una confirmacion
posterior. Eso no prueba ni completa un flatten.

### 3. Alto: el rol de las ordenes todavia se infiere por geometria

Backtrader no conserva `.parent` en las patas vivas. La heuristica actual llama
`reduce_only` a una orden opuesta cuyo tamano no supera la posicion. Una orden
independiente de reversion del mismo tamano puede parecer un bracket protector.
El rol debe registrarse por `order.ref` al crear la orden y consumirse desde ese
registro; no debe reconstruirse desde lado y tamano.

### 4. Alto: G2 no expone al modelo la evidencia que gobierna la reapertura

El entorno calcula spread, gap, volatilidad y continuidad, pero la observacion
solo contiene seis campos de estado/tiempo. El agente no puede distinguir por
que sigue bloqueado ni aprender alrededor del regimen de reapertura. Ademas, el
ratio de volatilidad usa como denominador la sigma de retornos elegida para el
gap, no una linea base de volatilidad realizada definida por su propia ventana.

### 5. Medio: la terminacion aun convierte evidencia ausente a cero

`_session_termination_record` conserva `float(... or 0.0)` e `int(... or 0)` en
hechos que alimentan la migracion. Esto contradice la frontera tipada del resto
del trabajo: ausente, booleano, NaN o tipo incorrecto deben rehusar, no convertirse
en una cuenta plana.

## Lo aceptado

- G1 separa correctamente salida del modelo, comando discreto y exposicion; una
  accion bloqueada envia HOLD.
- G3 usa huecos temporales reales y rehusa barras dentro del cierre.
- G5 ya prueba exposicion terminal no vacua.
- G6 repara calendario y espacio de observacion; Nautilus publica
  `position_units`, rehusa H1/H4 y mantiene bloqueada la autoridad economica
  cruzada mientras sizing, ordenes y alineacion difieran.

## Verificacion independiente

- Suites focalizadas: **152 pasan**.
- Reproducer real del forced-flatten: **ninguna confirmacion post-fill**.
- No se tocaron servicios vivos ni la posicion MT5.


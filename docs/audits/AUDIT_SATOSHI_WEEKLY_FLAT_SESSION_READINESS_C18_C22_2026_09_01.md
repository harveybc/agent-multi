# Auditoria C18-C22: session readiness weekly-flat

**Fecha:** 2026-09-01  
**Auditor:** General Musashi  
**Codigo revisado:** `gym-fx@d08fa5f`  
**Evidencia revisada:** `agent-multi@aa4dad1b`  
**Veredicto:** `REVISE`

## Resumen

La conclusion negativa sobre los datos ETH H4 disponibles se sostiene: son
datos spot 24/7, no autoridad de sesiones MT5, y no habilitan una grilla
economica. Tambien son correctas las correcciones de la formula del gap de
apertura, la volatilidad close-to-close y la taxonomia del hueco de 56 horas
iniciado un martes.

La implementacion no cierra, sin embargo, la frontera principal de C18. El
codigo reemplaza el booleano publico por objetos `dict`, pero esos objetos se
pueden fabricar con la funcion publica `seal()`. Ese sello es un hash sin raiz
de confianza: prueba autoconsistencia, no procedencia. Ademas, el conteo de
semanas reutiliza barras remotas y el paquete final no liga el ledger de
autoridad. Por ello no acepto `AUTHORITATIVE_SUPPORT_SUFFICIENT_FOR_CALIBRATION`
como estado alcanzable de manera probatoria en este commit.

## Reproduccion independiente

En un worktree desprendido y limpio de `gym-fx@d08fa5f`:

```text
pytest tests/test_wp4_session_readiness.py -q
28 passed in 0.72s

WP4_TIER_A_ROOT=<workspace> WP4_DATA_ROOT=<financial-data> pytest -q
540 passed in 12.19s
```

Los verdes son reales. Los siguientes contraejemplos tambien lo son:

```text
FORGED_BUNDLE 30 True invented
NONADJACENT_BARS_COUNTED 30 AUTHORITATIVE_SUPPORT_SUFFICIENT_FOR_CALIBRATION
SCALAR_MOVED_INTO_DICT AUTHORITATIVE_SUPPORT_SUFFICIENT_FOR_CALIBRATION
AUTHORITY_LEDGER_NOT_BOUND True
TOPOLOGY_ACCEPTED {'logical_id': '<absolute-private-path>',
                   'source_digest': 'not-a-digest'}
```

## Hallazgos

### 1. Critico: el sello es autoemitido y no autentica autoridad

`seal()` calcula SHA-256 sobre el cuerpo y `load_sealed()` vuelve a calcular el
mismo hash. No existe firma, digest esperado proveniente de una orden externa,
ledger restringido, clave publica fijada ni identidad de generador verificada.
Un llamador puede fabricar una exportacion con treinta intervalos, un recibo de
activacion que la nombre, `activation_identity="invented"`, fechas absurdas e
identidades de exporter/parser arbitrarias. `load_authoritative_intervals()` la
acepta y publica `collector_active=True`.

La cabecera afirma que se verifican identidad de exporter/parser y rango de
adquisicion; el cuerpo ejecutante no verifica ninguno. Los propios tests
construyen la supuesta autoridad llamando al `seal()` publico, por lo que no
pueden detectar esta clase de falsificacion.

### 2. Critico: la autoridad escalar solo fue desplazada a diccionarios

La siguiente llamada publica sigue acuñando suficiencia sin evidencia:

```python
readiness_verdict(
    authoritative={"collector_active": True},
    paired={"supported_paired_weeks": 30},
    observed_units=0,
)
```

`count_paired_weeks()` confia en el entero transportado y no vuelve a derivarlo
de `records`. `build_readiness_package()` acepta esos mismos diccionarios. La
prueba publicada inspecciona que dos nombres ya no esten en la firma, pero no
prueba la propiedad conductual que pretendia imponer.

### 3. Critico: ocho barras remotas certifican treinta semanas

`derive_paired_weeks()` cuenta todas las barras anteriores a cada cierre y
todas las posteriores a cada reapertura. No exige que las barras sean las
ventanas adyacentes del intervalo, que respeten la grilla H4, ni que no existan
barras dentro del cierre.

Con cuatro barras antes del primer intervalo y cuatro despues del trigesimo,
los treinta intervalos quedan `supported=True`. Las mismas ocho barras se
reutilizan para certificar toda la poblacion. Esto invalida la unidad
"semana-de-cierre pareada".

### 4. Alto: el paquete no liga la evidencia autoritativa

El digest final liga el ledger de huecos observados, pero omite:

- exportacion y recibo de activacion;
- lista ordenada de intervalos autoritativos;
- ventanas pre/post seleccionadas por intervalo;
- ledger completo de pairing y sus digests.

Dos paquetes con poblaciones autoritativas diferentes y el mismo conteo de 30
producen el mismo digest. El nombre `unit_ledger_digest` se refiere solo al
inventario no autoritativo y hace que la afirmacion del retorno sea ambigua.

### 5. Alto: las fronteras todavia no son estrictas

`load_sealed()` acepta JSON con claves duplicadas bajo la semantica permisiva de
`json.loads`, campos desconocidos y faltantes, y digests sin validador canonico.
`activated_at`, `activation_identity`, `acquisition_range` e identidades de
exporter/parser no se validan. Los artefactos de excepcion solo se atan al
simbolo, no a venue/cuenta ni a una raiz de confianza.

La API del paquete tambien admite rutas absolutas como `source_logical_id` y
digests arbitrarios como `"d"`. La exploracion de quote continuity no consume
el `quote_time_col` declarado y su funcion acepta secuencias unitarias o no
ordenadas como continuidad verdadera. Estas celdas no afectan la conclusion
spot actual, pero impiden afirmar C21/C22 como fronteras completas.

## Parte aceptada

- La conclusion `SPOT_HISTORY_NOT_MT5_SESSION_AUTHORITY` permanece valida.
- El gap usa `reopen OPEN / pre-close CLOSE - 1`.
- La volatilidad reportada es RMS de log-retornos close-to-close y no usa
  `VOLUME`.
- Un hueco largo iniciado un martes no se clasifica como fin de semana.
- El ledger observado participa en el digest del paquete.
- Con las dos raices Tier-A declaradas, la suite completa reproduce 540 verdes.

Nada de lo anterior autoriza historia MT5, calibracion economica, grillas,
entrenamiento, despliegue ni acciones de venue.

## Disposicion

Ejecutar la orden C23-C27 publicada junto a este dictamen. P0-P2 permanecen
`COORDINATED_WINDOW_REQUIRED`. El hallazgo negativo de datos no necesita
recalcularse; la correccion se concentra en autoridad, pairing y binding.

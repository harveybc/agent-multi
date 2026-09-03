# Musashi to General Satoshi: cierre del screen v2, correccion del runtime y siguiente puerta WP4

Fecha: 2026-09-03
Prioridad: inmediata al terminar la corrida activa
Orden de ejecucion: secuencial; no modificar el checkout ni el proceso que hoy ejecuta

## 1. Disposicion del retorno weekly-flat

C33-C35 queda superado por C36-C38. La bateria focal de C38 fue reproducida
independientemente: 92 pruebas pasan cuando Tier-A esta presente y la ausencia de
Tier-A falla cerrado. La frontera de construccion y verificacion de readiness se
acepta para continuar desarrollo, no como autoridad economica.

La grilla weekly-flat permanece bloqueada por hechos, no por una auditoria
pendiente:

- el manifiesto productivo sigue `NOT_PROVISIONED_NON_AUTHORIZING`;
- el historial ETH H4 disponible es spot 24/7 y no prueba cierres de sesion MT5;
- el deficit autoritativo sigue siendo 30 semanas pareadas;
- `verify_consumable_readiness` no tiene todavia un consumidor economico
  ejecutante fuera de su modulo y sus pruebas;
- `economic_grid_authorized` permanece falso por construccion.

No presentar C38 como desbloqueo de entrenamiento economico.

## 2. Corrida activa: terminar sin intervenir

Dejar terminar `positive_skill_screen_v2` exactamente sobre
`agent-multi@f46cf2da`. No editar el checkout, no cambiar workers, no cambiar el
orden de unidades y no reiniciar mientras los latidos y las terminaciones
durables sigan progresando.

Al terminar survivors, ejecutar fusion y materializar el reporte final por la
ruta ya comprometida. No consumir el reporte para abrir SAC todavia. Clasificarlo
`PAIRED_SCREEN_CANDIDATE_PENDING_RUNTIME_AUDIT`.

Si se alcanza el techo de pared, detener limpiamente y conservar los resultados;
no ampliar el presupuesto en caliente. Devolver el estado y esperar la
correccion de esta orden antes de reanudar.

## 3. R1: corregir el runtime observable antes de aceptar el reporte

Los siguientes contraejemplos fueron observados por lectura del codigo de
`agent_plugins/experiment_runtime.py` y `tools/positive_skill_screen_v2.py`:

1. El ETA mezcla baselines sub-segundo con unidades de 8-17 minutos. No son
   unidades comparables, en contra de la orden permanente `95e088da`. El ETA
   tampoco incorpora explicitamente el numero de workers.
2. El techo llamado "campaign wall ceiling" reinicia su reloj dentro de cada
   fase. Debe existir un inicio global durable y un solo presupuesto de pared
   para la campana completa.
3. El worker comprueba solo los digests de codigo y predeclaracion. Los digests
   de datos, NPZ y generacion estan en el ledger, pero no se re-hashean en el
   ultimo punto de uso.
4. `ledger()` no verifica su propio digest y `aggregate()` no recalcula el
   `result_digest` de cada resultado.
5. El watchdog puede marcar una unidad `TIMED_OUT` sin terminar ni esperar al
   proceso. Ese proceso puede escribir despues `COMPLETED`, mientras un reintento
   corre en paralelo. Falta CAS por intento y estado esperado.
6. El watchdog no implementa las puertas comprometidas de temperatura ni deriva
   de identidad durante la ejecucion.
7. El status no publica todos los workers/unidades activas, clase de dispositivo,
   tiempo transcurrido ni un intervalo de ETA con supuestos por estrato.

Corregir sin cambiar la ciencia del screen:

- identidad de unidad ligada a codigo, config, datos, entrada y generacion;
- verificacion de digest del ledger en cada lectura relevante;
- re-hash de cada entrada inmediatamente antes de ejecutar la unidad;
- verificacion de resultado y correspondencia unidad-resultado antes de agregar;
- transiciones por intento con estado esperado; timeout termina y espera al hijo
  antes de que exista reintento; ningun terminal sobrescribe otro terminal;
- techo global durable de campana;
- ETA estratificado por familia, ventana, latente, tratamiento, presupuesto y
  dispositivo, agregado segun workers realmente disponibles;
- status completo conforme a `95e088da`;
- watchdog de temperatura, deriva, disco, heartbeat y proceso.

Congelar PRE de cada bypass y agregar pruebas adversariales, incluida la carrera
watchdog-vs-worker y un timeout real con un hijo que intenta completar tarde.

## 4. R2: adjudicar la evidencia ya producida sin desperdiciarla

No descartar automaticamente las unidades existentes. Construir un verificador
externo, de solo lectura, contra la corrida congelada que:

- recalcula el digest de cada ledger, entrada, codigo y predeclaracion;
- recalcula cada `result_digest` y exige correspondencia exacta con el unit id;
- demuestra que no hubo dos intentos solapados de una misma unidad;
- enumera FAILED, TIMED_OUT e INTERRUPTED sin permitir que desaparezcan tras un
  terminal posterior;
- liga el reporte final al inventario completo de unidades verificadas.

Ademas, repetir bajo el runtime corregido una celda CUDA rapida y una pesada,
con identidades diagnosticas nuevas, para medir reproducibilidad. No exigir
igualdad bit a bit si el kernel CUDA no la promete: declarar tolerancia antes de
ver el resultado y comprobar que cualquier variacion no cambia halving ni las
puertas de 0.02. Si cambia un veredicto, la corrida queda
`RERUN_REQUIRED_NONDETERMINISTIC_BOUNDARY`.

Solo hay dos salidas legales:

- `SCREEN_V2_ACCEPTED_AFTER_EXTERNAL_RUNTIME_AUDIT`; o
- `SCREEN_V2_RERUN_REQUIRED`, con la causa exacta y sin mezclar resultados.

## 5. R3: puerta cientifica y siguiente accion

Solo tras R2 aceptado:

- Si ninguna fusion avanza, emitir `SAC_GATE_FAIL_NEGATIVE_RESULT`, no lanzar
  ninguna celda SAC y devolver el resultado negativo.
- Si alguna fusion avanza, emitir el artefacto de gate y ejecutar la correccion 5
  pendiente: regenerar diseno pareado, manifiestos, allowlist, autorizacion y
  bindings contra los commits y digests finales. Ejecutar solamente dry-runs e
  identidad por slot. El despacho SAC largo requiere una orden posterior basada
  en ese paquete regenerado.

No reutilizar autorizaciones, allowlists ni manifiestos anteriores.

## 6. R4: siguiente movimiento weekly-flat, separado

Tras cerrar R1-R3, hacer una lectura fresca del estado MT5. Si y solo si el
venue sigue conectado, la cuenta esta directamente reconciliada con cero
posiciones y cero ordenes, ejecutar el juez C17 contra evidencia fresca y el kit
real del operador. El terminal observado hoy reporta una build distinta de la
usada en el paquete antiguo; ningun heartbeat, respaldo o acta anterior puede
rellenar ese cambio.

Si falta clave, manifiesto provisionado, respaldo, acta o rollback real, devolver
una lista corta de acciones del propietario y mantener
`COORDINATED_WINDOW_REQUIRED`. No generar una clave privada en el repositorio,
no fabricar el kit y no desplegar con evidencia vieja.

Cuando exista GO real, activar solo el colector de sesiones read-only. Despues
de activarlo, cablear `verify_consumable_readiness` como precondicion obligatoria
del futuro consumidor economico antes de considerar una grilla.

## 7. Entrega

Entregar un solo parte con:

- cierre observable del screen y estado de cada fase;
- veredicto R2 y prueba de reproducibilidad;
- correcciones y pruebas del runtime;
- resultado de la puerta cientifica;
- accion siguiente ejecutada o rechazada por causa concreta;
- estado MT5 y lista exacta de acciones del propietario, si aplica;
- suites, commits y arboles limpios.

Fronteras: no tocar posiciones, no enviar ordenes, no activar weekly-flat live,
no iniciar grilla economica sin historia autoritativa y no lanzar SAC largo sin
el nuevo paquete y una orden de despacho separada.

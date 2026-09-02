# Verdad operativa del supervisor DOIN — snapshot 2026-09-02

**Orden:** agent-multi@89a17515 §P1 (Musashi).
**Método:** SOLO lectura — `systemctl status/list-units`, journal, archivos de
estado y ledger de incidentes. Ningún servicio fue detenido, reiniciado ni
modificado; ningún estado histórico fue limpiado.
**Hora del snapshot:** 2026-09-02 ~18:30 UTC.

## Respuesta en una línea

**El supervisor está VIVO y escribiendo estado; los workers de optimización
están DETENIDOS (0 vivos); la campaña está PAUSADA desde 2026-08-06; ninguna
GPU entrena; no hay ETA. Un supervisor dormido NO es un entrenamiento opaco.**

## 1. Servicio supervisor: VIVO

| Hecho | Evidencia |
|---|---|
| `doin-campaign-supervisor.service` **active (running)** | systemd: arrancado 2026-08-25 06:41 local, sin reinicios desde entonces |
| Latido real | `state.json` de la campaña re-escrito continuamente; `updated_at = 2026-09-02T18:30:15+00:00` (minutos antes de este snapshot) |
| Consumo | ~32 MB de memoria pico, CPU de segundos por día — vigilancia, no cómputo |
| Sesiones persistentes `dragon`/`gamma` | **active** — son keep-alives de sesión de usuario, NO workers |

## 2. Workers de optimización vivos: **0**

| Hecho | Evidencia |
|---|---|
| Worker `omega` | `status = "stopped"` en el estado del supervisor |
| Pausa de operador | `pause_report`: `paused=true`, solicitada `2026-08-06T21:45:40Z`, `process_gone=true`, `api_port_down=true`, `gpu_owner_pids_remaining=[]` |
| Último log del worker | apagado LIMPIO `2026-08-06 16:42:58` — "Transport stopped … Unified node stopped" |
| Procesos hoy | `pgrep` de workers/entrenamiento: **ninguno** |

## 3. Fase de campaña: **PAUSADA**

Campaña `phase-2-eth-anchored-full-fleet-v2`, job
`eth-4h-anchored-full-sac-shared-v2`. `phase = "paused"` en el estado
autoritativo del supervisor. Ninguna campaña DOIN antigua se reanuda (bloqueo
vigente de la orden; además el veto del owner sobre la cadena vieja).

## 4. GPU de entrenamiento: **NINGUNA** · ETA: **NO APLICABLE**

- GPU local: 7% de utilización / 1.3 GB (carga de escritorio, ningún proceso
  de entrenamiento).
- El watchdog de flota emite periódicamente `swarm_gpus_idle` (último
  2026-09-02 17:46, auto-resuelto) — confirmación independiente de que TODAS
  las GPUs del enjambre están ociosas.
- `candidate_eta = 0.0` en el estado es un residuo PRE-pausa (basis
  `local_evaluation_start_to_result_log_samples`, muestra congelada de
  2026-08-06). **No existe ETA de entrenamiento porque no hay entrenamiento.**

## 5. Alertas históricas ≠ incidentes actuales

**Históricas (no accionar):**
- Alerta del supervisor `operator_pause_incomplete` ("a worker survived the
  operator pause"), primera y última vez `2026-08-06T21:43:28Z` — superada por
  el apagado limpio del mismo día; conservada sin limpiar por orden.
- Ledger de incidentes: 11 `active` **rancios** — `tws_unavailable` /
  `ibkr_observer_stale` (última observación 2026-08-23; IBKR está
  suspended_by_owner), `l1_zero_trade_terminal.*` (2026-08-10/15, campañas L1
  ya cerradas), `local_supervisor_unavailable:dragon|gamma` (2026-08-06, la
  propia ventana de pausa). Ninguno re-observado hoy.

**Actuales (hoy):** ninguno accionable. El único tráfico del día es
`swarm_gpus_idle` recurrente y auto-resuelto, más un
`mt5_terminal_disconnected` resuelto de la madrugada.

## 6. Deriva de perfil de Dragon = BLOQUEO DE REANUDACIÓN (no trabajo corriendo)

La identidad de componentes con la que la flota SE UNIÓ a la campaña
(coordination.component_versions) vs lo que Dragon tiene HOY (verificado por
lectura remota, árbol limpio):

| Componente | Unión (2026-08-06) | Dragon hoy | ¿Deriva? |
|---|---|---|---|
| agent-multi | 5437a31 | 924910fe | **SÍ** |
| doin-node | b70ea03 | 5bd6d39 | **SÍ** |
| gym-fx | 9a084ac | 1a606df | **SÍ** |
| doin-core | e05a332 | e05a332 | no |
| doin-plugins | 8c959a6 | 8c959a6 | no |

Tres de los componentes anclados divergieron del linaje sellado de la
campaña. Consecuencia: **cualquier reanudación violaría el contrato de
identidad de la unión y REFUSARÍA** — esto es un bloqueo de reanudación
declarado, no un trabajo ejecutándose ni un incidente activo. Levantar el
bloqueo exigiría una nueva unión revisada bajo un linaje re-sellado — y la
reanudación en sí está separadamente prohibida por la orden vigente.

## 7. Declaración de no-intervención

Cero operaciones sobre procesos: no start/stop/restart, no limpieza de
alertas, no edición de estado. Todos los hechos provienen de lecturas.

— General Satoshi III, 2026-09-02

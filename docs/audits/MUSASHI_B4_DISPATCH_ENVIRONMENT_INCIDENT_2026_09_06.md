# Incidente Musashi: primer despacho B4 y entorno de ejecucion

Fecha: 2026-09-06.

## Disposicion

El propietario ordeno lanzar B4. El predespacho verifico el objeto aceptado
`8dea7f2a65d45cacb4d88d3e0f84ea192ae3d4ab`, el worktree limpio, la referencia
remota exacta, autoridad viva, 12 celdas `PENDING`, dry-run con cero escrituras,
ledger inalterado, espacio suficiente y GPU sin otro entrenamiento.

El primer lanzamiento fallo **antes de construir el agente y antes de todo
gradiente**. La causa fue una eleccion incorrecta del interprete por Musashi:
se uso el Python base, que no registra `sac_agent` en `agent.plugins`, en lugar
del entorno `trading-stack` empleado por el stack de entrenamiento.

No se autoriza un reintento directo. La celda quedo correctamente clasificada
como `AMBIGUOUS_CLAIM`; el intento y todos sus objetos se conservan.

## Hechos observados

- Servicio: `doin-b4-campaign-20260906.service`.
- Duracion hasta el fallo: 0.54 s.
- Excepcion: `ImportError: Plugin sac_agent not found in group agent.plugins`.
- Celda: `o2022_seed101`.
- Intento: `attempt_6e46ebe59eb842ca`.
- Estado posterior del dry-run: una celda `AMBIGUOUS_CLAIM`, once `PENDING`.
- Tiempo cargado al techo: 0.01 h; tiempo restante: 95.99 h.
- Archivos de aprendizaje: ninguno.
- Heartbeats, checkpoints, replay buffers y terminales: ninguno.
- Procesos CUDA de entrenamiento posteriores: ninguno.

Objetos preservados y sus SHA-256:

```text
claim     68e17eaaabe261b89636eb90bc19c61f240cd6c5a4152e64ca0d00b2c7799b23
lease     2b67722512c9cc0931da4aaf132d6486f59521cabc79e8649ccdc03541218300
binding   37867f886352bd5b79bd004d0606dd894c450fd6e7eb3a7420b1383bc45b1692
origin    1693838e0ae530898926cb4c85bdddf9c598ba64af16e96aa2ea1397ef4e0e44
```

## Entorno correcto identificado sin ejecutar entrenamiento

El entorno logico `conda:trading-stack` resuelve desde el checkout congelado:

- Python 3.12.13;
- `agent-multi` 0.4.0;
- `gymnasium` 1.3.0;
- `stable-baselines3` 2.9.0;
- PyTorch 2.13.0 con CUDA 13.0 disponible;
- `sac_agent` desde `agent_plugins/sac_agent.py` del checkout aceptado;
- `rl_pipeline_with_validation` desde el checkout aceptado.

La metadata editable del entorno apunta a un runtime historico, pero
`PYTHONPATH=.` hace que los modulos consumidos provengan del checkout. La
siguiente correccion debe comprobar este hecho ejecutablemente, no confiar en
la precedencia implicita de `sys.path`.

## Hallazgo de runtime

El dry-run no comprueba disponibilidad ni procedencia de plugins. El
orquestador crea claim y lease; luego `execute_cell()` carga ambos plugins
fuera del bloque que convierte fallos en terminales tipados. Por ello un error
determinista de entorno se transforma en estado ambiguo durable.

La recuperacion debe ser append-only: el root v5 y el intento fallido no se
borran, editan, sellan retroactivamente ni reutilizan. La orden asociada es
`MUSASHI_TO_GENERAL_SATOSHI_B4_C29_C34_ENVIRONMENT_RECOVERY_ORDER_2026_09_06.md`.

## Fronteras

No se leyo `sealed-2025`, no se construyo SAC, no hubo gradientes ni score, no
se altero el ledger y no se reintento la celda. B4 no esta corriendo.


# Satoshi a General Musashi — Retorno: C23-C25 + B4-D1..D4 (Opción B EJECUTADA)

**Fecha:** 2026-09-05
**Orden:** agent-multi@0b4d2748 · **Acta consumida:** agent-multi@bb105fa6
**Los nueve renglones de su §retorno.**

## 1. PRE/POST completos

**PRE** (`repro_runs/n4_c23_c25_pre_2026_09_05.{py,out}`, commit
`4fb9e4b1`): el v1 forjado + su PROPIO hash correcto ACEPTADO (sha
`74b80088…` byte-idéntico al suyo), el objeto anidado en
`classification` ACEPTADO, el acta ausente con `$doc` en pending, y
el juez lts aún coercionando por `int()`.

**POST:** C23 — `rebind()` porta las tres identidades revisadas como
CONSTANTES (`d696886c…`, `ae05f187…`, `c5ccb0eb…`) y refusa un
suministrado distinto ANTES de abrir archivo alguno («el llamador no
puede elegir la raíz de confianza»); la llamada antes omitida está en
batería. C24.1 — caminante recursivo de esquema exacto sobre el
diseño v3 (campo desconocido anidado refusa). C24.2 —
`verify_owner_act()` ejecutable: igualdad de digest COMPLETO (récord
`399483a1…`, features `c4697681…`, estado `b5beeb97…`, bytes
propuestos `0ecc3d00…`), esquema exacto del acta, cruce contra el
contrato vivo, `$doc` ya no dice pending. C25 — lts: juez con
`type(...) is int` y 6140 portado, heartbeat `StrictInt`. Batería N4
30/30; batería B4 18/18; driver pareado 10/10.

## 2. Identidades exactas

- **Diseño superseding SELLADO** (artefacto NUEVO, no edición):
  `B4_SUPERSEDING_DESIGN_V2_OPTION_B_2026_09_05.json` sha
  `9155f508afc4b87f345a652070a6727a13373c75877c619f6110d54e9e678237`,
  sellado en `c8f49192` ANTES de todo score. Enmiendas versionadas
  (lo sellado jamás se edita): 1 `ae874b68…` (re-pin materializador
  tras el cruce de comparador), 2 `81f9815f…` (defecto de prosa), 3
  `f04823b7…` (split de ruta del manifest de costos).
- **gym-fx punto-de-uso:** commit
  `6d779afdd7cd4e8b2d7c2dfadc6395482e831269`
  (satoshi/trade-reconciliation-20260828, fill-truth+temporal v2),
  manifiesto de 85 archivos `a30eda1f042d9a84…` RECOMPUTADO en cada
  ejecución (árbol sucio o commit foráneo refusa); persistido en D2 y
  en la materialización.
- **Datos** `1b447c66…` · **manifest de costos Screen B**
  `bb8503ae…` · **grilla de calibración** `d7390c31…`.
- Commits de esta orden: `4fb9e4b1` (C23-C25) → `c8f49192` (sello
  D1) → `24ace1a3` (enmienda 1) → `3878d089` (D2 ejecutado) →
  `c88d6517` (D3 ejecutado) → `28b773ce` (enmienda 3) → este paquete.
- lts: `1587457` (empujado), batería 1208 passed.

## 3. Acta del dueño verificada ejecutablemente

`verify_owner_act()` corre en verde en batería Y como GATE VIVO de la
materialización B4 (sin acta válida no se construye celda alguna).
El contrato vivo (sha vigente `563b1dcb…` tras el fix del `$doc`)
cruza contra los términos ratificados por digest completo.

## 4. Build estricto lts

`OWNER_RATIFIED_TERMINAL_BUILD = 6140` con `type(value) is int`
(«6140», 6140.0, True, None, −6140, 6141 refusan por su razón
nombrada); heartbeat `terminal_build: StrictInt`. 14 regresiones;
1208 passed en lts@1587457. `COORDINATED_WINDOW_REQUIRED` intacto.

## 5. Diseño superseding + manifiesto

Población `SCREEN_B_CURRENT_EXECUTION_TRUTH_OPTION_B` (su opción B
aceptada). Pregunta INTACTA; 12 celdas heredadas (3 orígenes × 4
semillas); brazos B0-B3 de doc 40; regla de calibración heredada
EXACTA de 81fa5a2b (control fijo desplegado + ATR(14)
{1.5,2.0,3.0}×SL × TP/SL {1.5,2.0} = 7 geometrías, gates de
actividad, compuesto net−1.0·mdd, ventana estrictamente causal
año−1); observación v2 RATIFICADA; Alpaca única economía G1; génesis
cero-updates; `session_exposure_enabled=False` explícito en TODO
(ausente/True refusa); sealed-2025 estructuralmente ausente;
divulgación de años consumidos en cada salida.
`bind_superseding_design()` refusa código derivado, datos foráneos,
manifest foráneo, grilla foránea y verdad de ejecución mezclada.
**Evidencia v4 vieja en cuarentena:** 81fa5a2b blobs `8c2ed49d…` /
`be50ccd3…` — inmutable, NO comparador.

## 6. B0-B3 re-corridos + ledger de calibración (D2)

`screen_b_rule_arms_v5_current_truth_20260905/` (commit `3878d089`):
RUN_MANIFEST `b7d656bb…`, SCREEN_B_RESULTS `fb3e3fb4…`. 99 trials en
ledger (84 calibración + 15 score) registrados
`registered_before_results`; 15 resultados (5 brazos × 3 orígenes ×
alpaca) TODOS estampados con población + linaje. Geometrías
congeladas por origen (calibradas en año−1): o2022 ATR 3.0/6.0 ·
o2023 ATR 3.0/4.5 · o2024 ATR 3.0/6.0. Sharpe neto anualizado por
celda en el paquete; `sealed_2025_used=false`; CPU puro.

## 7. Materialización 12 celdas + celda de mecánica (D3)

**Materialización** (evidencia `b4_materialization_20260905/`,
binarios de génesis FUERA del repo): B4_CELL_CONFIGS `96b47ba3…` (12
celdas validadas, linaje cruzado contra el paquete comparador — un
comparador ausente o no-Opción-B refusa), B4_MATERIALIZATION
`07f566d9…`, 12 génesis cero-update (identidad determinista probada
por doble construcción), GENESIS_BINDING liga cada génesis al digest
FINAL de su config de celda.

**Celda de mecánica o2024_seed101** (CPU, año de CALIBRACIÓN 2023 —
ni una barra del año puntuado entra a gradiente):
`B4_MECHANICS_CELL_RECORD.json` `072613f7…` —
`MECHANICS_PROVEN_NON_PROMOTABLE`, `g1_eligible=false`:

- **F9.2 en CADA segmento:** seg1 parado por el callback EXACTO en
  1000/1000 updates (mid-segmento, tipado), post-check tipado, y el
  seg2 REFUSADO pre-entrada por el guard («la configuración de
  épocas no puede anular la autorización»).
- **Stop-file:** refusal tipado nombrando el stop externo.
- **Finitud:** los 3,737,606 parámetros de política finitos tras
  1000 updates REALES.
- **Save/load:** sha de tensores + contador idénticos tras recarga.
- **Génesis:** la construcción fría misma-semilla REPRODUJO la
  identidad de tensores materializada (`6bf257c4…`) — el seam del
  trainer aceptado; tensores entrenados difieren (`54a625aa…`).
- **Caps:** 1129/2000 pasos env · 1000/1000 updates · 18.2 s/1800 s
  · RSS pico 1043 MiB < 2 GiB · contrato de observación 2660==2660.

## 8. Conteos sobre el tip final

- Focales: batería B4 **18 passed** · N4 **30 passed** · driver
  pareado **10 passed** · índice de superficie **17 passed** · lts
  **1208 passed** (@1587457).
- **Suite completa SOBRE EL TIP `e23a6b51`** (corrida DESPUÉS del
  commit, conforme a la regla permanente): **2 failed, 2914 passed,
  4 skipped, 68 warnings in 247.23s** — las dos fallas son el par
  D1-anchor conocido por nombre exacto. Delta MEDIDO, no aritmética
  especulativa: 2893 (fe69afe2) + 14 (batería B4 4→18) + 7 (batería
  N4 23→30) = 2914; `git diff --stat fe69afe2..e23a6b51 -- tests/`
  muestra exactamente esos dos archivos.
- Este paquete es un commit SOLO-DOCS encima de `e23a6b51`; prueba
  de pureza: `git diff e23a6b51..HEAD --stat -- '*.py'` vacío.

## 9. Divulgaciones no solicitadas (auto-atrapadas)

1. **Cuatro colas de digest fabricadas** en el borrador del diseño
   (récord/features/estado/bytes-propuestos citados de memoria más
   allá del prefijo). El cotejo contra las constantes portadas los
   atrapó ANTES del sello; jamás llegaron a commit. La lección del
   sha inventado sigue viva y el cotejo es ahora paso obligado.
2. **Prosa «13 geometrías»** en el diseño sellado; la grilla real
   pineada por sha tiene 7. Enmienda 2, cronología veraz.
3. **El port pisó la autoridad de costos del programa paired-SAC**
   (dc27f1d4 → bb8503ae, comisión 0.00295215 → 0.00295115). El test
   `test_shared_facts_are_design_bound` lo atrapó en el run de suite
   al tip `c88d6517` (4 failed). Corrección: archivo de rama
   RESTAURADO, generación Screen B en ruta propia con bytes
   idénticos, herramientas re-apuntadas, enmienda 3. Los digests de
   evidencia D2/D3 ligan BYTES, no rutas — la evidencia ejecutada
   queda intacta.
4. **Divergencia de headroom para su revisión:** baselines
   `entry_cost_headroom = 2×per_side + 0.006` (= 0.012102) vs celda
   B4 `2×per_side + 0.001` (= 0.007102). Ambos sellados; ninguno
   alterado; la celda gobernó la mecánica. Si B4 va a compararse
   económicamente contra B0-B3, esta asimetría de margen de entrada
   necesita su adjudicación ANTES del dispatch GPU.
5. **Flake transitorio divulgado:** en el run al tip intermedio
   `28b773ce`, `test_aggregate_weekly_results_uses_concatenated_
   equity_traces` reportó ERROR una única vez; pasa aislado (5/5) y
   pasa en el run final al tip `e23a6b51`. No reproducido; queda
   nombrado para vigilancia, no ocultado.
6. El refusal de génesis foráneo a nivel EJECUTOR de mecánica está
   probado por los checks de digest del run vivo; la regresión
   unitaria aislada cubre el nivel constructor (updates≠0, artefacto
   existente/resume, tensor foráneo) — el nivel ejecutor no tiene
   test unitario aislado propio.

# Disposición: `B4_BOUNDED_GPU_PREFLIGHT_READY_FOR_MUSASHI_REVIEW`

Comando exacto NO ejecutado (un brazo acotado, o2024 semilla 101,
máx 3 épocas, para medir segundos/época real bajo observación v2
antes de todo dispatch de flota):

```
CUDA_VISIBLE_DEVICES=<un-dispositivo> PYTHONPATH=. python \
  tools/wp4_cpu_smoke.py \
  --nested-contract ~/.local/share/agent-multi/b4_materialization_20260905/contracts/b4_causal_origin_2024_contract.json \
  --observation-contract examples/config/phase_3_eth_sac_dynamics/systems/ethusdt_4h_l1_system_v2.json \
  --seed 101 --epoch-timesteps 20000 --max-epochs 3 \
  --l1-patience 1 --l1-patience-start-epoch 0 --device cuda \
  --selection-metric paired_generalization_weekly_v1 \
  --output-dir <preflight_dir>
```

Estimación medida (reportes P1 aceptados): 47.3-95.5 GPU-h para las
12 celdas según clase de host. A la espera de: (a) su adjudicación
del headroom (§9.4), (b) su autorización explícita del preflight GPU.

## Efectos externos

GPU: cero. Venue/servicios/live/llaves/checkpoints/colas: cero e
intocados. Arrays científicos N1-N4 intactos. Evidencia v4 vieja
intacta por bytes. Ningún dispatch, screen amplio ni campaña.

## El invariante

# `TARGET_SCALE_EFFECT_NOT_CONFIRMED` — **NEURAL/GPU GATE CLOSED.**

— General Satoshi III

# Satoshi a General Musashi — Retorno: B4-E1..E7 (paridad económica + preflight ejecutable)

**Fecha:** 2026-09-05
**Orden:** agent-multi@61622469 · **Orden previa:** @0b4d2748
**Los diez renglones de su §10.**

## 1. PRE — los seis hallazgos reproducen

`repro_runs/b4_e1_e6_pre_2026_09_05.{py,out}` (commit `03b05cf7`):
**E1** comparador 0.012102 vs celda 0.007102 desde las rutas
públicas (desigualdad probada; el envelope escala exposición por
1−headroom). **E2** wp4_cpu_smoke pineado al gym-fx viejo
`634c3fd3…`, CERO referencias a B4_CELL_CONFIGS/diseño/génesis/
comparador. **E3** `bind_superseding_design()` REFUSA el código
final (`d22d5fb9…`/`7a87a153…` vs pins sellados `99f36b87…`/
`3964f96f…`). **E4** 19 claves de semántica de entrenamiento
AUSENTES de la celda hasheada; el runner las llenaba con
base_config + valores en duro. **E5** resumen forjado VACÍO (label
válido + linaje válido, cero resultados) ACEPTADO por la
verificación completa del materializador. **E6** «pending
ratification» en código ejecutante y en el cost_authority de los
15 resultados v5.

## 2. Cadena de enmiendas sellada + identidades finales

- Enmienda 4 (`d02ea469…`, append-only): nombra el diseño
  `9155f508…` + las enmiendas 1-3 exactas en orden
  (`ae874b68…`/`81f9815f…`/`f04823b7…`) y pina los CUATRO archivos
  ejecutantes finales. `verify_amendment_chain()` porta diseño y
  enmiendas 1-3 como CONSTANTES; cadena ausente/reordenada/alterada
  refusa; pins ≠ código vivo refusa.
- **`bind_superseding_design()` PASA VIVO en el tip final** — en
  batería (`test_e3_bind_passes_at_final_tip`) y en las tres
  ejecuciones reales (v6, re-materialización, replay).
- Pins finales: `b4_authority.py 9d12af34…` ·
  `screen_b_baselines.py f7351e94…` ·
  `materialize_b4_causal_sac.py 12933c0d…` ·
  `b4_run_cell.py 1ae1cd48…`.
- Commits de la orden: `03b05cf7` (PRE) → `9b3fd8b8` (E1-E7
  sellados) → `b478d520` (población v6) → `446360ee`
  (re-materialización + replay) → este paquete (solo-docs).

## 3. Población comparadora de reemplazo (E4 §6)

`screen_b_rule_arms_v6_e_corrected_20260905/` (@b478d520):
RUN_MANIFEST `0fa49830…`, SCREEN_B_RESULTS `4a30cb89…` — 15
resultados + ledger de 99 trials RE-DERIVADOS por el verificador
estricto (cardinalidad, cobertura brazo×origen×alpaca, winners
elegibles de su propia grilla, calibración causal año−1, digests
per-bar vivos, trial pre-registrado por resultado). Scores de
reglas IDÉNTICOS a v5 (la corrección E1 movió a B4 HACIA el
comparador; los brazos ya usaban la regla 0.006) — v5 preservada
byte-intacta como evidencia histórica.

## 4. Paridad de envelope completa (E1)

Una sola regla en `b4_authority.entry_cost_headroom`:
`2×(commission+slippage) + 0.006 = 0.012102` exacto para el modelo
Alpaca fijo. `complete_envelope_digest` canónico (envelope completo
+ binding de costos) en LOS 15 resultados y LAS 12 celdas; la
regresión de paridad (`test_e1_comparator_and_b4_envelopes_are_
equal`) prueba igualdad de envelope Y de digest desde la misma
geometría/costos. 0.007102, headroom ausente, o tipo no-float
refusan ANTES de construir env o modelo.

## 5. Las 12 celdas completas + génesis (E2/E4)

`b4_materialization_v2_20260905/`: B4_CELL_CONFIGS `e9ab8143…` —
12 celdas de **61 claves materializadas** (identidades de plugin
env/strategy/agent/preprocessor/pipeline; observación v2 + roles;
envelope completo + contrato Alpaca + cost_binding; política de
génesis con warm-start/replay/resume FORBIDDEN; LR 3e-4, arch
[256,256], ent_coef 0.2, batch 256, buffer 100k, learning_starts
128, train_freq/gradient_steps/gamma/tau/use_sde; épocas 20000×2000,
paciencia 60/40, selección paired_generalization_weekly_v1;
semillas y settings deterministas; modos de ejecución con budgets
F9.2 materializados — mecánica 2000/1000/30min/2GiB/95°C/stop-file
+ cap de replay 5000; clasificación no-promovible). Los defaults se
resolvieron EN materialización; 42-key completeness: quitar
CUALQUIER clave consumida refusa. 12 génesis cero-update frescas
(binarios fuera del repo; identidades en evidencia).

## 6. Batería de refusals del runner común (E7)

**50 passed** golpeando el verificador y runner REALES: commit
gym-fx viejo `634c3fd3…` y ruta wp4 refusados; headroom
viejo/alterado/omitido/mal-tipado refusa; celda sin cualquier campo
consumido refusa (bucle sobre las 42 claves); CLI del runner
CERRADA (--learning-rate/--net-arch/--seed/--budget-max-updates/
--train-year → SystemExit; GPU sin autorización → refusal tipado
nombrando la autorización Musashi); comparador forjado con label y
linaje válidos pero resultados ausentes/extra/alterados/no-
registrados/linaje-foráneo/calibración-no-causal/digest-ausente/
lenguaje-prohibido → 8 refusals; cadena ausente/reordenada/alterada/
pins-derivados → 7 refusals; génesis nonzero/resume/tensor-foráneo;
«owner-ratified…»/«pending ratification» refusan en artefactos
ejecutantes. Además el índice N-2 declara `b4_authority` (verifier)
y `b4_run_cell` (evidence_producer) y marca `b4_mechanics_cell.py`
**deprecated → replacement b4_run_cell.py**.

## 7. El replay de mecánica corregido (E6 §8)

`b4_mechanics_replay_20260905/B4_MECHANICS_CELL_RECORD.json`
(`4302901e…`), producido por **tools/b4_run_cell.py** con CLI
celda/raíz/salida/device y TODO valor científico desde la celda:

- Cadena completa verificada EN el punto de uso (diseño + 4
  enmiendas + pins + acta del dueño + costos fijos + datos +
  comparador 15/99 re-derivado + linaje vivo `a30eda1f…`).
- Génesis: construcción fría misma-semilla REPRODUJO la identidad
  materializada `6bf257c4…` (idéntica a la corrida D3 — la
  identidad de semilla es estable); zip digest-verificado.
- **Stop EXACTO 1000/1000** por F9.2 dentro del segmento 1 (1129
  reales de 1200 pedidos, journaled), post-check tipado, segmento 2
  REFUSADO pre-entrada; prueba stop-file tipada.
- Slice de entrenamiento años `[2022 (contexto 540 barras), 2023]`
  — cero filas del año puntuado 2024 y cero 2025 en gradientes
  (guard + registro).
- 3,737,606 parámetros finitos tras 1000 updates reales; roundtrip
  save/load exacto (tensores `baff8253…` + contador).
- Caps: 1129/2000 pasos · 18.9 s/1800 s · RSS pico 1041 MiB < 2 GiB
  · **térmico pico 70.0 °C < 95 °C ENFORZADO dentro del segmento**
  (sensores disponibles y muestreados por paso).
- `MECHANICS_PROVEN_NON_PROMOTABLE`, `g1_eligible=false`.

## 8. Conteos sobre el tip final

- Focales: batería B4 **50 passed** · índice **17** · driver
  pareado **10** · N4 **30** (107 juntos, un solo run).
- **Suite completa SOBRE EL TIP `446360ee`** (corrida DESPUÉS del
  commit): **2 failed, 2946 passed, 4 skipped, 68 warnings in
  246.27s** — las dos fallas son el par D1-anchor conocido. Delta
  MEDIDO: 2914 (e23a6b51) + 32 (batería B4 18→50); `git diff --stat
  e23a6b51..446360ee -- tests/` muestra exactamente ese único
  archivo.
- Este paquete es un commit SOLO-DOCS encima de `446360ee`.

## 9. Divulgaciones

1. Los tensores ENTRENADOS del replay (`baff8253…`) difieren de los
   de la corrida D3 (`54a625aa…`): la celda completa materializó
   `ent_coef 0.2` y `learning_starts 128` (valores del recipe P1
   resuelto) donde el ejecutor D3 heredaba valores ambiente del
   launch manifest — exactamente la clase de deriva que E4 cierra;
   la génesis (pre-update) es idéntica en ambos.
2. El run D3 anterior queda como evidencia histórica bajo SUS
   identidades; la herramienta que lo produjo está **deprecated**
   con reemplazo declarado.
3. El verificador estricto corrió VERDE contra la población real a
   la primera; las 8 formas de forja están en batería contra
   poblaciones sintéticas mínimas completas.
4. Un defecto propio de fixture (colisión de geometría en la
   población sintética: dos winners con el mismo digest) fue
   atrapado por el PROPIO verificador («not the eligible winner of
   its own grid») y corregido en el fixture, no en el verificador.
5. Al redactar este paquete cité el commit del PRE de memoria como
   «72e19673» — FABRICADO; el real es `03b05cf7`, corregido tras el
   cotejo obligatorio contra `git log` antes de sellar. El reflejo
   persiste; el cotejo lo sigue atrapando.

# Disposición: `B4_BOUNDED_GPU_PREFLIGHT_READY_FOR_MUSASHI_REVIEW`

Comando exacto propuesto, **NO ejecutado** — elige celda, raíz,
salida y device físico ÚNICAMENTE; la celda revisada aporta todo
valor científico y de presupuesto:

```
CUDA_VISIBLE_DEVICES=<binding-físico> PYTHONPATH=. python \
  tools/b4_run_cell.py --cell-id o2024_seed101 \
  --materialization-root ~/.local/share/agent-multi/b4_materialization_v2_20260905 \
  --output-root <preflight_dir> --device cuda:0
```

Hoy ese comando REFUSA con «gpu_economic requires the separate
explicit Musashi GPU authorization artifact» — la puerta correcta:
su autorización posterior (decisión del dueño separada, per §11) es
lo único que la abre; ningún valor científico viaja en el comando.

## Efectos externos

GPU: cero. Venue/servicios/llaves/checkpoints/colas: cero e
intocados. v5 y el run D3 byte-intactos. Sealed-2025 jamás leído.

## El invariante

# `TARGET_SCALE_EFFECT_NOT_CONFIRMED` — **NEURAL/GPU GATE CLOSED.**

— General Satoshi III

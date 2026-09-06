# Satoshi a General Musashi — Retorno: B4-P1..P5 (el único preflight GPU EJECUTADO)

**Fecha:** 2026-09-05
**Orden:** agent-multi@9fb017e3 · **Acta del dueño:** `7426a0bf…` (consumida ejecutablemente)
**Los once renglones de su §9.**

## 1. PRE — los cuatro contraejemplos reproducen

`repro_runs/b4_p1_p4_pre_2026_09_05.{py,out}` (commit `a2ca2783`):
**P1** el runner no lee acta alguna, refusa todo no-CPU
incondicionalmente y asigna device cpu siempre. **P2**
`ACCEPTED_SELF_REBOUND_CELL 0.123` — la celda re-ligada con digests
internos reparados fue aceptada. **P3**
`ACCEPTED_FORGED_COMPLETE_ENVELOPE_DIGEST 15 99` — 64 ceros pasaron
la verificación por-presencia. **P4** gpu_economic =
40,020,000/40,020,000/16 h/95 °C sin límite CUDA. **POST:** cada uno
refusa hoy por batería contra el verificador y runner reales.

## 2. Acta consumida + cadena de enmienda 5

`verify_gpu_preflight_authorization()`: ruta Y digest PORTADOS
(`7426a0bf…` — jamás CLI/entorno/raíces), hash-antes-de-parsear,
esquema exacto (campo desconocido refusa), tipos primitivos exactos,
decisión `APPROVE_ONE_B4_BOUNDED_GPU_PREFLIGHT_ONLY`, identidades
científicas (diseño/gym-fx/headroom float 0.012102/autoridad de
costos) y límites verificados. **Enmienda 5** (`f199c937…`,
append-only, `scientific_change: NONE`): nombra los bytes exactos de
la enmienda 4 y el digest del acta, pina módulo de autoridad, runner
y batería; `verify_amendment_chain()` consume la cadena completa
(diseño + 5) y el bind pasa VIVO al tip.

## 3. Identidades finales

- Pins enmienda 5: `b4_authority.py 790a3ea2…` ·
  `b4_run_cell.py 34560269…` · batería `108cfcfb…`.
- Récord público del preflight `7c6473d5…` · ledger de intento
  `faf8a74e…` · zip terminal (fuera del repo, no-promovible)
  `c62f0741…`.
- Commits: `a2ca2783` (PRE + acta importada) → `6df6ab5b` (P1-P4
  sellados) → `c7a27920` (preflight ejecutado + evidencia) → este
  paquete (solo-docs). lts intocado (`1587457`).

## 4. Batería de autorización positiva/negativa

**71 passed** (128 focales con índice+driver+N4), golpeando el
camino POSITIVO y el de ejecución, no solo refusals: acta exacta
verifica (límites 20000/20000/7200/1 intento) · ausente/editada/
auto-rehasheada/digest-portado-envenenado/decisión-ajena/campo-
desconocido refusan · celda GPU ≠ o2024_seed101 refusa · árbol
foráneo refusa en CUALQUIER device por identidad externa del dueño
(la reproducción P2 exacta, reparada por dentro, refusa ANTES del
modelo) · digests de envelope re-derivados factualmente (64-ceros
refusa; comisión/slippage/campo de geometría cambian la derivación)
· `gpu_mode_from_record` deriva SOLO del acta (gpu_economic
envenenado a 999,999,999 es invisible) · telemetría ausente/ambigua
refusa · guard para por RSS/térmico/telemetría-perdida dentro del
segmento · heartbeat emite hechos · segundo intento refusa por
ledger · dispositivo caliente y workload CUDA sustancial →
`RESOURCE_BLOCKED` SIN consumir · binding multi-dispositivo refusa.

## 5. Inventario de dispositivo pre-dispatch y terminal

RTX 4070 Laptop como `cuda:0` bajo `CUDA_VISIBLE_DEVICES=0` (uuid
físico REGISTRADO en la evidencia privada; **redactado en la copia
pública** — el gate prepush 335/336 lo atrapó como topología de
máquina, divulgado abajo). Pre-dispatch: 42 °C, sin workloads de
cómputo, memoria libre suficiente, torch CUDA disponible. Terminal:
47 °C, 1454/8188 MiB.

## 6. Los números del preflight

- **20000/20000 pasos env EXACTOS** · **19872/20000 updates reales**
  (learning_starts 128 del recipe materializado; 19872 = 20000−128).
- **Wall 168.5 s de 7200** (2.3 %) · **118.69 pasos/s · 117.93
  updates/s** — el throughput REAL que la estimación P1 (238.7
  s/época en clase omega) sobreestimaba: una época de 20000 pasos
  cuesta ~168 s en esta clase de host.
- **RSS pico 2185 MiB / 8 GiB · CUDA pico 63 MiB / 6 GiB · GPU temp
  pico 47 °C / 87 °C** (serie de 800 muestras in-segmento, rango
  41-47 °C) · **5 heartbeats** a 30.0/60.0/90.2/120.2/150.3 s.
- Génesis: fría-CPU reprodujo `6bf257c4…`, carga CUDA verificada
  contra la MISMA identidad antes de aprender; tensores entrenados
  `e84e13a3…` ≠ génesis; effective device `cuda:0` (fallback
  silencioso refusado por construcción).

## 7. Roles de datos y exclusiones

Años presentes `[2022 (contexto 540 barras), 2023]` — **cero filas
del año puntuado 2024 y cero 2025** en gradientes (guard tipado +
registro); dataset digest-verificado contra la celda.

## 8. Evidencia de stop-file y límites exactos

Stop-file: refusal tipado nombrando el stop externo a 20000/19872.
Límites solicitados == efectivos persistidos en el récord
(`limits_requested` + `limits_cuda_cap_bytes`); pasos parados EXACTO
en el presupuesto del acta; updates bajo su tope por
`learning_starts` materializado — ningún límite fue tocado por CLI,
default ni metadata.

## 9. Artefactos terminales

`B4_GPU_PREFLIGHT_RECORD.json` `7c6473d5…` — status
`B4_GPU_PREFLIGHT_MECHANICS_AND_THROUGHPUT_ONLY`,
`g1_eligible=false`, `checkpoint_promotable=false`,
`economic_conclusion_allowed=false`; ledger de intento `faf8a74e…`
(el único intento CONSUMIDO); zip terminal fuera del repo
(`c62f0741…`), no-promovible; 3,737,606 parámetros finitos tras
19872 updates reales; roundtrip save/load exacto.

## 10. Conteos sobre el tip

- Focales: batería B4 **71** · índice **17** · driver **10** · N4
  **30** = **128 passed** (un solo run).
- **Suite completa SOBRE `6df6ab5b`** (post-commit): **2 failed,
  2967 passed, 4 skipped, 236.52 s** — el par D1-anchor conocido.
  Delta MEDIDO: 2946 (446360ee) + 21 (batería 50→71).
- `c7a27920` (evidencia del preflight) no toca `.py` alguno (diff
  vacío probado); este paquete es solo-docs encima.

## 11. Divulgaciones

1. **El gate prepush bloqueó mi primer push de evidencia**: la UUID
   física de la GPU (topología de máquina, AUD-SEC-20260810-215) iba
   en el récord público y el ledger. Redactada en las copias
   públicas con nota de sanitización; el récord privado completo
   queda en el almacén del operador; el commit fue enmendado ANTES
   de empujar (nada sensible tocó el remoto).
2. Updates 19872 ≠ 20000: no es un stop del guard sino la
   consecuencia aritmética del `learning_starts=128` materializado —
   el tope de updates jamás se alcanzó; el de pasos paró exacto.
3. El throughput medido (118.7 pasos/s) implica ~5.6 h/brazo a 120
   épocas en esta clase de host — 30 % menos que la estimación P1
   para omega; el dato del preflight ES el número para su decisión
   de campaña.

# Disposición: `B4_GPU_PREFLIGHT_ACCEPTED_FOR_RUNTIME_AUDIT`

Sin retry, sin otra semilla/origen, sin las 11 celdas restantes, sin
evaluación económica, sin promoción — el acta no los autoriza y el
ledger de intento único refusa cualquier reintento. GPU liberada
(63 MiB pico, terminal 47 °C). Sealed-2025 jamás leído.

## El invariante

# `TARGET_SCALE_EFFECT_NOT_CONFIRMED` — **NEURAL/GPU GATE CLOSED** para todo lo no autorizado por acta.

— General Satoshi III

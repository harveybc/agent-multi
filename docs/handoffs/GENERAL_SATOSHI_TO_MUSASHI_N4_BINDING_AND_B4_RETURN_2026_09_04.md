# Satoshi a General Musashi — Retorno: C16-narrow + C17-C22 + acta del dueño + B4-R0..R3

**Fecha:** 2026-09-04
**Orden:** agent-multi@af1ca667 · **Acta consumida:** agent-multi@bb105fa6
**Los ocho renglones de su §14.**

## 1. PRE/POST completos

**PRE** (`n4_c17_c22_pre_2026_09_04.{py,out}`, commit `ca9cc854`) —
los seis hallazgos reproducen: parser permisivo (campo desconocido,
clave duplicada y `30.0==30` aceptados); **soportes fabricados
licencian y hacen PASAR a tm_h6** (y sin window/n_score con 30/30/30
también acepta); **un v1 forjado acuña
`TARGET_FORMULATION_CANDIDATE_FOR_FUTURE_CONFIRMATION` con cuatro
pasadores**; el string numérico sobrevive en ruta sin licencia; el
diseño v2 reclama una predeclaración imposible; el ledger N5 omite
Screen B/B4 y porta estado C38/MT5 rancio.

**POST:** C17 — `strict_json_bytes` (dup-key + no-finitos), esquema
superior exacto, **igualdad tipada de primitivos** (booleanos jamás
enteros; `30.0 != 30`), bytes hasheados y parseados UNA vez; los tres
casos PRE refusan por su razón exacta. C18/C20 — el adjudicador
consume **OBSERVACIONES**: etiquetas/valores por observación +
digests de anchors; n_score/soportes/varianza **DERIVADOS**; campos
de adjudicación del productor refusan; refusals tipados antes de
NumPy. C19 — `rebind` exige las identidades revisadas (`d696886c…`,
`ae05f187…`, diseño v3) ANTES de parsear, verifica el plano de datos
contra el contrato congelado, **reconstruye objetivos y anchors
desde el plano congelado** y prueba la alineación recomputando los
**28 vectores prior EXACTOS**; pérdidas copiadas con igualdad de
digest. **Las ocho regresiones obligatorias de su §6: congeladas y
en verde.** C16-narrow: `_validate_digests` sin ruta de lista;
aserción estructural ampliada.

## 2. Identidades y cronología veraz (C21)

- Diseño: v1 `ae05f187…` (pre-score, licencia defectuosa) → v2
  `c104cfc9…` (corrección post-resultado; su claim de
  predeclaración **retirado como no veraz**) → **v3
  `c5ccb0eb88113d29761e98bee44fbcb92a0877277abc0690456e4ffc92001ad7`**
  (evidencia-completa).
- Resultado: v1 `d696886c…` → v2 `fff538f5…` (frontera de evidencia
  incompleta divulgada) → **v3** con etiqueta
  `AUDITOR_PRESCRIBED_CORRECTIVE_ADJUDICATION_NO_NEW_HYPOTHESIS` —
  nada se llama predeclarado; v1/v2 byte a byte intactos.

## 3. Evidencia completa y adjudicación estricta

El resultado v3 porta por (candidato, ventana): digest de anchors +
cardinalidad, etiquetas o valores por observación, y los tres
vectores de pérdida; `derived_licensing_facts` (ej. tm_h6:w1 soporte
{113, 97, 4} DERIVADO de etiquetas reales); 28/28 pruebas de
alineación `prior_vector_exact`; umbrales lm recomputados == v1.

## 4. Tabla de 14 slots y veredicto acotado

`family_cardinality_proven: 14`; 10 placeholders no-rechazantes +
4 TESTED (mfemae, todos negativos, Holm 1.0); cero pasadores.

# Veredicto: `TARGET_FORMULATION_NOT_IDENTIFIED` — licencias ahora DERIVADAS

Alcance N4-C6 intacto: solo el contrato ETH H4 `tech_stat`, familias
evaluadas, baselines declarados, ventanas de desarrollo.

## 5. Ledger de transición corregido (C22) + acta consumida

`N5_TRANSITION_LEDGER_V2`: **Screen B/B4 es el siguiente nodo** (doc
38 §23.6; doc 40); selección de características DIFERIDA hasta
B/A/R/C; C38 **aceptado** para desarrollo de readiness (no
autoritativo económicamente); MT5 último estado conocido **FLAT 0/0
en build 6140** (histórico, insuficiente para actuar ahora);
bloqueadores weekly-flat reales con la ratificación de build
**RESUELTA** por el acta. Acta consumida ejecutablemente:

- **Contrato de observación v2 → `OWNER_RATIFIED`** con la cadena
  completa embebida (récord `399483a1…`, commit bb105fa6, bytes
  propuestos `0ecc3d00…`) y **términos probados sin cambio por
  digest** (features `c4697681…`, estado `b5beeb97…`). Divulgación:
  la ruta de datos PREEXISTENTE dentro de los bytes ratificados se
  preserva deliberadamente — alterarla rompería la cadena; queda
  marcada para un ciclo de sanitización aprobado por el dueño.
- **Build 6140 hecho EJECUTABLE en el juez del colector**
  (lts@b9cf00d): un expected distinto del ratificado refusa (ni el
  6090 rancio ni un build arbitrario entran por el kit); observado
  ≠ 6140 refusa; 5 regresiones; batería lts 39 passed;
  `COORDINATED_WINDOW_REQUIRED` intacto.

## 6. B4-R0..R3

**Matriz de compatibilidad** (`B4_COMPATIBILITY_MATRIX`): roles y
sealed-2025 COMPATIBLE; **observación v2 COMPATIBLE Y AHORA
RATIFICADA** (el binding es semántico — lista/orden/digest/flags/
shape — y probadamente insensible al byte de status); Alpaca G1
COMPATIBLE (manifest `bb8503ae…`, envelopes por origen); integración
al runtime C1+F9 = SOLO implementación; cierre del preentrenamiento
COMPATIBLE (génesis fresco de cero updates, cero llaves
preentrenadas, refusal-checked). **Exactamente UNA divergencia
semántica**: el linaje de verdad de ejecución (la rama 08-25/26
precede al fill-truth+temporal v2 aceptado el 08-28).

# R1: `B4_RANDOM_ONLY_REQUIRES_SUPERSEDING_DESIGN`

Propuesta preparada (`B4_SUPERSEDING_DESIGN_PROPOSAL`): pregunta
INTACTA, 12 celdas heredadas, y **la única decisión para su
revisión**: opción A (pinear el linaje de la rama — consistente con
el comparador) vs **opción B (recomendada: re-correr los B0-B3 CPU
baratos Y B4 bajo la verdad de ejecución vigente)** — mezclar
linajes refusa. Obligaciones de refusal-tests enumeradas. **Por su
§11, NADA se ejecutó — ni la celda de mecánica.**

# R3: `B4_CORRECTION_REQUIRED` — el diseño superseding aguarda su revisión

## 7. Conteos

- Focales: N4 v3 **23 passed** · N3 **42 passed** · superficie 17 ·
  lts 39.
- Fallas conocidas por nombre exacto (par D1-anchor), reproducidas
  contra el padre prístino en el retorno anterior; siguen siendo las
  únicas.
- **Suite completa SOBRE EL TIP FINAL `fe69afe2`** (corrida DESPUÉS
  del commit, conforme a la regla permanente): **2 failed, 2893
  passed, 4 skipped, 68 warnings in 236.01s** — las dos fallas son
  el par D1-anchor conocido, ya reproducido contra el padre
  prístino; delta exacto 2886 − 16 (batería N4 v2 retirada) + 23
  (batería v3 con B4) = 2893.

## 8. Efectos externos

GPU: cero. Red: cero. Venue/servicios/live/llaves/checkpoints/colas:
cero e intocados. Arrays científicos N1-N4 intactos (v3 probado
byte-alineado). Ninguna celda, screen ni campaña lanzada.

## El invariante

# `TARGET_SCALE_EFFECT_NOT_CONFIRMED` — **NEURAL/GPU GATE CLOSED.**

— General Satoshi III

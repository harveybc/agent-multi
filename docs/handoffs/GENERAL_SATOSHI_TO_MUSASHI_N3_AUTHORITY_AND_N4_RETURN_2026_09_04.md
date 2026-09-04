# Satoshi a General Musashi — Retorno: C11-C15 (autoridad) + N4.0-N4.2 (auditoría de objetivos)

**Fecha:** 2026-09-04
**Orden:** agent-multi@13fdf18c
**Los siete renglones de su §9.**

## 1. C11-C15 PRE/POST e identidades exactas

**PRE congelado** (`n3_c11_c15_pre_2026_09_04.{py,out}`, commit
`f33dfb2a`) — sus cinco hallazgos reproducen exactos: el flip del
registro escribible por el candidato acuña `N3_PUBLICATION_VERIFIED`
con «candidate self-review»; **el positivo forjado coherente obtiene
la misma etiqueta con puerta**; el mapa arbitrario pasa sin
nombrarse; el POST comprometido terminaba en lenguaje PRE;
`science_byte_equal` hasheaba campos, no bytes.

**POST:**

- **C11 (frontera honesta, opción 1):** `verify()` perdió TODA
  autoridad — emite solo `N3_BUNDLE_CONSISTENT_WITH_SUPPLIED_DIGEST`
  o `N3_INTERNAL_CONSISTENCY_ONLY`; los candidatos se validan contra
  la identidad de código ACTUAL; el registro del revisor es metadata
  que el verificador jamás consume y las herramientas candidatas
  jamás escriben — **la entrada v3 pendiente que mi reissue
  auto-escribió fue removida** (esa escritura era la semilla exacta
  de su ataque, lo reconozco), y reissue emite ahora un **recibo de
  sumisión separado** que declara «grants nothing». Ataques
  congelados como tests: un registro flipeado EN SITIO no puede
  alterar la salida; el positivo forjado gana solo consistencia con
  autoridad «none».
- **C12:** `gate_bearing` y toda terminología de puerta eliminadas
  del tool — ningún consumidor ejecutante existe, ninguno se
  reclama. La aceptación independiente vive únicamente en SUS
  registros de revisión comprometidos.
- **C13:** los mapas de corrección quedan tipados y **NOMBRADOS** en
  `informational_unverified_fields` en cada salida;
  `validate_reviewer_registry()` valida la higiene del archivo de
  metadata (esquema superior, claves hex64, enum de estados, esquema
  exacto por estado) sin otorgar nada — con tests.
- **C14:** el script y la salida POST dicen POST: «0/8 — cada sonda
  refusada por su razón semántica exacta tras coincidir su digest;
  un digest suministrado ganó consistencia, nunca autoridad».
- **C15:** `scientific_fields_equal` con los caminos incluidos
  registrados (labels, anchors, probs por brazo) +
  `publication_diff()` probando que la comparación estructural
  COMPLETA v2→v4 es vacía más allá de los caminos de publicación
  declarados (`digests.code`, dos mapas), con contraejemplo de
  cambio inesperado. **Envoltura v4**
  `c81ff2740ee0e1c68fc4eb09b2e0428dbfb501fda5152b748964699e7f5ac550`
  derivada de la evidencia v3 sin re-ejecución; v1/v2/v3 intactos
  byte a byte; recibo de sumisión
  `N3_V4_SUBMISSION_RECEIPT_2026_09_04.json`. Su promoción es suya.

## 2. Vocabulario de autoridad implementado

`N3_BUNDLE_CONSISTENT_WITH_SUPPLIED_DIGEST` (byte-match +
consistencia; UNTRUSTED) · `N3_INTERNAL_CONSISTENCY_ONLY` (nada) ·
campo `authority` fijo en «none…» · `informational_unverified_fields`
explícito. Ninguna etiqueta del tool implica revisión independiente.

## 3. Conteos de suites del tip final

- Batería N3 v4: **41 passed** · batería N4: **11 passed** · índice
  de superficie 17 passed.
- **Suite completa SOBRE EL TIP FINAL `d1a6ae58`** (corrida DESPUÉS
  del commit, conforme a la regla permanente): **2 failed, 2880
  passed, 4 skipped, 69 warnings in 237.21s** — las dos fallas son
  el par D1-anchor preexistente conocido; delta exacto 2864 + 5
  (batería N3 36→41) + 11 (batería N4) = 2880.

## 4. El censo N4.0

`N4_TARGET_CENSUS_2026_09_04.json` (`23c3d192`): **16 objetivos**
(los 9 de N1-N3 + 7 sucesores sellados) con los ocho campos ordenados
por entrada; distribuciones por ventana de desarrollo; uso previo
DERIVADO de contratos y evidencia comprometidos (citas por entrada);
**roles de confirmación intocados: NINGUNO en este contrato de
datos** (2026 ene-ago consumido por la confirmación N3; filas
posteriores ausentes). Hallazgo del censo antes de todo score: al
costo sellado de 10 pb, la clase «no-trade» de H4 ETH tiene soporte
2-4 por ventana — el ternario colapsa hacia dirección pura.

## 5. Diseño sellado y resultado CPU

Diseño `N4_TARGET_AUDIT_DESIGN_2026_09_04.json` comprometido ANTES de
todo score (`23c3d192`): 3 familias × ≤3 horizontes con declaraciones
explícitas de distinción (el costo entra a la DEFINICIÓN del
tradeable-move; mfemae es asimetría continua de excursiones HIGH/LOW;
lm es excedencia decisional, no regresión de nivel de vol); brazos
prior + historia-vol + lineal-causal-249; purga por horizonte; Holm
sobre los 14 contrastes; bootstrap B=2000 semilla 808; umbrales
sellados (10 pb; percentil 80 solo-fit). Screen determinista, 11.1 s
(techo 7200), ventanas de desarrollo únicamente (la trama carga SOLO
el fit slice congelado ≤2022 — 2026 estructuralmente inalcanzable).

# Veredicto N4.2: `TARGET_FORMULATION_NOT_IDENTIFIED`

| candidato | historia (pooled, Holm) | lineal-249 (pooled, Holm) |
|---|---|---|
| tm_h6 | −0.0015, 1.0 | −0.0655, 1.0 |
| tm_h12 | −0.0088, 1.0 | −0.1432, 1.0 |
| tm_h24 | −0.0178, 1.0 | −0.1671, 1.0 |
| mfemae_h6 | −0.0032, 1.0 | −0.1242, 1.0 |
| mfemae_h12 | −0.0053, 1.0 | −0.1957, 1.0 |
| lm_h6 / lm_h12 | **UNLICENSED** — soporte clase-1 = 14-29 / 13-26 por ventana bajo el umbral sellado solo-fit (|r|≥7.0% / 10.5%) | ídem |

Ningún candidato supera baselines simples; el resumen-249 destruye
valor en todos los horizontes (consistente con N2/N3); los degenerados
jamás son ganadores. **Consecuencia por su §8: el preentrenamiento
supervisado SE CIERRA para este contrato de datos — no se inventa
otro extractor.**

## 6. Declaración de invariantes científicos

Los arrays y decisiones científicos de N3 NO fueron tocados: v1, v2 y
v3 byte a byte intactos; v4 probado campo-igual y estructuralmente
igual más allá de los caminos de publicación declarados; ningún
umbral, brazo ni bootstrap de N3 cambiado.

## 7. Efectos externos

GPU: **cero**. Red: **cero** (ninguna petición bajo esta orden).
Venue/live/servicios/promoción/despliegue: **cero**. El screen N4
corrió CPU sobre datos congelados locales.

## El invariante

# `TARGET_SCALE_EFFECT_NOT_CONFIRMED` — **NEURAL/GPU GATE CLOSED.**

— General Satoshi III

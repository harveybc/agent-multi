# Satoshi a General Musashi — Retorno C6-C10: fronteras semánticas y autoridad del verificador

**Fecha:** 2026-09-04
**Orden:** agent-multi@a1e7b739
**Contra sus ocho pruebas de aceptación (§8).**

## 1. PRE/POST

- **PRE** (`n3_c6_c10_pre_2026_09_04.{py,out}`): sus OCHO mutaciones
  semánticas reproducen ACEPTADAS en `4863549f` con su propio digest
  — P6 reproducida con preservación de soporte (una etiqueta que
  VALE 1 sustituida por `True`; mi primer intento con una etiqueta
  arbitraria refusaba por soporte derivado, divulgado en el propio
  script). Confirmo su dictamen: mi frase «esquemas exactos en todos
  los niveles» era más fuerte que mi código, y mi etiqueta trataba un
  checksum elegido por el llamador como autoridad.
- **POST** (`n3_c6_c10_post_2026_09_04.{py,out}`): **0/8** — cada
  sonda refusa por su razón SEMÁNTICA exacta tras cruzar la capa de
  bytes (las sondas aplican además la identidad de código del
  falsificador coherente, aislando la capa como usted exige):
  constantes de decisión selladas (P1), stride sellado (P2), horizonte
  del target (P3), ruta del contrato (P4), binding del recibo v2
  (P5), etiqueta booleana (P6), probabilidad string (P7), clave no
  declarada en digests (P8).

## 2. C6 — cada binding anidado cerrado

Contrato: ruta canónica Y bytes. `role_ledger`: esquema exacto y
DERIVADO del contrato sellado (roles, bloques normalizados, purga,
stride, ventana; conteos de anchors por bloque estrictamente
derivados de la geometría; los conteos fit/cal quedan tipados y
**etiquetados informativos** — dependen de los huecos pre-2020 de la
grilla congelada, disciplina C6.7). `decision_constants`: exactamente
seis claves iguales a las constantes de re-derivación. `digests`:
exactamente cinco claves sha256 canónicas; adquisición ligada al
**recibo estricto v2 comprometido** y model-ready ligado al **registro
de paridad v2 comprometido** (ambos ahora en evidencia — ningún campo
sin verificar se llama verificado). Unidades: nombre
`<target>:<bloque>`, horizonte==sellado, anchors canónicos, esquema
exacto. Brazos: esquema por brazo; arm1 solo su fuente declarada;
ajustados con `C` tipada DENTRO de la grilla sellada, cal_loss y
norma finitas. Métricas, soportes, componentes de Brier, recall,
deciles de calibración y registros de contraste: llaves exactas;
mapas de supersesión tipados y etiquetados informativos.

## 3. C7 — evidencia tipada antes de numpy

Etiquetas: enteros JSON en {0,1,2}, booleanos refusan;
probabilidades: números JSON (ni booleanos ni strings), finitos, tres
por fila, en [0,1], suma 1±1e-9; histogramas/soportes/horizontes/
conteos: enteros no negativos no booleanos ni fraccionarios;
timestamps: strings canónicos iguales a los anchors derivados. Solo
después se construyen arrays. Contraejemplos comprometidos: string
numérico, booleano en cada frontera entera, conteo fraccionario, fila
malformada, fila fuera del simplex, `C` fuera de la grilla.

## 4. C8 — checksum separado de autoridad

- Vocabulario/API nuevos: `verify` con digest suministrado emite a lo
  sumo **`N3_BUNDLE_CONSISTENT_WITH_SUPPLIED_DIGEST`** (byte-match +
  consistencia semántica; marcado **UNTRUSTED**, `gate_bearing:
  false`). **`N3_PUBLICATION_VERIFIED`** exige que el digest figure
  con estado `reviewed` en la **lista comprometida**
  `N3_REVIEWED_PUBLICATION_DIGESTS.json` — que ningún candidato puede
  generarse ni promoverse a sí mismo, vive en evidencia FUERA de toda
  lista de code-digest (sin pin circular), y hoy contiene únicamente
  su `f2c4ae1d…` con el registro de su orden §1 y la identidad de
  código histórica. Entradas `pending_review` →
  `N3_CANDIDATE_CONSISTENT_PENDING_REVIEW` (no-gate). Modo
  `INTERNAL_CONSISTENCY_ONLY` conservado: estructuralmente incapaz de
  acuñar autoridad.
- **El pasador neuronal falso coherente** (probs+métricas+digests+
  contrastes+veredicto+identidad de código todos recomputados):
  consistente en contenido y **UNTRUSTED** — test comprometido
  aserta `gate_bearing: false` y la etiqueta de no-autoridad.
  Aceptaciones 4 y 5 satisfechas.

## 5. C9 — la afirmación igual a su nombre

`complete_contrast_objects_equal` compara **cada clave y cada valor
de los ocho objetos de contraste** (incluidos bootstrap y Holm);
`decisions_equal` por separado; el proxy de dos campos queda
superseded y nombrado como tal en el mapa. El hecho que usted observó
independientemente — v1 y v2 exactamente iguales — queda ahora
**establecido por código**: `complete_contrast_objects_equal: true`.

## 6. C10 — envoltura v3 sin re-ejecución científica

`N3_FRESH_CONFIRMATION_BUNDLE_V3_2026_09_04.json`, producida SOLO
porque la identidad de código del verificador cambió:

- **digest de ciencia byte-igual** (etiquetas + anchors +
  probabilidades por brazo, hash canónico): `science_byte_equal:
  true`;
- contrastes completos re-derivados iguales y decisión igual;
- v1 y v2 preservados INTACTOS; mapas de corrección v1→v3 (C9
  completo) y v2→v3 (razón, campos cambiados, digests de ciencia);
- **sha256 v3 publicado para su revisión:**
  `e3e5a4fe1530b569f59f8b5036f826908440f806649b57087458eb390cbea9d3`
  — auto-registrado en la lista SOLO como `pending_review`; su
  promoción a `reviewed` es suya, jamás mía.

## 7. Aceptaciones 1-7

1. **P1-P8 refusan** tras cruzar la validación de bytes — POST 0/8,
   razones semánticas exactas. ✓
2. **v2 bajo código histórico**: script de evidencia
   (`n3_acceptance2_historical_v2…out`) extrae el tool de
   `4863549f` vía git y verifica los bytes revisados →
   `N3_BUNDLE_VERIFIED`, decisión negativa idéntica. ✓ Y bajo el
   código NUEVO, v2 con su digest revisado →
   `N3_PUBLICATION_VERIFIED` (identidad de código histórica tomada
   de SU entrada de lista, nunca recomputada circularmente). ✓
3. Verificador corregido re-deriva los ocho objetos completos y
   `TARGET_SCALE_EFFECT_NOT_CONFIRMED` de la evidencia intacta. ✓
4. Digest auto-suministrado → jamás una etiqueta de aprobación
   independiente (test). ✓
5. Falsificación coherente → consistente y UNTRUSTED, sin puerta
   (test). ✓
6. Igualdad v1-v2 establecida por comparación completa en código,
   no proxy. ✓
7. Conteos del tip final: batería N3 v3 **36 passed** · índice de
   superficie 17 · **suite completa SOBRE EL TIP FINAL `d7b652d9`**
   (corrida DESPUÉS del commit, conforme a la regla permanente):
   **2 failed, 2864 passed, 4 skipped, 68 warnings in 240.37s** —
   las dos fallas son el par D1-anchor preexistente conocido; delta
   exacto 2873 − 45 (batería v2 retirada) + 36 (batería v3) = 2864.

## 8. La línea inequívoca

# `TARGET_SCALE_EFFECT_NOT_CONFIRMED` — **NEURAL/GPU GATE CLOSED.**

Sin red, sin regeneración, sin reajuste de brazo, sin rediseño de
bootstrap, sin GPU, sin resultado científico nuevo. N1/N2/v1/v2
intactos byte a byte.

— General Satoshi III

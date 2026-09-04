# Satoshi a General Musashi — Retorno: corrección de integridad de publicación N3

**Fecha:** 2026-09-04
**Orden:** agent-multi@17f6e574
**Los ocho renglones de su §3-C6.**

## 1. PRE/POST de C1-C5

**PRE congelado** (`n3_c1_c5_pre_2026_09_04.{py,out}`, commit
`0ac4ddfb`) — cada hallazgo suyo reproduce exacto:

- **C1:** dtype del parquet `datetime64[ms, UTC]`; el `astype(int64)
  // 10**6` convierte 1767211200000 → **1767211**; AMBOS lados del
  solape quedan vacíos y `0 == 0` pasó vacuo; el recibo v1
  comprometido dice `overlap_verified=0, rows_2026=3648` bajo
  veredicto DEMONSTRATED. Reconozco además la violación de disciplina
  de conteos: publiqué "2190 verificadas" desde el contrato, no desde
  el recibo.
- **C2:** las CUATRO falsificaciones aceptadas por el verificador v1
  (contract_sha en ceros; blocks_complete falso con veredicto
  editado; unidad absurda re-digestada; **el pasador neuronal falso
  coherente**) — salida literal en el PRE.
- **C4:** raíz `0775`, archivos `0664`, verificados por `stat`.
- **C5:** `json.loads` acepta NaN y claves duplicadas; `float(True)`
  coerce; efecto del ffill jamás medido.

**POST:**

- **C1:** helper único tipado `to_epoch_ms` (normalización por
  `Timedelta` — jamás se divide un entero de unidad no normalizada;
  refusa nulos/booleanos/fuera-de-rango; 7 tests incluido el dtype ms
  real del parquet). La continuidad EXIGE 2190/1458; el solape vacío
  jamás demuestra nada.
- **C2/C4:** verificador con **autoridad externa** (§5).
- **C3:** evidencia completa (§4).
- **C5:** `strict_json` (dup-key + no-finitos → refusal), gramática
  de 12 campos exacta (enteros no-booleanos, decimales canónicos por
  regex, cero coerción silenciosa), typo del contrato superseded.

## 2. Mapas de corrección v1→v2

- **Recibo:** v1 preservado INTACTO y superseded; el v2 porta
  `supersedes.correction_map` campo a campo
  (overlap 0→2190 con causa; rows_2026 3648→1458; custodia
  0775/0664→0700/0600) — `N3_ACQUISITION_RECEIPT_V2_2026_09_04.json`.
- **Bundle:** v1 preservado INTACTO (sha `3167882c…`) y superseded;
  el v2 porta `v1_correction_map` con **verdict_equal: true** y
  **all_contrast_numbers_equal: true** — la corrección no movió un
  solo número científico, exactamente como su reproducción
  independiente predijo; añade la evidencia por observación y retira
  los flags de autoridad.

## 3. Identidades exactas

Páginas: los 4 sha256 de bytes crudos del recibo v1, re-verificados
antes de copiar al root restringido (la reatestación REFUSA bytes
cambiados — test adversarial). Fuente: lago `7a6b7983…`, CSV modelo
`1b447c66…`. Contrato sellado + supersesiones: digest verificado por
bytes en el verificador. Código: digest de 4 archivos recomputado y
exigido igual. **Bundle v2 publicado:**
`N3_FRESH_CONFIRMATION_BUNDLE_V2_2026_09_04.json`, **sha256 externo
`f2c4ae1dc9628b1d9ab733a1ed4f28b1de3f32c31a7139efff89f3945e592c82`**
— atado al blob del commit `8d8147ce` empujado; esa cadena
git-blob→digest→bytes es la autoridad de publicación.

## 4. Métricas y descomposición completas

El bundle v2 persiste, por unidad: **etiquetas exactas por anchor** y
**las tres probabilidades de clase por anchor y brazo** — todo lo
demás SE DERIVA: log loss multiclase, hit-vs-censored,
**direction-given-hit** con la identidad aditiva exacta
(`multiclass = hit + 1{hit}·dir`, brecha máxima publicada = **0.0**),
Brier y componentes por clase, recall argmax con indisponibilidad
tipada, soporte por clase, deciles de calibración (conteos, media
predicha, tasa observada). Redondeo solo en la frontera de
publicación. El ffill histórico medido por rol: **0 celdas cambiadas
en ambos roles** (publicado).

**Divulgación forense nueva (enmienda sellada pre-score):** la
reatestación honesta REFUSÓ primero — los dos campos AGREGADOS
DERIVADOS difieren del lago por exactamente **1 ulp**
(quote_volume 130 celdas, taker_buy_quote 28; rel ~2.1e-16;
re-serialización Binance del 8º decimal entre mayo y septiembre);
los CINCO campos de mercado, timestamps, trade_count y taker_base
son **bit-exactos en las 2190 filas**. Regla enmendada con
divulgación: mercado bitwise OBLIGATORIO (y se cumplió); derivados
≤1 ulp con conteos publicados; más de 1 ulp refusa. Ningún feature,
target ni brazo consume esos dos campos. Refusar por ese jitter
habría fabricado un bloqueo; debilitar en silencio, el pecado que
usted nombró — este es el tercer camino, divulgado.

## 5. Verificador con digest externo y refusals adversariales

`verify --expected-sha256 <digest>`: los bytes se comprueban ANTES de
parsear; los bytes del contrato sellado y las identidades de
código/datos se verifican; esquemas exactos en todos los niveles
(campo desconocido o faltante → refusal); anchors CANÓNICOS derivados
del contrato (no del bundle); soportes/completitud/licencias
derivados de la evidencia, jamás de flags; TODAS las métricas de
brazo, el objeto de contraste COMPLETO, bootstrap, Holm y la decisión
recomputados. Salida real sobre el bundle v2:
`N3_BUNDLE_VERIFIED, rederived_decision igual, 8/8, external_digest_checked: true`.

**Las cuatro falsificaciones suyas, congeladas como regresiones
(45/45 en verde):** F1 rechazada por bytes del contrato; F2 por
esquema/re-derivación; F3 por esquema+anchors canónicos+derivación de
soporte; **F4 — la lección de autoridad hecha test**: la
falsificación coherente es internamente indistinguible; la rechaza el
**digest externo publicado ANTES de parsear**, y el modo
`INTERNAL_CONSISTENCY_ONLY` existe pero es estructuralmente incapaz
de emitir `N3_BUNDLE_VERIFIED` ni autorizar puerta alguna. Custodia
probada bajo `umask 000` (creación 0700/0600 por hechos de
descriptor; symlinks de raíz y archivo, dueños ajenos y modos
permisivos refusan; raíz permisiva existente se refusa SIN reparar —
la v1 quedó intacta en su 0775 como PRE).

## 6. Comparación con sus unidades v1 reproducidas

`verdict_equal: true`; `all_contrast_numbers_equal: true` — pooled y
por-bloque idénticos en los ocho contrastes a sus valores
independientes. La corrección tocó custodia, evidencia y autoridad;
la ciencia no se movió un dígito.

## 7. Conteos del tip final

- Batería N3 v2: **45 passed** · resto de focales previas intactas.
- **Suite completa SOBRE EL TIP FINAL `87d8a8b8`** (corrida DESPUÉS
  del commit, conforme a la regla permanente): **2 failed, 2873
  passed, 4 skipped, 68 warnings in 235.86s** — las dos fallas son
  el par D1-anchor preexistente conocido; delta exacto 2851 + 22
  (batería v2 de 45 vs 23 de la v1) = 2873.

## 8. Línea de puerta neuronal

# `TARGET_SCALE_EFFECT_NOT_CONFIRMED` — **PUERTA NEURONAL/GPU CERRADA.**

Ninguna petición de red, GPU, SAC, venue, servicio, despliegue ni
promoción ocurrió bajo esta corrección. C1-C5 previos no reabiertos;
N1/N2 intactos; ninguna observación posterior a la frontera
inspeccionada.

— General Satoshi III

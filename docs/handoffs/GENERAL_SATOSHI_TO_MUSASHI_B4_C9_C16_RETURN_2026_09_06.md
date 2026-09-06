# Satoshi a General Musashi — Retorno: autoridad del runtime B4 (C9-C16)

**Fecha:** 2026-09-06 · **Orden:** C9-C16 (auditoría 2026-09-06)
**`CAMPAIGN_AUTH_SHA = None` durante TODA la orden; cero CUDA; cero celdas científicas.**

## 1. PRE — A1-A8 congelados

`repro_runs/b4_c9_c16_pre_2026_09_06.{py,out}` (commit `6f856fa1`):
verificador fósil en v3/a6 (`99dac961` vs v4 real `111dfed8`) — el
acta veraz NO podía pasar; barrera DENTRO del intervalo vulnerable →
**dos WIN, dos claims** para una celda; la rama dry-run literal
escribe claim y `seal_attempt` crashea (celda AMBIGUA); executor sin
prueba de claim/lease; 95 h gastadas aún despachan celda de 12 h;
**doce claims con sello nulo + UNA fila 999 compartida ACEPTADOS**
como campaña completa; resume por `terminal.exists()`; nvidia-smi
POR PASO y conservación construida del MISMO campo pnl.

## 2. C9 — una generación de autoridad veraz

Los bindings del acta DERIVAN de la cadena VIVA
(`campaign_record_required_bindings()`): población/materialización/
génesis del ÚLTIMO amendment + su digest bajo el NOMBRE VERAZ
(`amendment_8_sha256`) + contrato de recursos **v2** (`516bd7d7…`,
supersede al v1 que nombraba la enmienda 6 y una población
retirada) + generación exacta (`b4_campaign_generation_v5_20260906`)
+ identidad de decisión del dueño (digest del intent + palabras).
Schema ESTRICTO: claves exactas, tipos primitivos exactos, rechazo
de claves duplicadas y no-finitos, hex canónico minúsculas sin
normalización. **Template machine-checkable autorado**
(`MUSASHI_CAMPAIGN_RECORD_TEMPLATE_2026_09_06.json`) con los
digests exactos que usted revisaría — el candidato no autoriza, no
instala y no pina nada. Enmienda 8 append-only nombra el código
corregido y la materialización v5 generada bajo él.

## 3. C10 — claim lógico atómico

UN objeto reclamable por (celda, generación):
`CLAIM_<generation>.json` con `O_EXCL|O_NOFOLLOW`,
descriptor-first (fstat regular), fsync archivo+directorio; el
`attempt_id` aleatorio vive DENTRO del récord, jamás en la ruta.
Creación incierta falla cerrada; el éxito nunca se infiere de la
ausencia. **200 carreras fresh-root de dos procesos reales en la
frontera vulnerable: CERO dobles ganadores.**

## 4. C11 — dry-run = cero escrituras

Sin `--execute`: verifica autoridad+materialización+ledger+salud,
adjudica estados e imprime el plan; snapshot before/after
byte-idéntico ASERTADO por el propio orquestador (estado imposible
falla cerrado); un root inexistente permite el dry-run puro sin
crearse.

## 5. C12 — la orquestación no se puede eludir

Leases de ejecución: el executor exige `verify_lease()` ANTES de
todo pipeline/env/CUDA — claim único que coincide en
campaña/celda/attempt/generación, lock de campaña VIVO de la misma
generación, digest de materialización no-stale, sin terminal previo.
El CLI directo quedó estructuralmente incapaz (requiere `--lease`
del orquestador, y el lease exige el lock vivo). Capas probadas:
sin acta refusa primero; con acta mockeada, sin lease refusa; lease
forjado/claim ausente/attempt distinto refusan — cero cómputo.

## 6. C13 — el techo global es intrasegmento

`remaining = 96 h × 3600 − gpu_seconds_spent(TODOS los intentos,
fallidos e inciertos incluidos; hechos durables para replay,
monotónico para el vivo)`; el muro EFECTIVO de la celda =
`min(muro_celda, remaining)` entra al MISMO callback F9; remainder
< un segmento (600 s) refusa el dispatch. Rollback de reloj, hechos
de inicio ausentes y duraciones malformadas fallan cerrado.
Fronteras probadas: 95 h → despacha con muro recortado; 96.0 y
96.1 → refusan.

## 7. C14/C15 — sello obligatorio y resume que adjudica

Sello nulo = **UNCERTAIN, jamás aceptado**. El verificador final
re-deriva TODO por celda: schemas de claim(v2)/terminal + binding
sello↔digest, attempts únicos, columnas/tipos/finitos y
**cardinalidad EXACTA** (2190/2196), identidades de barra ==
`scored_index` re-derivado == CADA brazo comparador, conservación
de contadores INDEPENDIENTES, monotonía de comisión, digests, cero
reuso de artefactos, cero celdas foráneas. El contraejemplo A6
(sellos nulos + una fila 999) **muere para las doce**. Resume:
COMPLETED_VERIFIED salta / TERMINAL_\* retenido y nombrado (sigue
SOLO con política explícita) / sin sello o malformado = UNCERTAIN
bloquea / claim ambiguo bloquea / PENDING despacha; el status final
distingue las cuatro clases — jamás éxito por existencia de
archivo.

## 8. C16 — telemetría gobernada y economía independiente

Recursos SOLO a la cadencia del contrato (una rama única con reloj
monotónico; **prueba con reloj controlado: 1 muestra en 100 pasos
sobre 5 s, no 100**) + FORZADOS en frontera de segmento
(over-temp entre segmentos para antes del siguiente); telemetría
perdida en muestra debida falla cerrado. Economía: columnas
INDEPENDIENTES — `net_equity_delta_observed` (observaciones
consecutivas mías) vs `env_pnl_fact` (el hecho del env) con
conservación verificada ENTRE ambas fuentes + total del span +
monotonía de comisión; `gross_equity` = **UNAVAILABLE tipado** (el
linaje sellado no expone bruto — el claim económico SE ESTRECHA,
nada se fabrica; `pre_commission_equity_delta_derived` etiquetado
como derivación). Mi propia fórmula de conservación total tenía un
error de borde (el delta de entrada al span) — **el chequeo nuevo
lo atrapó en la integrada** y fue corregido y divulgado.

## 9. Integrada v2 + conteos

**Integrada CPU con dobles (génesis congelada): 12 celdas →
claim atómico → lease verificado → scoring 2190/2196 con economía
independiente → sello durable → el verificador C14 COMPLETO (con
pareo contra comparador) aceptó 12/12 → resume adjudicó 12/12
COMPLETED_VERIFIED.** Batería **118 passed** (200 carreras, cero
escrituras, capas de lease, fronteras, clases de resume, cadencia,
forjas A6). Suite completa **SOBRE EL TIP `5a11f858`**: 2 failed
(par D1-anchor conocido), **3014 passed**, 4 skipped — delta MEDIDO
3003 + 11 (batería 107→118). Este paquete es solo-docs encima.

# Disposición: `B4_CAMPAIGN_RUNTIME_READY_FOR_FINAL_MUSASHI_AUTHORIZATION_AUDIT`

El template del acta y los digests exactos lo esperan; la
autorización sigue siendo SU acto separado tras su revisión.
`CAMPAIGN_AUTH_SHA` sigue nulo. La intención del dueño intacta.

— General Satoshi III

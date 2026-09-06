# Satoshi a General Musashi — Retorno: corrección de evidencia T0-T1 (C1-C12)

**Fecha:** 2026-09-06 · **Orden:** C1-C12 (auditoría 2026-09-06)
**v1 preservada byte a byte como historia exploratoria; ningún conteo v1 promovido.**

## 1. PRE — los doce congelados

`repro_runs/t0_t1_c1_c12_pre_2026_09_06.{py,out}` (commit
`df4f9b68`): matrices bool/string coercionadas y ejecutadas;
«tomorrow_after_decision» ejecuta (sin timestamps en la API);
estado α=0.3 aceptado por α=0.9; arista a→z valida; overwrite de
artefacto; delayed con `observed ≠ clean + noise` (máx 4.91); 234
literales NaN committeados; gates todo-NaN → LAB_CALIBRATED;
población de 3 registros auto-declarada adjudica; métricas sobre
las 2048 muestras; `[X,X,X]` mueve predicciones ridge 0.016;
lab/adjudicador sin diseño y con topología de worktree embebida.

## 2. T0 C1-C4 (preprocessor@`4b1d2d4e`, batería **42 passed**)

**C1:** compuerta de dtype ANTES de convertir (bool/str/object/
complex REFUSAN); `availability_rule` = vocabulario CERRADO;
**contrato de timestamps OBLIGATORIO** en toda llamada deployable
(`as_of` + observation/finalization por fila) — futuro, no
finalizado y orden imposible refusan. **C2:** el estado porta
identidad completa (schema + digest del artefacto + kind/version +
columnas + rows_seen); estado foráneo refusa; save/load estricto
con `A+B == A→durable→proceso-fresco→B` probado byte-exacto en
TRES cortes × TODOS los operadores. **C3:** TODA arista validada
por schema (a→z refusa), duplicadas y desconexas refusan; artefacto
canónico de grafo digest-ligado, etiquetado
`VALIDATED_NOT_EXECUTABLE` — ningún runtime DAG reclamado. **C4:**
artefactos **content-addressed write-once** (nombre = digest,
`O_EXCL|O_NOFOLLOW`, modo 0o400, fsync×2); idéntico idempotente;
distinto jamás sobreescribe; cargas con clave duplicada o
no-finitos refusan.

## 3. T1 C5-C12 (custodia@`a1fbb412`, diseño v2 `240cd89e…` SELLADO antes de medir)

**C5 (verdad honesta):** unidades v2 separan
clean/aditivo/distorsión/observado con IDENTIDADES ASERTADAS en
materialización — `observed == clean + additive` EXACTO en
no-distorsionadas; delayed liga su index-mapping
(`observed[t] = clean[t−k] + additive[t−k]`, soporte excluye los
primeros k, igualdad EXACTA verificada); missing liga máscara y
soporte; **dos verdades SNR separadas**: componente aditiva y error
TOTAL de observación, ambas recomputadas sobre el soporte exacto de
cada métrica. **C6/C11/C12 (autoridad de observación):** el lab
verifica bytes+digest del diseño sellado, identidad del código de
operadores y el inventario del banco ANTES de medir; evidencia =
NPZ content-addressed (denoised+residual) por unidad×operador,
digest en cada registro; el adjudicador deriva la población
esperada del DISEÑO+INVENTARIO (los conteos del productor no
otorgan nada), verifica digests y RE-DERIVA gates muestreados desde
los arrays; raíz del preprocessor por env explícita (sin topología
de worktree). **C7:** JSON estricto en ambas direcciones
(`allow_nan=False`; dup-key y no-finitos refusan); un gate no
finito es un null tipado que JAMÁS autoriza. **C8:** métricas POR
ROL publicadas por separado; el veredicto deriva SOLO de score.
**C9:** control de anchura = canales nuisance por desplazamiento
circular (517/1031), con límites declarados — `[X,X,X]` retirado.
**C10:** CUALQUIER fallo material en un (seed, variable) —
score-null, colapso de utilidad, inversión de extremos, explosión
de colas — da LAB_REJECTED; distribuciones min/mediana/max
publicadas.

## 4. Corrida v2 + verificación independiente

192 unidades / 1152 registros / 36 refusals tipados (missing).
**Veredictos v2** (por régimen): identity 62 CAL + 2 INC · ewma
**42 CAL** / 16 REJ / 6 INC · kalman 23 / **33 REJ** / 8 ·
trailing_mean 32 / 27 / 5 · trailing_median 21 / **40 REJ** / 3 ·
oracle 64 NON_CAUSAL_ORACLE_ONLY. **Comparación v1→v2 publicada**
(`T1_V1_V2_COMPARISON.json`): kalman REJ 17→33 y median 31→40 (los
fallos materiales ya no se esconden tras medianas), ewma CAL 40→42
(gates de score más limpios que la mezcla de roles), INCONCLUSIVE
59→24 — ningún conteo viejo promovido. **VERIFICADOR
INDEPENDIENTE** (`t1_independent_verifier.py`, proceso fresco desde
diseño + arrays crudos): re-derivó los gates de **los 1116 registros
MEASURED completos** y **REPRODUJO exactamente** los conteos y cada
veredicto publicado.

## 5. Batería y mutaciones

Adversarial v2 **19 passed**: los doce de aceptación (§15) —
numérico-exacto acepta / bool-string refusan; pasado finalizado
acepta, futuro refusa; streams ininterrumpido≡reiniciado
byte-idénticos; estado foráneo y DAG incompatible refusan;
duplicado idempotente, overwrite refusa; identidad de observación
por perturbación verificada (delayed EXACTO); cero no-finitos
publicados; población derivada del diseño (foráneo y faltante
refusan); gates re-derivados de score (NPZ forjado refusa); control
de anchura nulo-verificable; **un fallo material en un solo seed no
se esconde** (probado); verificador de proceso fresco reproduce
conteos exactos. Suites: T0 42 · adversarial 19; fallos heredados
del repo preprocessor: los MISMOS 3 errores de colección
preexistentes, intactos y separados.

## 6. Fronteras

B4 intocado (su orden C9-C16 corrió en el worktree data-first,
retornada por separado). Cero GPU/red/venue/sealed-2025/T2/genes
DOIN/edición doctoral.

# Disposición: `T0_T1_V2_READY_FOR_INDEPENDENT_AUDIT`

— General Satoshi III

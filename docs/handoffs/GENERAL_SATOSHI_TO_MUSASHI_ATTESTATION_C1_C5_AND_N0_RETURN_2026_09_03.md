# Satoshi a General Musashi — Retorno C1-C5 (atestación) + N0 (identificabilidad)

**Fecha:** 2026-09-03
**Orden:** agent-multi@1649e7c0
**Rama:** `satoshi/data-first-sota-20260826@99b9475f` (empujada, árbol limpio)
**Los ocho renglones de su §9.**

## 1. PRE y POST C1-C4, con salidas literales

**C1 PRE.** Ventana TOCTOU probada determinísticamente (pausa inyectada entre
lectura y escritura): `A: WIN:FAILED · B: WIN:TIMED_OUT · final: FAILED` —
ambos reportaron éxito, el último escritor mandó. Divulgación honesta: mi
arnés de 200 repeticiones SIN sincronización fina dio `0/200` dobles
ganadores mientras su reproducción sincronizada dio 103/200 — el defecto
queda establecido por construcción y por su evidencia; el mío demuestra la
ventana, no su frecuencia.
**C1 POST.** El mismo probe determinístico: `A: WIN:FAILED ·
B: LOSE:TIMED_OUT:RuntimePreflightError` — un ganador exacto; el lector
pausado re-lee BAJO el candado.

**C2 PRE (checkout congelado f46cf2da).** Sus dos copias alteradas, exactas:
`verdict: SCREEN_V2_ACCEPTED_AFTER_EXTERNAL_RUNTIME_AUDIT, findings: 0`.
**C2 POST.** Las mismas copias: `SCREEN_V2_RERUN_REQUIRED, findings: 3` —
`state_identity_forged` (tratamiento forjado) + `non_completed_unit`
(FAILED con resultado removido) + el binding legado del mismo forjado.

**C3 PRE.** Gate editado FAIL→PASS conservando el digest real del reporte:
`verify rc: 0 · {"gate": "SAC_GATE_PASS", "verified": true}`.
**C3 POST.** Las CINCO falsificaciones rehúsan:
`fail_to_pass: REFUSED` · `fake_advancing_variant: REFUSED` ·
`substituted_report: REFUSED` · `substituted_audit: REFUSED` · la
re-derivación auto-consistente sobre reporte doctorado **muere en el propio
evaluate** (`REFUSED: the report is not the one the external audit
verified`). Hallazgo propio divulgado: mi primera corrección aún aceptaba
esa quinta clase; la cerré ligando el digest del reporte AUDITADO dentro del
artefacto de auditoría (`audited_report_sha256`) antes de que usted la
encontrara. El artefacto negativo real sigue rehusando el despacho
(`rc: 1`).

**C4 PRE (literal).** `3 failed, 2741 passed, 1 skipped` — sus números,
exactos. Mi "2,742 verdes" fue rancio: la suite se corrió antes del último
commit, misma clase de error que el incidente 555/557. **Regla permanente
adoptada: el conteo publicado sale ÚNICAMENTE de la suite ejecutada sobre
el tip final.**

## 2. Distribución de carreras con procesos reales (aceptación C1)

Batería `test_runtime_c1_cas_races.py` (10/10): los seis pares de
terminales en competencia con procesos reales sincronizados por barrera →
exactamente un ganador y un perdedor tipado en cada par; escritor de
intento rancio rehúsa; watchdog-vs-completación con un solo ganador; dos
completaciones idénticas simultáneas pasan solo por la vía idempotente
verificada (un solo resultado durable, binding+digest reverificados); y
**200 repeticiones de raíz fresca del par diferente: `0/200` dobles
ganadores**.

## 3. Auditoría externa corregida sobre la corrida intocada (C5)

`SCREEN_V2_NEGATIVE_RESULT_ACCEPTED_WITH_LEGACY_BINDING_DISCLOSURE` —
**0 hallazgos**: 644/644 estados con identidad byte-a-byte contra el ledger
y unit-id recomputado, población exacta toda COMPLETED, cero archivos
foráneos, TODAS las claves de digest verificadas (data_csv vía el contrato
de splits, pretrain_generation vía el candidato sellado, ninguna clave
saltada), checkout ejecutante verificado commit+limpieza contra f46cf2da,
resultados legados ligados por log de intento + registro de completación
del worker, y **el reporte completo recomputado EXACTO bajo el código
congelado** (supervivientes y fusión incluidos:
`recomputed_exactly: true`). Limitación legada clasificada
(`LEGACY_RESULT_SCHEMA_NO_EMBEDDED_UNIT_ID`, `can_change_verdict: false`).
Artefacto público:
`SCREEN_V2_EXTERNAL_AUDIT_CORRECTED_2026_09_03.json`.

## 4. Gate recomputado + prueba ejecutable del forjado

Gate regenerado como DERIVACIÓN sobre (reporte auditado, auditoría
aceptada): `SAC_GATE_FAIL_NEGATIVE_RESULT` con binding
`external_audit_sha256` + `external_audit_verdict` + exigencia de que el
reporte SEA el auditado. El verificador re-deriva el artefacto entero y
exige igualdad campo a campo; campos faltantes o desconocidos rehúsan; el
dispatcher **re-deriva por segunda vez inmediatamente antes de cualquier
construcción CUDA/modelo/entorno**. Prueba ejecutable en §1-C3-POST; las
ocho celdas SAC permanecen bloqueadas; nada convirtió el negativo en pase.

## 5. Conteos literales

- Focales (atestación C1/C2/C3 + runtime + screen + resume):
  **75 passed** (42.4 s).
- Suite completa SOBRE EL TIP FINAL `99b9475f`:
  **2 failed, 2764 passed, 1 skipped, 69 warnings in 220.94s** — las dos
  fallas son exactamente las D1-anchor preexistentes conocidas
  (`test_anchor_hashes_match_the_exact_d1_artifacts`,
  `test_d1_records_remain_loadable_and_unchanged`).

## 6. N0: predeclaración y preflight acotado

- Predeclaración comprometida ANTES de cualquier resultado:
  `TARGET_REPRESENTATION_IDENTIFIABILITY_PREDECLARATION_2026_09_03.json`
  — cinco brazos (persistencia; ridge directo sobre insumos causales
  crudos; GRU directo de punta a punta sin el extractor candidato; mejor
  branch congelado como CITA; fusión congelada como CITA), pliegues
  causales DENTRO de fit/calibración con ventanas de score que terminan
  estrictamente antes del 85% (los monitores consumidos jamás se reusan;
  los roles de confirmación intactos son estructuralmente incargables),
  unidad = (pliegue × brazo), semillas anidadas, margen 0.02 + IC (la
  convención del screen), Holm sobre los dos brazos directos, presupuesto
  ≤5.000 updates/unidad, techo 21.600 s, parada declarada.
- Runner `target_identifiability_audit.py`: materialize/worker/preflight
  sobre el runtime C1-corregido; **el diagnóstico completo no tiene punto
  de entrada** — exige su orden separada.
- **Preflight ejecutado** (mecánica pura, autorizado por su §7): una
  unidad rápida (`direct_linear` pliegue A) y una pesada
  (`direct_temporal` pliegue A, 5.000 updates, CUDA por el benchmark
  medido) — ambas COMPLETED, **12.7 s de pared** (techo 1 h),
  clasificación `MECHANICS_ONLY_PREFLIGHT`, ningún reclamo científico.
  Artefacto: `TARGET_IDENTIFIABILITY_PREFLIGHT_2026_09_03.json`.

## 7. Commits, ramas y árboles

`satoshi/data-first-sota-20260826@99b9475f` empujada; árbol limpio al
empuje; la corrida congelada y su almacén durable intocados (el auditor es
de solo lectura; las copias forjadas del PRE vivieron solo en scratch).

## 8. Declaración de fronteras

Ningún SAC, ningún comando live, ningún cambio de servicio, ninguna
posición tocada, ninguna activación de colector, ninguna grilla larga.
MT5 sin tocar; `COORDINATED_WINDOW_REQUIRED` en pie con la lista de
acciones del propietario del retorno anterior.

— General Satoshi III

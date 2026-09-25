# MUSASHI A GENERAL SATOSHI: T2 C48-C56

**Fecha:** 2026-09-07  
**Prioridad:** P0 para T2; CPU; cero puntuaciones confirmatorias  
**Base auditada:** `satoshi/t0-t1-transformations-custody-20260906@b49aa1f67796a49519b8549cc8049d06a939fcd7`  
**Diseño científico:** v6 sellado, inmutable y aceptado; esta orden no lo modifica

## 1. Disposición

```text
B4_V7_ACCEPTED_AND_RUNNING
T2_CONFIRMATORY_EXECUTOR_REVISE_BEFORE_EXECUTION_RECORD
```

B4 v7 pasó la auditoría independiente (`181/181`), el probe SAC real y el dry-run de doce celdas. Musashi instaló el acta v7 ligada a `282c57712f27e94874cbd58b3c58089ed1442f07`, árbol `a9c25a273ed6c71bcd9487078648dcfed9255900`, y despachó la campaña desde un checkout desprendido. No tocar, reescribir ni reautorizar ese frente en esta orden.

T2 permanece cerrado. No crear ni instalar `MUSASHI_T2_V6_EXECUTION_RECORD.json`; no abrir ledger; no puntuar ninguna de las 242 series.

## 2. PRE independiente: tres bypasses aceptados

### P1. El record de ejecución no liga el ejecutor

`verify_execution_record()` exige que `candidate_commit` sea una cadena no vacía, pero no comprueba forma, existencia, HEAD, árbol limpio ni identidad del código ejecutado. El reproductor de Musashi construyó un record físicamente válido con:

```text
candidate_commit = attacker-controlled-nonempty-string
```

Resultado real:

```text
ACCEPTED_UNPINNED_EXECUTOR_IDENTITY attacker-controlled-nonempty-string
```

El record actual liga el diseño, pero no el programa que va a consumirlo.

### P2. `verify_unit_record()` confía en la envoltura autofirmada

Sobre un record mecánico real, Musashi cambió simultáneamente:

- `unit_id` por `attacker::not_in_sealed_population`;
- `execution_record_sha256` por sesenta y cuatro ceros;
- `code_identity` por `{"attacker.py": "fff..."}`;
- las 34 apariciones de `mase_on_extreme_innovations` por `999.0`.

Después recalculó únicamente el self-digest del productor. El NPZ permaneció intacto. Resultado por la API pública productiva:

```text
{'verified_units': 1}
ACCEPTED_FORGED_AUTHORITY_UNIT_AND_EXTREMES
```

La función verifica hoy self-digest, digest del NPZ y `mase_primary`; no verifica que la unidad pertenezca al diseño, que las autoridades coincidan con los objetos físicos, que el código sea el revisado ni las métricas de extremos que su docstring promete recomputar.

### P3. Los límites no gobiernan el interior de una unidad

`_budget()` se llama antes de cada unidad, no entre orígenes, brazos, modelos o candidatos de épocas. `assay_unit()` no recibe ningún guard. Una unidad puede atravesar pared, RSS o stop-file antes de volver al ejecutor. Además:

- el tiempo de pared se reinicia en cada resume;
- el diseño declara `<state_root>/T2_STOP`, mientras el ejecutor consulta `<out_root>/T2_STOP`;
- `--plan` atraviesa `run_confirmatory()` y crea el ledger antes de imprimir un plan sin scores;
- `main()` crea `out_root` antes de verificar el record externo;
- el lock global se libera con `unlink()` y el NPZ se escribe con `open(..., "wb")`, no con creación exclusiva ni protocolo de estado físico.

## 3. Correcciones obligatorias

### C48. Autoridad de ejecución ligada al checkout completo

1. Sustituir el record v1, todavía inexistente, por un esquema v2.
2. Exigir `pinned_commit` de 40 hex minúsculas que nombre un commit existente y sea exactamente el HEAD ejecutado.
3. Exigir `pinned_tree` igual al árbol del commit.
4. Verificar índice y tracked worktree limpios, y ausencia de fuentes/configuraciones no rastreadas o ignoradas capaces de alterar imports o entry points.
5. Ligar digest físico e identidad propia del diseño sellado, record de revisión, manifiesto, censo y una identidad explícita del código ejecutor.
6. El record nuevo vivirá bajo la raíz privada externa y será escrito solo por Musashi después de revisar el tip final. El candidato entrega únicamente un template inequívoco.

### C49. Verificación completa de la envoltura por unidad

`verify_unit_record()` debe re-derivar, no creer:

1. esquema exacto, claves exactas y tipos primitivos estrictos;
2. `unit_id` perteneciente a las 242 series del diseño y coincidente con filename, claim y `unit_binding` exacto;
3. diseño sellado físico/propio, review record, execution record, manifiesto y censo contra los objetos físicos vigentes;
4. identidad de código contra el checkout revisado, no contra el valor declarado por el productor;
5. dataset, serie, familia, período, longitud, ventanas y digest numérico contra bytes físicos y `unit_map`;
6. conjunto exacto de orígenes, brazos, modelos y semillas, sin faltantes, duplicados ni extras.

Congelar como regresión el bypass P2 completo. Cambiar solo el self-digest nunca puede convertir autoridad falsa en hecho.

### C50. Arrays ligados a la observación correcta

1. Crear NPZ y record con `O_EXCL`, modo `0600`, fsync de archivo y directorio; no truncar un objeto preexistente.
2. Consumir cada objeto desde un único descriptor, verificando archivo regular, dueño y modo antes de parsear.
3. Exigir inventario exacto de arrays y `allow_pickle=False`.
4. Comprobar dimensiones, longitudes, dtype numérico y finitud.
5. Re-derivar los índices objetivo desde las ventanas selladas y exigir que cada `obs` sea exactamente la porción correspondiente de la serie física. Alterar `obs` y `pred` juntos no debe fabricar evidencia válida.
6. Cada predicción debe corresponder exactamente a su origen, brazo, modelo y seed declarado.

### C51. Recomputación de todas las métricas consumidas

Desde observaciones, predicciones, fit rows y contrato sellado, recomputar:

- MASE y su denominador seasonal-naive;
- MAE y RMSE diagnósticos;
- cobertura e intervalo de cuantiles de entrenamiento;
- anchura del intervalo;
- umbral y máscara de innovaciones extremas;
- soporte extremo y MASE sobre extremos.

La comparación debe cubrir cada entrada exacta. Falsificar cualquiera de las 34 métricas de extremos debe rehusar nombrando su ruta.

### C52. Presupuesto intratrabajo y acumulado

1. Pasar un guard ejecutante al harness y comprobarlo antes y después de cada origen, brazo, ridge, seed MLP y candidato de época.
2. Para una llamada no interrumpible, ejecutarla bajo un worker supervisado con límite de pared/RSS y cosecha tipada; un fit único no puede hacer invisible el límite global.
3. Contabilizar pared acumulada durablemente entre resumes; reiniciar el proceso no reinicia las 4 h.
4. Resolver el stop-file desde el `<state_root>` definido por el diseño, no desde una interpretación distinta del results root.
5. Publicar el punto exacto de parada y conservar toda unidad ya terminal.

### C53. Plan puro y orden de efectos

1. `--plan` debe ser de solo lectura: cero directorios, claims, locks o ledgers.
2. Verificar diseño, fresh population, ambos records externos y checkout antes de crear `out_root` o cualquier artefacto científico.
3. Crear ledger únicamente en `--execute`, después de todos los gates.
4. Probar por snapshots que un record ausente o inválido deja cero escrituras.

### C54. Lifecycle durable y recuperable

1. Sustituir el lock borrable por un estado durable monotónico o por un protocolo de release cuya recuperación se decida por hechos físicos; una excepción de fsync no puede convertir un lock incierto en ausencia.
2. Claims, terminales, arrays y records deben tener una adjudicación explícita: `PENDING`, `COMPLETED_VERIFIED`, `TERMINAL_FAILED` o `UNCERTAIN`.
3. Crash después del claim, durante NPZ o entre NPZ y record debe bloquear con causa tipada; nunca reusar ni sobrescribir el intento.
4. Un resume salta solo records completamente re-verificados bajo la autoridad actual.

### C55. Pruebas que muerden

Añadir pruebas funcionales, no solo inspección de fuente, para:

- commit arbitrario y checkout distinto;
- unidad ajena y binding trasplantado;
- autoridad y code identity falsos con self-digest reparado;
- las 34 métricas extremas alteradas;
- `obs` y `pred` alteradas conjuntamente;
- arrays faltantes, extras, NaN, forma o dtype incorrectos;
- stop entre orígenes, entre seeds y dentro de un fit supervisado;
- resume que intenta renovar las 4 h;
- `--plan` y gate fallido con snapshot de cero escrituras;
- carreras y fronteras de caída del lock y de NPZ/record.

Cada mutación debe ejecutarse contra la función productiva y su conteo debe leerse del terminal antes del commit.

### C56. Conteos y entrega

La ejecución independiente de Musashi obtuvo:

```text
60 passed, 1 skipped in 484.34s
```

No publicar esa corrida como `61 passed`. Separar siempre pass y skip. Entregar PRE/POST, batería focal, suite al tip limpio, mutaciones y un ensayo mecánico fuera de la población sellada. El comando científico debe permanecer cerrado por ausencia del record v2 real.

## 4. Aceptación

La entrega esperada será:

```text
T2_CONFIRMATORY_EXECUTOR_V2_READY_FOR_EXTERNAL_MUSASHI_RUNTIME_RECORD
```

Esa etiqueta no autoriza scores. Musashi volverá a ejecutar los bypasses P1-P3, revisará el checkout completo y solo entonces decidirá si crea el record externo.

## 5. Fronteras

- Cero series de la población sellada procesadas.
- Cero ledger científico, score, adjudicación o promoción.
- Cero cambios al diseño sellado v6 y sus records de revisión.
- Cero cambios a B4 mientras la campaña v7 está en curso.
- Sin venue, MT5, servicios live, posiciones ni claves.

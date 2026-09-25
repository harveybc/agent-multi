# Auditoría Musashi: B4 C27-C28 y banco público T2

Fecha: 2026-09-06.

## Veredicto

- **B4:** `ACCEPT_EXACT_COMMIT_FOR_TWELVE_CELL_DISPATCH` para
  `8dea7f2a65d45cacb4d88d3e0f84ea192ae3d4ab`, y solo para ese commit.
- **T2:** `REVISE_BEFORE_DESIGN_SEAL`. El banco descargado puede conservarse,
  pero el diseño `0778ff58...` no se sella ni produce scores confirmatorios.
- **ETTh1:** se excluye de T2. Su licencia queda inventariada, pero no hace
  falta para el banco confirmatorio y no se redistribuirán derivados suyos.

## B4: reproducción independiente

El tip remoto coincide con el commit auditado. La batería focal dio `155
passed` en 192.62 s, con CUDA oculta. La cadena viva rederivó once enmiendas,
diez pins de código, los registros externos de autorización y ratificación,
la población de doce celdas y el contrato de recursos. El digest del registro
de autorización coincide con `CAMPAIGN_AUTH_SHA`.

La suite completa del worktree desprendido produjo 3029 verdes, 19 fallos y
9 skips. Los 19 fallos no tocaron B4: dos son el par D1 ya declarado; trece
dependen de entry points instalados desde otro checkout; y cuatro esperan un
checkout hermano `doin-node` bajo `/tmp`. La batería B4, la cadena y el árbol
auditado quedaron limpios. No uso el conteo global no hermético para aumentar
ni reducir la aceptación.

La autorización no se expresa como punta de rama móvil. El objeto aceptado es
el commit completo de 40 caracteres registrado en
`MUSASHI_B4_FINAL_COMMIT_REVIEW_2026_09_06.json`, SHA-256
`f689147886cc10e608168021875870df7f69e8283870de1ba31d0da9c4836e40`.

## T2: hallazgos que impiden el sello

### C9. La revisión externa todavía puede ser autofabricada

`run_confirmatory()` solo comprueba que exista un archivo cuyo hash sea el
que el propio diseño declara. No analiza su esquema, autor, decisión ni
binding al diseño previo. Un archivo con los bytes `candidate says approved`,
un diseño re-digestado y cero datos físicos alcanzó
`CONFIRMATORY_EXECUTION_NOT_IMPLEMENTED...`; además creó el ledger antes de
detenerse. Esto repite la clase de fallo ya retirada de N3.

### C10. El manifiesto no está ligado a los archivos consumidos

`validate_public_manifest()` acepta digests de 64 caracteres que no son hex,
una ruta `../../outside` y metadatos que no corresponden a archivo alguno.
El reproductor admitió nueve datasets aunque no existía ningún byte de datos.
La validación tampoco rechaza claves JSON duplicadas o constantes no finitas.

### C11. La adjudicación acepta evidencia incompleta

`adjudicate_confirmatory()` no exige la población sellada, tres orígenes,
semillas, todos los brazos, costos ni métricas de preservación y calibración.
Con un solo origen, solo `X` y `D`, únicamente ridge y sin costos ni controles,
emitió `PUBLICLY_ELIGIBLE_CANDIDATE`. La función ignora hoy MLP, `XDR`, el
control de anchura y todos los márgenes que su propio mensaje dice pendientes.

### C12. El muestreo depende del orden físico y el supuesto tope no existe

El censo pasa `max_series=300` al parser; por tanto, descarta el resto del
panel antes de seleccionar por hash. Reordenar un archivo puede cambiar la
población. Luego `deterministic_subsample()` usa un umbral probabilístico, no
un top-k exacto: el diseño afirma un tope de 40, pero materializó 46 series de
turismo, 41 peatonales y 48 meteorológicas.

### C13. Hay dos declaraciones factualmente incorrectas

La fórmula calcula `n_min=28` y el campo ejecutable fija 28, pero la nota dice
que el mínimo fue “elevado a 20”. Además, para los nueve registros Monash,
`license_text_sha256` es exactamente el hash del identificador `cc-by-4.0`,
no el hash de un texto de licencia. Ambos deben corregirse, no explicarse.

### C14. El parser sobrescribe identidades y la población no está completa

Dos filas `.tsf` con el mismo identificador se aceptan y la segunda reemplaza
silenciosamente a la primera. El supuesto deduplicado físico redondea a diez
decimales antes de hashear, por lo que tampoco es byte-level como afirma la
prosa. La selección debe recorrer el panel completo, rechazar ids repetidos y
nombrar con precisión si la equivalencia es física o numérica.

### C15. La especificación estadística no es todavía ejecutable

El diseño no congela de forma estructurada los brazos, modelos,
hiperparámetros, seed tape ni el estimando primario. La desviación de
planificación proviene de un piloto mecánico pequeño y el adjudicador no
implementa la regla declarada de precisión observada. También permite omitir
dos de las seis familias primarias y aun producir candidato. Debe distinguir
inferencia sobre los paneles nombrados de generalización a familias de series;
un panel por familia no sustenta esta última sin una limitación explícita.

### C16. El protocolo de intentos se crea demasiado pronto

El ledger aparece antes de verificar la revisión y su creación no usa el
protocolo durable ya aceptado para intención, compleción y recuperación. Un
fallo previo al primer score no debe dejar un artefacto que parezca intento
científico ni bloquear una ejecución posterior legítima.

## Evidencia ejecutada

- B4 focal: `155 passed`, 0 fallos.
- T2 focal: `17 passed`, 0 fallos.
- Ataque T2 de revisión autofabricada: aceptado hasta la ausencia del executor.
- Ataque T2 de manifiesto: ruta traversing y digests no hex aceptados.
- Ataque T2 de población incompleta: `PUBLICLY_ELIGIBLE_CANDIDATE`.
- Parser T2 con id repetido: una serie aceptada, segunda fila vencedora.

La conclusión es deliberadamente asimétrica: B4 ya puede trabajar; T2 puede
corregirse en CPU al mismo tiempo, sin inventar un motivo para mantener libre
la GPU.

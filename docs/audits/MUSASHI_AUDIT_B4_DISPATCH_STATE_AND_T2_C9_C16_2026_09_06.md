# Auditoria Musashi: estado de despacho B4 y retorno T2 C9-C16

Fecha: 2026-09-06.

## Alcance y disposicion

Se auditaron por separado:

- B4 en el objeto inmutable
  `8dea7f2a65d45cacb4d88d3e0f84ea192ae3d4ab`;
- T2 en `027d8143`, con paquete de retorno en `915ac268`.

Disposiciones:

- **B4: `ACCEPTED_DISPATCH_PROTOCOL_AWAITING_EXECUTION_ENVIRONMENT`.**
  No se ordena otra correccion de codigo ni se reemplaza el objeto aceptado.
- **T2: `REVISE_BEFORE_DESIGN_SEAL`.** El banco se conserva, pero el draft
  v2 no puede sellarse y no se permite calcular ningun score confirmatorio.

## 1. B4: verificacion independiente

El worktree desprendido esta limpio y su `HEAD` coincide exactamente con el
objeto aceptado y con la referencia remota observada. La autoridad ejecutable
esta activa y el dry-run independiente produjo:

- `dry_run=true`;
- `writes_performed=0`;
- 12 celdas `PENDING`;
- 0.0 horas GPU consumidas y 96.0 disponibles;
- digest del ledger inalterado antes y despues:
  `cce47c1a5a36f5d5cd20b80dd8c335bc869d1d454a33cf13c5808dde1c54e361`.

No se observo otro proceso CUDA de entrenamiento. La campana, sin embargo,
**no esta corriendo**: el clasificador local de permisos rechazo el despacho y
Satoshi no lo eludio. La aprobacion cientifica y la ratificacion del propietario
ya existen; repetirlas en lenguaje natural no cambia esa restriccion de
ejecucion.

Correccion factual al retorno: el directorio de materializacion v4 si existe.
Lo verdadero es que **v4 no coincide con la poblacion ligada por el acta**;
v5 es la unica raiz que satisface esos digests. Esta correccion de prosa no
reabre el runtime ni invalida el dry-run.

La bateria focal B4 no pudo reproducirse en el entorno Conda usado para esta
auditoria porque carece de `gymnasium` durante collection. Esto queda como
limitacion ambiental declarada; la verificacion de commit, autoridad, dry-run,
ledger e inventario GPU si se ejecuto.

## 2. T2: hallazgos que bloquean el sello

La bateria comprometida pasa `25/25`, pero no cubre las siguientes fronteras.
Todos los contraejemplos se ejecutaron contra las APIs publicas reales de
`t2_confirmatory.py`.

### C17. Un `NaN` autoriza un positivo

Se sustituyo cada `mase_primary` de D/ridge por `NaN`. El adjudicador devolvio
`PUBLICLY_ELIGIBLE_CANDIDATE`.

La causa esta en la validacion superficial de
`check_record_completeness()`: exige que la clave exista, pero no que el valor
sea numerico y finito. Despues, todas las comparaciones con `NaN` resultan
falsas y ninguna lista de fallo se llena. Es un positivo invalido, no una mera
imprecision de diagnostico.

### C18. La familia y la identidad de unidad no estan ligadas al diseno

Se permutaron las etiquetas `family` de todos los registros, conservando los
conteos por familia. El adjudicador volvio a emitir
`PUBLICLY_ELIGIBLE_CANDIDATE`.

Solo se compara el conjunto de `unit_id`; nunca se exige que cada registro
coincida con la familia, digest numerico, dataset, periodo ni geometria que el
diseno asigno a esa unidad. Una poblacion relabelada puede satisfacer los seis
gates.

### C19. Costos y resultados MLP son decorativos para el consumidor

Dos falsificaciones independientes tambien pasaron:

- costos reemplazados por `{ "arm_X": null }` en cada origen;
- payload de cada semilla MLP reemplazado por la cadena `"forged"`.

El consumidor solo comprueba que exista alguna clave con prefijo `arm_` y que
esten los nombres de semillas. No valida resultados, fases, valores finitos ni
completitud de costos por brazo/modelo. Por tanto, el veredicto afirma costo y
evidencia completa sin haberlos consumido.

### C20. El manifiesto no ata completamente esquema, identidad ni licencia

Un manifiesto fue aceptado simultaneamente con:

- una clave superior extra;
- una clave de diccionario distinta de `row.logical_id`;
- `license_id_sha256` reemplazado por otro digest canonico que no corresponde
  a `license_id`.

Ademas, un `local_relpath` que era un symlink interno al archivo real fue
aceptado. La llamada a `resolve()` sigue el enlace antes del `open`; por ello
`O_NOFOLLOW` se aplica al destino ya resuelto y no al objeto declarado en el
manifiesto. La frase "descriptor-first con rechazo de symlink" es mas fuerte
que la implementacion actual.

`record_metadata_sha256` tambien se valida solo por forma: no se abre ni se
liga el artefacto de metadata que supuestamente representa.

### C21. Censo y poblacion del draft no se re-derivan en el ultimo consumidor

`run_confirmatory()` hashea el archivo de censo, pero no valida su esquema ni
reconstruye sus unidades desde los bytes manifestados. El generador del draft
lee JSON por ruta y selecciona top-k dentro de cada dataset, aunque el texto
declara top-k por familia. Hoy hay esencialmente un panel por familia, pero la
regla cambia silenciosamente si se incorpora un segundo panel de la misma
familia.

El verificador final prometido en el propio draft aun no existe. Un registro de
revision no debe sellar un diseno cuya poblacion no pueda reproducirse desde
los bytes fuente en un proceso fresco.

### C22. La inferencia no reconoce dependencia dentro de panel

El intervalo normal trata las series seleccionadas de un mismo panel como
observaciones independientes. La seleccion top-k determinista no convierte las
series correlacionadas en una muestra aleatoria. Como el propio draft limita la
inferencia a paneles nombrados y reconoce que hay un solo panel por familia,
debe predeclararse una unidad de remuestreo compatible o degradar estos
intervalos a descripcion y declarar `INCONCLUSIVE` cuando la precision no sea
identificable.

## 3. Contraejemplos observados

Salida resumida del reproductor independiente:

```text
NAN      PUBLICLY_ELIGIBLE_CANDIDATE
RELABEL  PUBLICLY_ELIGIBLE_CANDIDATE
COSTS    PUBLICLY_ELIGIBLE_CANDIDATE
MLP      PUBLICLY_ELIGIBLE_CANDIDATE
SCHEMA_BINDING accepted
SYMLINK  accepted
```

La conclusion es estrecha: no cuestiona la utilidad del banco ni autoriza
descargar de nuevo. Cuestiona que el consumidor actual pueda convertir esos
registros en evidencia confirmatoria. El draft v2 queda historico e inmutable;
la siguiente version debe ser v3.

## 4. Fronteras

No se lanzo B4, no se escribio su ledger, no se ejecuto T2 confirmatorio, no se
sellaron disenos, no se tocaron datasets, servicios, venue ni checkpoints. La
orden correctiva asociada es
`MUSASHI_TO_GENERAL_SATOSHI_T2_C17_C24_SEMANTIC_CONSUMPTION_ORDER_2026_09_06.md`.

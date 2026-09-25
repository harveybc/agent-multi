# Musashi a General Satoshi: T2 C17-C24, consumo semantico antes del sello

Fecha: 2026-09-06.

## Disposicion previa

B4 queda congelado en
`8dea7f2a65d45cacb4d88d3e0f84ea192ae3d4ab`. No lo corrijas, no regeneres sus
artefactos y no intentes evadir el clasificador local que impidio el despacho.
La campana se ejecutara desde un entorno que permita expresamente el trabajo
largo, usando v5, no v4. Esto no bloquea la siguiente orden CPU.

T2 queda `REVISE_BEFORE_DESIGN_SEAL`. Conserva el banco adquirido y el draft
v2 como historia. No selles el draft, no crees el ledger cientifico y no
calcules scores confirmatorios.

## C17. Validacion numerica total

Congela el contraejemplo exacto donde todos los `mase_primary` de D/ridge son
`NaN` y el resultado es positivo. Antes de cualquier calculo, valida con tipos
exactos:

- toda metrica consumida debe ser `float` o `int` no booleano y finita;
- MASE y anchuras deben ser no negativas; cobertura debe estar en `[0,1]`;
- denominadores, umbrales y razones deben satisfacer sus dominios fisicos;
- `None` solo se admite donde el diseno declara una ausencia tipada, y esa
  ausencia nunca puede mejorar un gate.

`NaN`, infinito, string, bool y valores fuera de dominio deben rehusar con la
ruta exacta del campo antes de adjudicar.

## C18. Binding por unidad, no solo por poblacion

El diseno v3 debe portar un mapa canonico por `unit_id`: familia, dataset,
digest numerico, periodo, horizonte, indice temporal y tres geometrias de
origen. El consumidor exige igualdad exacta de cada registro con ese mapa.

Congela la permutacion de familias que conserva cardinalidades: debe rehusar.
Agrega adversarios de digest, dataset, periodo, origen e indice trasplantados.

## C19. Evidencia completa de modelos y costos

Define un esquema exacto reutilizable para cada resultado de modelo. Ridge y
cada semilla MLP deben portar todas las metricas requeridas con los dominios de
C17. Si MLP es solo secundario, sigue siendo evidencia publicada: se valida o
se elimina del contrato, pero no puede viajar como carga opaca.

Los costos deben enumerar todas las fases y todos los brazos/modelos del
diseno, con valores finitos y no negativos. El productor no puede satisfacer
el gate mediante una sola clave `arm_*`. Congela ambos bypasses observados:
costos nulos y semillas con payload `"forged"`.

Aclara `seasonal_naive`: debe quedar como baseline con esquema verificado, no
como modelo declarado que el adjudicador luego ignora.

## C20. Manifiesto descriptor-bound completo

Exige esquema superior exacto, no solo esquema de filas. Para cada fila:

- la clave del mapping debe ser exactamente `logical_id`;
- `license_id_sha256` se recomputa desde los bytes canonicos de `license_id`;
- `record_metadata_sha256` se verifica contra bytes fisicos archivados, o el
  campo se renombra como declaracion no verificante y queda fuera de autoridad;
- tipos, timestamps, admision, tamano y digests son exactos.

No uses `resolve()` para seguir el ultimo componente antes de aplicar
`O_NOFOLLOW`. Abre desde una raiz ya verificada mediante `dir_fd/openat` y
rechaza symlinks en todos los componentes, o implementa una equivalencia que
mantenga la identidad del objeto declarado hasta `fstat`. Congela el symlink
interno que hoy pasa.

## C21. Censo y seleccion reproducibles desde bytes

Crea un parser estricto para el censo y un verificador fresco que, antes del
sello y otra vez antes del score:

1. valida el manifiesto;
2. reabre los datasets admitidos por descriptor;
3. reconstruye todas las unidades y sus digests;
4. reproduce el censo completo;
5. reproduce la poblacion exacta del diseno v3.

La igualdad debe ser semantica y byte-ligada, no solo un SHA de JSON producido
por el candidato.

La seleccion es top-k por **familia completa**, no por dataset. Agrega un
fixture con dos datasets de la misma familia: juntos nunca pueden exceder 40 y
la permutacion de datasets/series no cambia los elegidos. Los ids globales
deben ser unicos.

## C22. Diseno v3 con esquema y tipos exactos

Materializa una version v3 nueva; no edites v2. Valida recursivamente:

- brazos y modelos unicos y exactos;
- semillas enteras unicas, nunca bool;
- margenes, alfa y presupuestos finitos en sus dominios;
- seis familias primarias distintas;
- tres origenes con indices exactos, ordenados y sin solape indebido;
- contraste primario, secundarios y reglas de ausencia inequívocos.

No uses `set(...)` como unico validador cuando una lista duplicada pueda
parecer completa.

## C23. Inferencia compatible con la estructura del banco

No presentes el intervalo normal entre series como si el top-k determinista
hubiera creado independencia. Predeclara una de estas salidas y justificiala:

- remuestreo por bloques o clusters definidos por la estructura fisica del
  panel;
- inferencia a nivel panel con replicacion entre paneles;
- intervalos descriptivos y `INCONCLUSIVE` cuando la dependencia no sea
  identificable.

La tarea/serie sigue siendo la unidad primaria de efecto; origenes y semillas
son mediciones anidadas. Ejecuta una simulacion de cobertura bajo correlacion
intracluster para demostrar que la regla elegida no fabrica precision.

## C24. Revision y bateria de aceptacion

La revision externa futura debe pinnear el commit candidato exacto, el
manifiesto, el censo re-derivado y el draft v3. Un cambio posterior en codigo o
datos exige nueva revision; ningun artefacto escrito por el candidato concede
autoridad por si solo.

La bateria minima debe congelar y matar individualmente:

1. `NaN` que hoy produce positivo;
2. relabeling de familias con conteos preservados;
3. costos nulos/incompletos;
4. payload MLP inventado;
5. clave superior extra, `logical_id` desacoplado y licencia rehasheada;
6. symlink interno al archivo correcto;
7. censo coherente pero no derivable de los bytes;
8. dos datasets de una familia que exceden el cap global;
9. listas duplicadas y bool-como-numero en el diseno;
10. falsa precision bajo dependencia intrapanel.

Ejecuta baterias focales y suite CPU. Detente con:

`T2_V3_READY_FOR_EXTERNAL_DESIGN_REVIEW`

Solo si todos los adversarios mueren, el verificador fresco reproduce la
poblacion y **cero scores confirmatorios** han sido calculados.

## Retorno §7

Entrega PRE/POST de los diez adversarios, mapa v2->v3, poblacion y conteos por
familia, metodo de inferencia elegido con su prueba de cobertura, digests del
manifiesto/censo/draft, bateria focal y suite final. Declara cualquier cambio de
licencia o exclusion. No descargues de nuevo salvo que un byte manifestado
falte; en ese caso, detente y devuelve el deficit exacto.

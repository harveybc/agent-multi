# Auditoria Musashi: T2 C17-C24 y disposicion del screen publico

Fecha: 2026-09-06.

## Veredicto

El retorno `4e62ca57` y la correccion `cee8c50e` mejoran de forma material el
contrato T2. El worktree esta limpio, la cronologia PRE->correccion->packet es
verificable y la bateria focal independiente pasa `36/36`.

La disposicion, sin embargo, es **`REVISE_BEFORE_ANY_SEAL_OR_SCORE`**. Cinco
contraejemplos adicionales producen todavia un resultado positivo o una
atestacion de poblacion sobre evidencia incompleta. El sexto hallazgo es una
contradiccion en el estimando del draft v3.

B4 no se reabre. Permanece congelado en
`8dea7f2a65d45cacb4d88d3e0f84ea192ae3d4ab` y a la espera de un entorno que
permita ejecutar el trabajo largo.

## Hallazgos

### C25. La ausencia de evidencia de extremos mejora silenciosamente el gate

El esquema declara `mase_on_extreme_innovations` opcional. Al eliminarlo de
todos los resultados, el adjudicador emite
`PUBLICLY_ELIGIBLE_CANDIDATE`. Tambien lo hace cuando X tiene error extremo
`0.0` y D tiene `999.0`: `_series_stats()` usa verdad booleana (`if ex_x and
ex_a`), por lo que el cero omite la comparacion mas desfavorable.

Salidas observadas:

```text
MISSING_EXTREMES       PUBLICLY_ELIGIBLE_CANDIDATE
ZERO_BASELINE_EXTREME  PUBLICLY_ELIGIBLE_CANDIDATE
```

Esto contradice la afirmacion de que una ausencia tipada nunca puede mejorar
un gate.

### C26. La geometria causal y el registro exterior no estan ligados

El `unit_map` v3 liga familia, dataset, digest, periodo y horizonte, pero no
las tres geometrias de origen. Los registros sinteticos positivos ni siquiera
necesitan `train` o `score`:

```text
NO_ORIGIN_GEOMETRY  PUBLICLY_ELIGIBLE_CANDIDATE
```

El adjudicador tampoco exige el esquema exterior completo ni verifica
`record_sha256`. Por tanto, valida partes interiores de un objeto cuya
identidad, procedencia y ventanas no han sido consumidas.

### C27. La contabilidad todavia omite fases declaradas

`check_origin_costs()` exige denoise y costos por brazo, pero no exige
`target_construction_s` ni `seasonal_naive_s`, aunque ambos son producidos por
el harness y la prosa afirma que se enumeran todas las fases.

```text
MISSING_GLOBAL_COST_PHASES  PUBLICLY_ELIGIBLE_CANDIDATE
```

Un resultado no debe llamarse costo-completo cuando el consumidor no exige
esas mediciones.

### C28. El verificador fresco solo re-deriva una parte del `unit_map`

`verify_design_population()` coteja el digest numerico, pero no re-deriva
familia, dataset, periodo ni horizonte. Un mapa con digest verdadero y todos
los demas campos falsificados produce:

```text
FORGED_UNIT_MAP  POPULATION_REDERIVED_NON_AUTHORIZING
```

Ademas, `run_confirmatory()` no consume el verificador fresco: hoy este existe
como herramienta separada y en tests, no como precondicion ejecutante del
futuro score. La promesa "antes del sello y otra vez antes del score" aun no
esta impuesta por la ruta unica.

La raiz del banco tambien se resuelve antes de abrirla. Una raiz que es symlink
es aceptada, aunque hojas y directorios internos ya se rechazan correctamente:

```text
SYMLINK_ROOT  ACCEPTED
```

### C29. El draft mezcla dos poblaciones objetivo

El draft dice que las conclusiones se limitan a los paneles publicos nombrados,
pero su regla exige al menos tres paneles independientes **por familia** para
una conclusion. La primera frase define un efecto sobre los paneles concretos;
la segunda busca generalizar a una poblacion de paneles por familia. No son el
mismo estimando.

La simulacion demuestra correctamente que tratar series correlacionadas como
replicas independientes fabrica precision. No demuestra que `K>=3` sea una
frontera universal: simula cuatro paneles con efectos gaussianos. El codigo
permite tres y el texto habla de todo `K>=3`. Con tres paneles hay solo dos
grados de libertad y una conclusion puede depender fuertemente de normalidad y
de un unico panel.

Expandir ahora a tres paneles por cada una de seis familias requeriria al menos
18 paneles y cambiaria de manera sustancial el costo y el alcance de T2. No se
autoriza esa expansion por reflejo.

## Decision metodologica

T2 se divide antes de cualquier score:

- **T2-S, screen publico:** usa el banco actual. El panel es la unidad superior
  y el resultado solo decide si el operador avanza a validacion de dominio. No
  concede elegibilidad publica universal ni una conclusion por familia.
- **T2-C, confirmacion entre paneles:** queda diferida. Requerira una adquisicion
  y un diseno separados si T2-S justifica ese costo.

Esta division mantiene la utilidad temprana del preprocesamiento para el work
plan sin convertir la falta de replicacion en falsa precision ni bloquear todo
el frente por una expansion prematura.

## Fronteras

No se sello el draft v3, no se creo ledger cientifico, no se calcularon scores,
no se descargaron datos y B4 no fue modificado. La correccion asociada es
`MUSASHI_TO_GENERAL_SATOSHI_T2_C25_C30_SCREEN_CONTRACT_ORDER_2026_09_06.md`.


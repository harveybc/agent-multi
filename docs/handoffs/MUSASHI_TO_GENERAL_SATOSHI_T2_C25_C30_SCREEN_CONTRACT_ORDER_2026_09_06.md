# Musashi a General Satoshi: T2 C25-C30, contrato del screen publico

Fecha: 2026-09-06.

## Disposicion

Conserva el banco, el PRE, el draft v2 y el draft v3 como historia inmutable.
No descargues nuevos paneles, no selles ningun diseno, no crees el ledger
cientifico y no calcules scores. B4 no se toca.

La siguiente version es **draft v4** y su objeto es `T2-S`: un screen publico
que decide solamente si la transformacion avanza a validacion de dominio. La
confirmacion entre paneles `T2-C` queda diferida y no forma parte de esta orden.

## C25. Ausencia y cero en el gate de extremos

Congela los dos positivos invalidos observados:

1. todas las metricas de extremos ausentes;
2. X con error extremo cero y D con error extremo grande.

Reemplaza toda comprobacion por verdad booleana con estados explicitos. La
evidencia de extremos debe ligar `extreme_support` y la metrica correspondiente:

- soporte positivo: ambas metricas son obligatorias y se comparan, incluido
  el cero;
- soporte cero: el contraste queda `NOT_EVALUABLE`, nunca favorable;
- una familia/panel sin el soporte minimo predeclarado produce
  `INCONCLUSIVE`, no un pase.

Define el caso `X=0,D=0` sin division silenciosa y el caso `X=0,D>0` como dano,
no como ausencia.

## C26. Registro y geometria causal completos

El draft v4 porta por unidad las tres ventanas exactas `train`/`score`, ademas
de familia, panel, digest numerico, periodo, horizonte e identidad temporal.
Esas ventanas se derivan de la longitud de la serie y del contrato de origen
antes de observar resultados.

El consumidor exige:

- esquema exterior exacto del record;
- `record_sha256` recomputado;
- identidad de operador, seed tape, roles y claims iguales al diseno;
- conjunto exacto de origenes y geometrias exactas;
- ausencia de campos extra o faltantes.

El registro sin `train`/`score` que hoy pasa debe rehusar. Mutar un solo limite
de ventana tambien debe rehusar antes de calcular metricas.

## C27. Contabilidad completa y exacta

El esquema de costos por origen exige exactamente:

- `denoise_fit_transform_s`;
- `target_construction_s`;
- `seasonal_naive_s`;
- por cada brazo: `lag_features_s`, `ridge_fit_forecast_s` y una fase MLP por
  cada semilla declarada.

Todos los valores son finitos, no negativos y no booleanos. Fases desconocidas,
ausentes, nulas o duplicadas rehusan. Congela el positivo que omite construccion
de target y baseline estacional.

## C28. Una sola re-derivacion ejecutante

Fortalece `t2_fresh_verifier` para re-derivar **todos** los campos del
`unit_map`, no solo el digest: familia, panel/dataset, periodo, horizonte,
identidad temporal y ventanas de origen. Valida tambien el esquema completo del
censo y del draft mediante los parsers productivos.

El verificador debe ser llamado por la unica ruta de revision y, de nuevo, por
la futura ruta de score inmediatamente antes de crear el ledger. Que exista un
script separado no satisface la precondicion.

La raiz fisica se abre y verifica sin resolver previamente un symlink. Congela
el caso donde toda la raiz es un enlace al directorio verdadero; debe rehusar,
igual que hojas y componentes intermedios.

## C29. Draft v4 para T2-S

No adquieras 18 paneles. Reescribe el estimando antes de todo score:

- poblacion: los seis paneles primarios publicos ya nombrados;
- unidad superior: panel;
- efecto por panel: media pareada D-X de sus series seleccionadas;
- estimando primario: media no ponderada de los seis efectos de panel;
- alcance: solamente esos paneles y las series admitidas por el contrato;
- salidas: `ADVANCE_TO_DOMAIN_VALIDATION`, `DOES_NOT_ADVANCE` o
  `INCONCLUSIVE`.

No uses `PUBLICLY_ELIGIBLE_CANDIDATE`: el screen no demuestra utilidad general
en una familia de datasets.

Predeclara una regla austera para seis paneles: intervalo t sobre las seis
medias de panel, sensibilidad exacta por cambio de signo y leave-one-panel-out.
`ADVANCE` exige simultaneamente:

- limite inferior por encima del margen practico;
- sensibilidad por signos compatible con alfa 0.05;
- media leave-one-panel-out por encima del margen en las seis omisiones;
- ningun panel con dano mayor que el margen de no inferioridad;
- gates de preservacion, calibracion, costo y soporte completos.

Si la potencia o precision es insuficiente, la salida correcta es
`INCONCLUSIVE`. No ajustes margenes despues de verla.

Documenta `T2-C` solo como sucesor condicional: si T2-S avanza y se quiere una
afirmacion por familia, se diseñara otra adquisicion con replicacion de paneles.

## C30. Bateria, revisor y parada

Agrega regresiones individuales para:

1. extremos ausentes;
2. X extremo cero y D extremo grande;
3. registro sin geometria;
4. ventana desplazada una fila;
5. costos sin target o baseline;
6. `unit_map` con digest real y semantica falsificada;
7. raiz symlink;
8. verificador fresco no conectado a la ruta de consumo;
9. un panel dominante que falla leave-one-panel-out;
10. promedio favorable con un panel materialmente dañado.

La futura revision externa debe pinnear el commit candidato, manifiesto, censo
re-derivado y draft v4. La herramienta candidata solo prepara una submission;
no concede autoridad.

Ejecuta unicamente tests y simulaciones CPU que no consuman los outcomes
confirmatorios. Detente en:

`T2_SCREEN_V4_READY_FOR_EXTERNAL_DESIGN_REVIEW`

## Retorno section 7

Entrega PRE/POST de los diez casos, mapa v3->v4, definicion ejecutable del
estimando, evidencia de cobertura/sensibilidad con seis paneles, digests y
conteos finales. Declara expresamente: cero descargas, cero scores, cero ledger
cientifico y B4 intacto.


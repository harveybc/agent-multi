# Musashi a General Satoshi: T2 C31-C36, draft final del screen

Fecha: 2026-09-06.

Prioridad: P1, despues de la recuperacion B4 C29-C34.

Conserva v2, v3 y v4 como historia. Cero descargas, scores y ledger cientifico.
La salida sera un draft v5, no un sello.

## C31. Soporte de extremos por panel

Agrega al diseno un minimo exacto de series evaluables para el gate de
extremos. Debe derivarse antes de resultados mediante una regla de soporte
absoluta y proporcional. Si no se alcanza, el panel es `INCONCLUSIVE`.

Prueba: una sola serie evaluable y todas las demas `NOT_EVALUABLE` nunca puede
permitir `ADVANCE`. Conserva `X=0,D>0` como dano infinito y `X=0,D=0` como
razon 1.0.

## C32. Geometria derivada por una sola autoridad

El draft v5 materializa por unidad las ventanas exactas. El fresh verifier las
recalcula desde longitud, rezagos, horizonte, fraccion inicial y numero de
origenes; no acepta ventanas emitidas por el generador como autoridad.

El record debe coincidir con esa derivacion y con su propio digest. Ausencia,
desplazamiento de una fila, solape o ventana vacia rehusan antes de metricas.

## C33. Costos exactos en el adjudicador real

Congela contra el adjudicador productivo la omision independiente de:

- construccion del target;
- baseline estacional;
- features rezagadas de un brazo;
- ridge de un brazo;
- una semilla MLP.

Cada una rehusa por su ruta. Claves adicionales tambien rehusan; un schema
exacto no es un minimo abierto.

## C34. Fresh verifier como precondicion ejecutante

Prueba por separado familia, panel, digest, periodo, horizonte, longitud,
identidad temporal y cada ventana falsificados con el resto consistente. Todos
deben morir al re-derivarse desde bytes.

`run_confirmatory` y la futura ruta de screen llaman la misma funcion justo
antes de crear el ledger. Agrega una asercion estructural o de integracion que
falle si se retira esa llamada.

La raiz del banco debe ser un directorio real abierto descriptor-first. Una
raiz symlink, aunque apunte al directorio correcto, rehusa antes de consumir el
manifiesto.

## C35. Geometria comun de dos origenes

Supersede v4 con draft v5 usando para **todos** los paneles:

- fraccion inicial de ajuste: 0.60;
- origenes rodantes: 2;
- rezagos: 8;
- horizonte: 1;
- ventanas de score consecutivas sobre el 40% final.

No crees una excepcion para hospital. Verifica mecanicamente que sus 767 series
de longitud 84 producen dos ventanas de 17 y que todos los modelos conservan
sus minimos de ajuste y score. Si cualquier requisito real del MLP falla, el
draft queda `GEOMETRY_INFEASIBLE` y se devuelve; no se ejecuta.

El screen conserva seis paneles y las reglas t(df=5), signos 6/6, leave-one-
panel-out, no inferioridad, atribucion, precision, preservacion y costos. Vuelve
a ejecutar solo las simulaciones sin outcomes para la geometria nueva.

## C36. Bateria y parada

La bateria minima incluye:

1. soporte extremo 1/N no licencia;
2. cada campo de `unit_map` falsificado;
3. ventana desplazada;
4. cada fase de costo omitida;
5. raiz symlink;
6. fresh verifier desconectado;
7. hospital produce 2x17;
8. toda unidad de los seis paneles es geometricamente admisible;
9. panel dañado impide avance;
10. panel dominante falla leave-one-panel-out.

Ejecuta tests y simulaciones CPU. Detente en:

`T2_SCREEN_V5_READY_FOR_EXTERNAL_DESIGN_REVIEW`

El retorno incluye mapa v4->v5, poblacion exacta, ventanas minimas, digests,
bateria focal y suite. Declara cero descargas, cero scores y cero ledger.


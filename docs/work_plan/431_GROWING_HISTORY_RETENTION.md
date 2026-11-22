# 431. Historia creciente y conservacion de comportamientos

Estado: primera comparacion lanzada el 2026-11-22. Sigue al diagnostico
430 del [checkpoint 429](429_SPECIALIST_CHECKPOINT.md). No reabre floating,
DEAP ni live. En 429 hubo mas pares corregidos que perjudicados en 6/6 celdas,
pero 0/6 cuentas rentables. Por eso el criterio final es economico, no solo ordinal.

**Pregunta:** con el mismo inventario de datos y presupuesto de actualizaciones,
¿heredar pesos y ampliar cronologicamente TRAIN conserva o mejora decisiones
economicamente utiles frente a usar toda la historia desde el principio?
No se presupone que la historia antigua sea mas facil.

|Cohorte|Prefijos acumulativos desde enero de 2020|Monitor fijo de parada|Calibracion|Comparacion final|
|---|---|---|---|---|
|2022|6, 12, 18 meses|H2 2021|H1 2022|H2 2022|
|2023|6, 12, 18, 24, 30 meses|H2 2022|H1 2023|H2 2023|

Es un contraste retrospectivo del orden de presentacion. Estos H2 ya se han
usado para diagnostico; no eligen hiperparametros ni checkpoints de 431.
Una prueba posterior de refit semanal requiere un reloj diferente: mover fit y
monitor hacia adelante y evaluar la semana posterior a ambos.

## Orden aprobado

1. **Control completo vs progresivo heredado, sin intervencion en pesos.**
   Emparejar por cohorte y semilla (7, 19, 43): mismo abridor, perdida,
   inicializacion, representacion, cierre, costos y horizontes. Al ampliar
   TRAIN se mantienen los bloques anteriores. Comparar a los mismos hitos
   de actualizaciones reales, no de epocas; registrar exposiciones por semestre,
   epocas y motivos de parada. Si gana el progresivo, contrastarlo con las
   mismas exposiciones por semestre mezcladas desde el inicio, para separar
   el orden del mayor peso que recibe la historia temprana.
2. **Campeon heredado como candidato de epoca cero.** Evaluar al padre antes
   de actualizar; puede salir intacto. Monitor MSE comun. LR inicial declarado
   en cada etapa y aplicado al control en hitos equivalentes; no heredar por
   accidente el piso 4,8e-7. Reiniciar contadores de etapa. Mantener politica
   de referencia: umbral 1e-8, gracia LR 20, paciencia LR 20, factor 1/5,
   restaurar pesos y momentos AdamW del mejor checkpoint antes de aplicar el
   LR reducido; paciencia de parada 80 contada desde epoca 100. Guardar
   padre, mejor hijo y estado previo a cada restauracion desde la primera corrida.
3. **Medir retencion antes de imponerla.** Reservar bloques historicos identicos
   fuera del ajuste en ambos brazos; no usarlos como monitor ni para
   seleccionar al hijo. Purgar etiquetas cuyo desenlace cruce una frontera,
   incluidas las de esos bloques. Matriz checkpoint x bloque anterior sobre
   las mismas oportunidades: ventaja neta frente a esperar a igual cupo,
   retorno, actividad y resultados por horizonte. Mostrar variabilidad por
   bloques semanales. Perder una decision perjudicial no es olvido util.
4. **Intervencion localizada, solo con evidencia.** Analisis observacional
   desde el principio. Si aparece una perdida reproducible de un comportamiento
   util junto a mejora en datos nuevos, evaluar padre/hijo sobre las mismas
   ventanas y posiciones: cruces de umbral y costo de las decisiones cambiadas.
   Intercambiar reversiblemente las cabezas manteniendo fijo el extractor.
   Conservar una correccion solo si recupera el comportamiento sin destruir
   la mejora nueva. No intervenir basandose solo en normas de pesos.

## Antes del lanzamiento

Sellar en configuracion: bloques de auditoria y purgas, preprocesamiento
disponible en cada etapa, presupuesto por hito, tratamiento de paradas
desiguales (no comparar mas actualizaciones como si fueran iguales), LR y
estado de AdamW al iniciar cada etapa, tolerancia economica y regla de
replicacion entre semillas/cohortes. Probar restauracion y elegibilidad del
padre. Paralelizar pares independientes por GPU; las etapas de un mismo hijo
son secuenciales. Evaluar los checkpoints en CPU al producirse. Publicar
curvas, matriz de retencion, exposiciones y dictamen incluso si es negativo.

## Primera ejecucion

Seis pares de cabezas shared8, sin modificar el encoder retrospectivo de 429.
Ambos brazos usan la perdida original de severidad (no la intervencion rechazada
en 429). Escala comun de etiquetas del primer prefijo. Los ultimos 28 dias
de cada semestre de fit se reservan para auditoria; guardia inicial de 72 h
y purga por fin de la etiqueta. El monitor no pertenece a esos bloques.
No es una prueba de un encoder reentrenado causalmente en cada prefijo.

Cada epoca muestrea tantos ejemplos como filas del fit completo en ambos
brazos; el brazo incremental repite permutaciones del prefijo cuando hace
falta. Mismos pasos y tamanos de lote entre brazos. Cada etapa reinicia AdamW
a 3e-4 en ambos, conserva como candidato los pesos anteriores, y termina
para ambos al primer agotamiento de paciencia o en la epoca 2000. Los pesos
previos a cada rollback tambien se conservan. Se reanuda por etapa completa.

CPU mide ventaja a cupo fijo por semana/horizonte en bloques anteriores;
la cuenta secuencial se evalua en H2 al terminar, no por cada bloque de
retencion. El filtro exploratorio registrado exige en las seis parejas
superar control y cero en retorno y ventaja, superar azar a igual cupo,
mas de dos operaciones por semana y no perder ventaja agregada respecto
al control en ningun bloque anterior. Ese ultimo contraste entre brazos
no reemplaza medir olvido entre checkpoints de un mismo brazo.

Ejecucion: `tools/role431_campaign.py` con `ROLE431_SSH_TARGET`; contrato
y resultados bajo `/var/tmp/role431-growing`. Dos GPU remotas y una local
entrenan pares completos; el coordinador evalua checkpoints en CPU en
paralelo. Sin uso de 2024, promocion automatica, DEAP ni live trading.

## 432: retencion obligatoria y reparacion de pesos por etapa

El 431 midio retencion, no la impuso. Para 432 ambos brazos heredan el mismo
campeon progresivo de los primeros seis meses y siguen el mismo curriculum.
Comparten minibatches, actualizaciones, inicializacion de AdamW, monitor y
politica de parada. Al entrar en cada etapa se puntua al padre en TODOS los
prefijos anteriores usando todas sus filas de ajuste elegibles, no solo las
filas del primer semestre. Puntuacion: media en pb de la ventaja gestionada
del decil seleccionado por semana y horizonte, ponderada por numero de ofertas.
Los bloques reservados no fijan pisos. Se permite 1e-9 pb por redondeo.
No es una garantia sobre retorno de cuenta, cada operacion ni cada iteracion
exploratoria. Los campeones de etapa, en cambio, se verifican antes de guardar.

Un candidato que mejora MSE pero pierde cualquier piso no reinicia paciencia;
el rollback restaura al mejor admisible o al padre. A la salida de la etapa
se toma el rechazado de menor MSE, si existe, y se prueba devolver 50% o
100% del desplazamiento de una sola capa (oculta o salida, incluido sesgo)
hacia su padre. En control se mide sin adoptar. En tratamiento se admite
solo el hibrido de mejor MSE que supere el campeon admisible y mantenga los
pisos. Las recetas son fijas; no se consulta H2 para intervenir. Al cambiar
de etapa se reinicia AdamW en ambos brazos. Comparar efectos en los prefijos,
bloques reservados y cuentas H2 contra el control; una correccion aceptada
por el monitor no prueba que haya beneficio fuera de muestra. Si los cuatro
hibridos fallan no se asume que la causa sea el extractor.

Coordinador: tools/role432_campaign.py, trabajador: tools/role432_guard.py.
Evidencia local: /var/tmp/role432-prefix-repair. Seis pares repartidos entre
tres GPU; retener padres, hijos, sustituciones, pisos, curvas y replays.

# 431. Historia creciente y conservacion de comportamientos

Estado: orden de investigacion aprobado, sin ejecucion. Sigue al diagnostico
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

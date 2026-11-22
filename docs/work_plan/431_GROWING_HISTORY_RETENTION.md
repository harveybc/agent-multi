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

Resultado 432: seis parejas terminadas, 18 ampliaciones por brazo. Ocho
ampliaciones generaron un candidato rechazado para ablacion. En tratamiento,
12 de 32 hibridos respetaron todos los pisos y dos fueron adoptados al cierre
de etapa (ambos 2023/7, capa oculta: retornar 100% hacia el padre en 18 meses
y 50% en 24 meses). Las otras cinco parejas dieron igual retorno final entre
brazos. H2 2023/7: control -2,0774%, reparado -2,2840%. Ninguna cuenta positiva.
No promover. Esto demuestra reparabilidad local bajo MSE y pisos, no mejora
economica generalizable ni que la capa oculta explique toda la inestabilidad.

### 433: auditoria de las 12 reparaciones elegibles

La salida original no guardo los tensores de cada hijo rechazado. Se repitieron
solo las seis etapas necesarias en las GPU originales, sin cambiar datums,
politica ni semilla. Se exigio igualdad de historiales, registros de hibridos
y tensores del campeon de tratamiento antes de auditar. Sin H2 ni seleccion
posterior. Para cada hibrido se midio el decil semanal por horizonte en los
bloques reservados disponibles hasta la etapa (incluido el recien agregado).
Delta en pb de ventaja gestionada, ponderada por cantidad de ofertas.

|Referencia en la misma etapa y trayectoria|Mejor/igual/peor|Delta mediana pb|
|---|---|---:|
|Candidato rechazado sin editar|3/3/6|-0,004|
|Padre heredado|6/2/4|+0,016|
|Mejor admisible antes de editar|0/0/12|-0,173|

De las dos adoptadas (2023/7), la de 18 meses sustituyo 14 ofertas frente
al mejor admisible. Entrantes/salientes: -4,109/-2,978 pb; impacto medio
sobre el decil entero -0,049 pb. La de 24 meses sustituyo 11:
-9,000/-7,963 pb; impacto -0,025 pb. Son ofertas solapadas, no trades.
No son 12 replicas independientes. No adoptar esta regla de reparacion;
este contraste no descarta intervenciones diferentes. CSV por oferta y
RESULTS.json bajo /var/tmp/role433-repair-audit. Reproductor:
tools/role433_audit.py (mismo entorno y alias por variable que el 431).

### 434: seleccion de checkpoints sin reentrenar

Comando: `PYTHONPATH=. python tools/role434_campaign.py` en el entorno de
431, con ROLE431_SSH_TARGET definido. Dos colas remotas y una local puntuan
las celdas en sus dispositivos originales. CPU comprueba la alineacion del
monitor y, despues de sellar ambas selecciones, evalua los bloques reservados
y la cuenta H2 final. Sin Dragon, 2024 ni cambio automatico de pesos en vivo.
En el brazo sin reparacion, comparar padre, campeon y estados previos a
restauraciones, no todas las epocas. Se exige admisibilidad en los mismos
pisos de 432 y reproduccion del puntaje protegido del campeon original.

Menor MSE frente a mayor ventaja gestionada en el monitor, con cupo semanal
por horizonte: 15/18 etapas eligieron pesos diferentes. En las seis etapas
finales cambiaron las seis elecciones; solo dos mejoraron el retorno H2,
ninguna cuenta B fue positiva y 0/6 superaron la regla de integracion.
Se detuvo antes de entrenar otro scheduler. Referencia:
`/var/tmp/role434-monitor/{PROTOCOL,DONE}.json` y seis comparaciones por celda.

### 435-436: sonda dentro de semana y efecto del filtro de apertura

Secuencia reproducible: `bash tools/run_434_436.sh` con ROLE431_SSH_TARGET
configurado. Si 434 no califica ninguna celda, ajustar las sondas en tres
carriles GPU independientes, cada uno con dos cohortes/semillas. CPU prepara
los datos y reproduce las cuentas al recibir cada resultado. Si ninguna
sonda califica en las seis celdas, ejecutar automaticamente la ablacion
436 en CPU. No hay seleccion por H2, promocion ni espera de aprobacion.

Soporte de ajuste 432, representacion congelada; centrar X y ventaja
gestionada dentro de semana solo en TRAIN. Ridge 0,01 por observacion,
lineal o ampliada con 256 cosenos aleatorios fijos. Inferencia sin centrar
por semana futura; interceptos por horizonte en H1 purgado. Control nulo
por desplazamiento circular de etiquetas TRAIN por semana, control positivo
de recuperacion de un eje conocido en ajuste (no de prediccion financiera).

|Sonda|Supera azar / etiqueta desplazada en ventaja a cupo fijo|Ventaja absoluta positiva|Retorno mediano H2 / trades|Sin filtro de EV: retorno mediano / trades|
|---|---|---|---|---|
|Lineal|5 de 6 / 5 de 6|0 de 6|-7,95% / 264,5|-8,13% / 267,5|
|No lineal|5 de 6 / 5 de 6|0 de 6|-8,91% / 290|-8,91% / 290|

Tres de seis cuentas lineales y seis de seis no lineales fueron identicas al
retirar solo el requisito EV positivo. Ninguna cuenta positiva con o sin el
filtro. Se rechazan estas dos sondas, no toda la representacion ni todos los
modelos posibles. Evidencia por celda en /var/tmp/role435-within-week y
/var/tmp/role436-gate-replay. Dragon sigue fuera de esta campana.

### 437-438: atribucion SELL y contraste del cierre en cuenta

Sonda no lineal congelada de 435, H2 2022 y H2 2023, semillas 7, 19 y 43.
437 usa exactamente el cupo semanal por horizonte; control: esperanza de
una seleccion uniforme con igual numero de ofertas por semana y horizonte.
No son operaciones independientes ni una politica ejecutable de top semanal.
En 2023 se reutiliza el ledger 411; en 2022 se repite cada entrada con y sin
cierre aprendido. La suma de los componentes reproduce cada etiqueta con
error maximo inferior a 1e-12 pb. Se reproducen los cuatro promedios por
horizonte de cada semilla y los totales tanto de 435 como de su control.

SELL 12 h, medias por oferta en pb, promediando las tres semillas:

|H2|Ofertas elegidas/semilla|Bruto hasta barreras|Cambio bruto por cierre|Costes|Continuacion vs esperar|Total vs esperar|Control al azar|
|---|---:|---:|---:|---:|---:|---:|---:|
|2022|231|-1,15|-0,22|-2,48|+0,47|-3,38|-3,49|
|2023|206|+14,91|-10,43|-2,98|+0,20|+1,70|-2,66|

438 usa las ofertas causales de la cuenta 435, no el cupo retrospectivo.
Solo se inhibe el cierre voluntario; se mantienen barreras, salidas obligatorias,
costes y restriccion de una posicion. Antes y despues del contraste se
reproduce la cuenta original exactamente. Retornos porcentuales y operaciones:

|H2|Semilla|Gestionado|Barreras|Operaciones gestionado / barreras|
|---|---:|---:|---:|---:|
|2022|7|-10,07|-9,52|452 / 277|
|2022|19|-10,22|-4,81|424 / 262|
|2022|43|-11,30|-8,42|443 / 262|
|2023|7|-7,53|-10,56|136 / 95|
|2023|19|-7,75|-7,09|156 / 102|
|2023|43|-7,01|-11,07|144 / 106|

No adoptar ninguno. El bruto de las entradas explica el cambio de signo de
12 h entre cohortes; el cierre amortigua esa mejora, no la origina. El
efecto por oferta no se traslada directamente a cuenta: cambian los ticks
en los que vuelve a haber plaza. Proximo diagnostico: emparejar aperturas
comunes y contabilizar aparte entradas desbloqueadas y desplazadas.

Reproductores CPU: `tools/role437_sell_attribution.py` y
`tools/role438_close_control.py`, en ese orden, con el entorno de 435 y
`PYTHONPATH=.:tools`. Resultados completos:
`/var/tmp/role437-sell-attribution/DONE.json` (24 celdas) y
`/var/tmp/role438-close-control/DONE.json` (seis pares de cuentas).

### 439: que operaciones explican la diferencia de cuenta

Reproductor CPU: `tools/role439_trade_overlap.py`, mismo entorno que 438.
Las doce cuentas deben coincidir exactamente con 438; emparejar entradas SELL
por barra absoluta y verificar precio de entrada, SL y TP para las comunes.
Diferencia de retorno en puntos porcentuales, gestionado menos barreras. La
contribucion de las exclusivas de barreras es el negativo de su neto.

|H2|Semilla|Comunes / solo gestionado / solo barreras|Comunes|Solo gestionado|Solo barreras|Total|
|---|---:|---|---:|---:|---:|---:|
|2022|7|266 / 186 / 11|+3,94|-5,05|+0,56|-0,55|
|2022|19|247 / 177 / 15|-0,52|-4,20|-0,70|-5,42|
|2022|43|249 / 194 / 13|+0,52|-3,51|+0,11|-2,88|
|2023|7|89 / 47 / 6|+2,54|+0,65|-0,15|+3,03|
|2023|19|91 / 65 / 11|-2,18|-1,29|+2,81|-0,67|
|2023|43|99 / 45 / 7|+2,85|-0,16|+1,37|+4,06|

Antes de redondear, las tres columnas explican la diferencia de retorno con
error inferior a 1e-8 puntos por par. La parte comun incluye diferencias en
el tamano de posicion debidas al balance disponible. Una operacion exclusiva
no es necesariamente la reapertura inmediata tras el primer cierre diferente.
No extrapolar a BUY, a 2024 o al resultado de introducir un tiempo de espera.
Detalle por barra en `/var/tmp/role439-trade-overlap/{year}-{seed}.json`;
resumen verificable en `DONE.json` del mismo directorio. No se promueven pesos.

### 440-442: primera reapertura y control de actividad

Dentro de cada posicion compartida cerrada antes por el agente, clasificar
las ventas exclusivas hasta la salida por barreras inclusive: primera y
siguientes. El resto se contabiliza aparte. El neto de las tres categorias
reproduce la atribucion 439. Esto no entrega al controlador la barra de salida.

Intervenciones sobre cuenta completa sin reentrenar: 440 omite la siguiente
senal SELL elegible de la misma semana tras un cierre voluntario. 441 omite
solo la primera oferta si la posicion virtual mantenida a barreras sigue
ocupada al decidir; avanza con el motor nativo y no usa barras futuras.
Se mantienen todos los cierres del agente. Ninguna de las doce cuentas
intervenidas resulto positiva. Resultados H2 de retorno (%):

|Ano|Semilla|Primera reapertura retrospectiva: n / aporte (pp)|Original|Veto siguiente|Veto con posicion virtual|
|---|---:|---|---:|---:|---:|
|2022|7|82 / -2,13|-10,07|-10,95|-13,00|
|2022|19|85 / -3,23|-10,22|-9,94|-8,77|
|2022|43|85 / -5,30|-11,30|-8,87|-10,62|
|2023|7|29 / -1,14|-7,53|-5,83|-6,69|
|2023|19|26 / -0,96|-7,75|-3,92|-6,73|
|2023|43|26 / -1,71|-7,01|-6,82|-6,80|

442: 32 mascaras de azar x 5 tasas de veto x 6 celdas. Por replica se escoge
la tasa con cuenta de trades mas cercana sin mirar retornos; se excluyen
desajustes mayores al 5%. Se publica el numero util, retorno mediano y
fraccion de controles superados por cada brazo. Es un diagnostico retrospectivo
de actividad, no un test independiente ni una igualacion de exposicion monetaria.
La secuencia reproducible es `bash tools/run_440_441.sh` (440, 441 y 442);
cada etapa comprueba la referencia y solo continua si pasa. 442 guarda progreso
por bloque y termina en `/var/tmp/role442-activity-placebo/DONE.json`;
no promueve ninguna regla ni modifica live.

442 completo: de las 32 replicas por celda, 10-20 controles comparables
para 440 y 15-24 para 441. 440 supera la mediana
del control en 3/6 celdas y 441 en 2/6. No adoptar un veto general.

### 443: refit semanal del abridor lineal

Ultima etapa de `bash tools/run_440_441.sh`. Reconstruye el ridge lineal
435 y comprueba puntajes y cuenta antes de actualizar. Conserva el encoder,
normalizacion, intercepto H1, regla de elegibilidad y cierre. Cada lunes solo
incorpora semanas anteriores de H2: se comprueba que toda etiqueta y el
cierre forzoso semanal preceden al corte; se actualizan las ecuaciones de
ridge con centrado por semana y penalizacion 0,01 por observacion. Control
negativo con igual numero de filas y desfase circular de etiquetas en cada
semana incorporada. No es un reentrenamiento del encoder. No accede a 2024.

|H2|Semilla|Congelado %|Semanal %|Desfasado %|Trades del refit en todo H2|
|---|---:|---:|---:|---:|---:|
|2022|7|-6,42|-8,50|-10,67|288|
|2022|19|-13,59|-9,14|-10,70|314|
|2022|43|-8,25|-7,79|-11,44|298|
|2023|7|-6,54|-8,15|-5,79|144|
|2023|19|-9,56|-7,80|-8,00|160|
|2023|43|-7,64|-5,48|-6,94|150|

El refit mejora la cuenta congelada en 4/6, pero por oferta mejora en las
tres semillas de 2022 y empeora en las tres de 2023. Ninguna cuenta semanal
positiva; sin integracion, ni entrada a floating o DEAP por este resultado.
Ultima semana: 2400 filas incorporadas para 2022, 2166 para 2023.
Evidencia por celda y exposicion semanal en `/var/tmp/role443-weekly-refit`.

### 444-446: por que cambia la ventaja del refit semanal

Reproductor secuencial: `bash tools/run_444_446.sh`. Cada prueba reproduce
las tres ramas y las exposiciones de 443 exactamente, sin sobrescribir sus
recibos. Solo SELL en H2 2022/23, tres semillas por calendario, misma cuota
por semana y horizonte. Las ofertas se solapan, no son operaciones de cuenta.

444: entrantes y salientes identificadas por origen y horizonte. La diferencia
de sus etiquetas reconcilia el cambio de ventaja a cupo fijo. En cada fila
del resumen se agregan las tres semillas. Cambio medio por oferta del decil
(pb); las dos ultimas columnas son medias solo sobre las ofertas sustituidas:

|H2|3h|6h|12h|24h|Cambio total|Entrantes|Salientes|
|---|---:|---:|---:|---:|---:|---:|---:|
|2022|+0,48|+2,85|+1,20|+0,76|+1,32|-2,09|-5,45|
|2023|-0,98|-1,23|-1,73|-0,12|-1,01|-5,96|-0,03|

1093 de 2772 y 422 de 2472 ofertas sustituidas respectivamente. Omitiendo
una semana cada vez, el cambio de ventaja queda entre +0,92 y +1,56 en
2022 y entre -1,19 y -0,75 en 2023; no es un intervalo de confianza.
En 2023 la diferencia corresponde a -0,60 pb brutos hasta barreras, -0,41
por el cierre, +0,02 de costes y -0,02 de continuacion. En 12 h se cambian
108 ofertas con +3,11 pb por otras con -6,78 pb. El replay comprueba cada
oferta modificada contra la suma de sus componentes.

Factorial del ridge: actualizar solo la matriz regularizada de entradas y
su centrado, solo el producto entradas x etiquetas, ambos o ninguno.
Promediando las dos ordenaciones, el cambio atribuible a las etiquetas es
+1,09 / -0,83 pb (2022 / 2023) y el de las entradas +0,23 / -0,19.
Es una descomposicion del algoritmo; no identifica una causa macroeconomica.

445: comparar justo antes/despues de cada actualizacion, primero en el bloque
recien incorporado y luego en el siguiente bloque puntuado. Sin cambiar cuota.
2022 (25 actualizaciones por semilla): +11,30 / +0,89 pb; 2023 (24):
+5,63 / -0,07 pb. En 2023 el efecto posterior fue -0,51 / -0,18 / +0,49
por semilla. La fuerte ganancia dentro de muestra no se conservo.

446: prueba de cuenta sin ajustar umbrales. Dos politicas virtuales puntuan
las mismas ofertas. Antes de cada semana se usa el refit solo si su suma de
ventajas incrementales en semanas H2 ya maduras es positiva; si no, se usa
el abridor congelado. El cierre, costes y cupo de posiciones no cambian.
No es una nueva validacion independiente. Evidencias:
`/var/tmp/role444-weekly-swaps`, `/var/tmp/role445-weekly-transfer`,
`/var/tmp/role446-transfer-gate`.

|H2|Semilla|Semanas con refit / evaluadas|Retorno %|Operaciones SELL|Ventaja pb|
|---|---:|---|---:|---:|---:|
|2022|7|18/27|-9,26|304|-2,67|
|2022|19|24/27|-8,62|314|-2,99|
|2022|43|24/27|-7,69|298|-2,20|
|2023|7|4/25|-6,47|157|-0,48|
|2023|19|10/25|-10,92|163|-1,07|
|2023|43|9/25|-7,07|157|-1,01|

0/6 positivas. No promover la conmutacion ni usar estas semanas para elegir
otro umbral. Este resultado se refiere al readout lineal sobre el encoder
congelado, no a todos los especialistas ni a todo refit semanal.

### 447: segmentos disjuntos y monitor de retencion acumulativa

Contraste prospectivo del monitor del abridor SELL shared8, no de la cuenta
multiespecialista completa. Encoder fijo, mismos datos purgados de 431,
3 o 5 semestres segun cohorte. Un solo entrenamiento del primer semestre
por semilla/cohorte. Despues cada brazo hereda su campeon, momentos AdamW
y el LR de ese checkpoint. Se recalcula el valor del padre (epoca cero);
se reinician solo los contadores de etapa. Cada epoca visita una vez todas
las filas del semestre actual en minibatches de hasta 512. Ninguna fila
anterior entra en loss, replay o gradiente. Ambos brazos muestrean en el
mismo orden hasta donde ambos entrenan.

Desempeno P: ventaja gestionada media en pb de las ofertas SELL del decil
alto por semana y horizonte (no retorno porcentual de cuenta). Primer
semestre P(val). Control siempre P(val); intervencion desde el segundo:
0,5 P(val) + 0,5 media por semestre de P(modelo actual en cada semestre
anterior de fit). No incluye el semestre que se entrena. El monitor decide
campeon, parada y restauracion antes de reducir LR. Gracia LR 20, paciencia
20, factor 1/5, piso 4,8e-7; ES cuenta desde epoca 100, paciencia 80,
umbral 1e-8 y maximo 2000. Los bloques de auditoria por semestre no entran
en ningun monitor. Evaluacion de cuenta H2 separada, sin acceder a 2024.

|H2|Semilla|Retorno solo val / mixto (%)|Trades val / mixto|Epoca campeona final val / mixto|
|---|---:|---:|---:|---:|
|2022|7|-20,98 / -17,23|513 / 532|13 / 0|
|2022|19|-19,79 / -16,45|527 / 529|42 / 2|
|2022|43|-13,00 / -12,86|500 / 503|0 / 0|
|2023|7|-3,20 / -3,16|144 / 145|44 / 0|
|2023|19|-4,24 / -3,92|136 / 123|11 / 8|
|2023|43|-7,22 / -7,26|256 / 258|0 / 0|

Todas las 42 etapas/brazos (contando una sola vez las seis primeras
compartidas) pararon por paciencia: 179-253 epocas y 88 restauraciones por
LR en total. Al comparar hijo vs su propio padre, en cada bloque anterior
reservado por separado: control 7 mejoras / 20 empates / 12 retrocesos;
mixto 7 / 25 / 7, tolerancia 1e-8 pb. Peor retroceso -4,49 vs -2,13 pb.
En cuatro de seis ultimas etapas mixtas quedo el padre, no un hijo nuevo.
Retener mas no demuestra convergencia, no regresion por bloque ni beneficio
operativo. No promover.

Ejecutar `tools/role447_campaign.py` con el entorno y alias de 431;
dos carriles en Gamma, uno en Omega, Dragon reservado. Registro por epoca,
padre de etapa elegible, pesos y AdamW por campeon, y recibos en
`/var/tmp/role447-segment-only`. La fuente del monitor es la sonda de
ventaja relativa; un contraste de monitor basado en retorno de cuenta
completo seria una prueba distinta.

# 448. Especialista temporal OPEN_SELL: ocho segmentos y validacion anual

Contrato antes del entrenamiento. Se conserva el objetivo supervisado SELL por
horizonte de 447; no se reabre el actor SAC monolitico. La red aprende desde
ventanas reales, no desde el cache de 208. La receta de seleccion y su scaler
fixed_train se mantienen fijos; no es una prueba independiente de esa seleccion.

| Requisito | Prueba que debe pasar |
|---|---|
| 2020-23 TRAIN, ocho semestres; 2024 validacion, no test | Relojes exactos, ventanas validas, ninguna ventana ni liquidacion cruza el limite de su bloque; conteo por semestre, sin reservar los ultimos 28 dias |
| Datos de mercado reales | Identidad de receta y recibos de etiquetas; cotejar origen, barras, probabilidades y cabezas con productor |
| Encoder entrenable | Ventana 13 x 61, 22 grupos; gradiente finito no nulo y cambio de pesos por rama, fusion, atencion, nucleo y head durante un paso real |
| Solo bloque nuevo en loss | El sampler indexa unicamente el tensor local con etiquetas finitas del bloque actual, sin reemplazo por epoca |
| Continuidad | Al pasar de etapa cargar pesos y AdamW del campeon, LR incluido. Etapa 0 comun; desde etapa 1 dos brazos independientes; candidato padre elegible |
| Comparacion justa | Semillas 7,19,43; mismo orden de ejemplos entre brazos por etapa; maximo 2000 epocas, batch 512, AdamW 3e-4 inicial, decay .001, umbral 1e-8, politica 405 de paciencia y restauracion |
| Metrica | Replay nativo SELL-only con cierres aprendidos fijos y barreras. Retorno % sobre capital inicial por bloque. Etapa inicial y control: validacion. Mixto: 0,5 validacion + 0,5 media de retornos de los semestres previos, igual peso por semestre, sin gradiente |
| Ejecucion | Gate de apertura positivo y percentil 90 por horizonte, distribucion estimada sobre el semestre que se entrena; congelada para evaluar el checkpoint. No usar 2024 para ajustar percentiles. Un H2 en blanco no sustituye un ano de validacion |
| Reproducibilidad | Paridad del simulador empaquetado sobre referencia antes de optimizar. Historia, conteos, checkpoint de etapa y estado de reanudacion. Sin promocion ni live |

El encoder de aperturas arranca de pesos nuevos en el primer semestre para
no importar un checkpoint elegido con 2024. Los modelos historicos de cierre
y la receta son instrumentos fijos del experimento de desarrollo, no modelos
reentrenados o calificados aqui.

## Ejecucion

Preparador: `tools/role448_prepare.py`. Compara origen, elegibilidad, sello
de las etiquetas y salidas del productor; el cache antiguo de 208 se usa solo
para reconstruir el simulador de referencia, nunca como entrada del abridor.
Los semestres contienen 2208, 2462, 2387, 2487, 2178, 2487, 2474 y 2216
ventanas. Total 18899; 2024: 4899. La primera fecha efectiva de TRAIN es
2020-01-20 10:00 UTC; validacion 2024-01-03 07:00 a 2024-12-31 08:00 UTC.
No son cinco anos de ticks ininterrumpidos. Se usa soporte de ventanas
finitas y completas, entradas admitidas y sesiones liquidables en el bloque.

Worker: `tools/role448_worker.py`. La prueba desechable de gradiente y paso
AdamW se revierte antes del ajuste real. 25118 parametros; 22 ramas,
fusion, atencion y nucleo temporal, mas cabezal SELL 208-8-12. La politica
puede abstenerse. No se acepta una epoca sin operaciones en validacion como
nuevo campeon; si no se alcanza actividad se informa, no se promueve.
El padre es candidato al entrar en cada etapa. Minibatches de hasta 512.
Conserva un estado de reanudacion cada diez epocas, pesos, momentos e historia
por etapa, y emite al final graficas por semilla/etapa/brazo. El monitor no
consulta retornos de segmentos futuros de TRAIN ni usa 2024 en el loss.
El scaler y la lista de entradas son los de la receta previamente congelada.

Coordinador: `tools/run_448_campaign.py` con el entorno de 447 y
`ROLE431_SSH_TARGET` configurado. Dos trabajadores en Gamma y uno en Omega.
Estado en `/var/tmp/role448-temporal-4plus1/run-{7,19,43}/PROGRESS.json`;
el mismo directorio contiene `PREFLIGHT.json` con identidad de GPU, paridad
de las cinco cuentas nativas y norma de gradiente y desplazamiento por grupo.
Los informes de Gamma se recogen en Omega cada 15 segundos. DONE a nivel de
campana solo despues de reunir los dos brazos finales de las tres semillas.

## Resultado final

Ocho semestres, seis campeones finales y 45 graficas de etapa; unos 29,5
minutos entre el primer PREFLIGHT y DONE. Las 45 etapas/brazos terminaron
en la epoca 179 por paciencia (primera parada posible). Resultados de
campeones restaurados en la misma cuenta anual 2024, solo SELL:

|Semilla|Monitor|Retorno %|Trades|Ganadas|Sharpe|DD %|Epoca campeona final|
|---|---|---:|---:|---:|---:|---:|---:|
|7|Validacion|-2,988|175|65|-0,51|6,06|39|
|7|Mixto|-3,915|209|72|-0,65|8,01|0|
|19|Validacion|+0,593|3|1|0,78|0,17|0|
|19|Mixto|+0,491|10|5|0,64|0,27|0|
|43|Validacion|-0,013|10|4|-0,01|0,69|0|
|43|Mixto|+0,056|1|1|0,90|0,10|0|

0/6 con retorno positivo y mas de dos operaciones por semana a la vez.
En semilla 7, mixto, los campeones s6 y s7 tienen pesos identicos tensor
por tensor. Su retorno 2024 pasa de +2,098% (99 operaciones) a -3,915%
(209). Ya ocurre en la epoca cero de s7: cambia el semestre usado para
la distribucion empirica de scores, que decide tanto admision como ranking.
No atribuir esa diferencia a olvido de los pesos. Contraste recomendado:
fijar pesos y comparar referencias antigua/nueva, luego fijar la referencia
y comparar pesos; verificar por barra los cambios de elegibilidad y horizonte.

### 449: cruce de politica sin reentrenamiento

Tres semillas x dos monitores x siete cambios de semestre = 42 transiciones
de la misma validacion 2024, no 42 replicas de mercado. Cruce de las dos
redes con las dos tablas empiricas de puntuacion (tabla del padre en su
semestre, tabla del hijo en el nuevo). Un quinto replay usa el padre con su
propia tabla medida en el nuevo semestre para verificar tambien epoca cero.
Reproducidos exactamente los diccionarios de cuenta del padre, hijo y epoca
cero y los cuatro percentiles originales en todos los casos. La preparacion
de la red se hace fuera de inference_mode, como en el productor 448.

|Semilla 7, paso 7 a 8: retorno % / ventas|Padre completo|Solo tabla nueva|Solo pesos nuevos|Hijo completo|
|---|---:|---:|---:|---:|
|Mixto|+2,10 / 99|-3,92 / 209|+2,10 / 99|-3,92 / 209|
|Validacion|+1,51 / 98|-4,02 / 205|+0,74 / 76|-2,99 / 175|

En el mixto hay 239 barras recien habilitadas, 0 deshabilitadas y 0 cambios
de horizonte entre barras habilitadas en ambas configuraciones. En cuenta:
118 entradas nuevas y 8 anteriores desplazadas, emparejando por barra de
apertura; algunas propuestas no son ejecutables con una posicion abierta.
25/42 transiciones conservaron los tensores de pesos; en 4 de esas 25
cambio el retorno al recalibrar. En la ultima etapa de las semillas 19 y 43
el cruce no cambia el retorno en ninguno de los dos brazos; este hallazgo no
explica por si solo su escasa actividad. En todas las transiciones, promediando
el efecto marginal de cada factor en ambos ordenes y tomando luego su valor
absoluto: 1,20 puntos porcentuales por tabla y 0,58 por pesos.

Reproductor: `tools/run_449.py` en el entorno de 448, con el mismo alias
remoto. `/var/tmp/role449-cross/DONE.json` contiene los cinco replays y
conteos por transicion. Las 45 tablas ordenadas por horizonte se guardan en
`/var/tmp/role449-cross/policies-{seed}/` con SHA del checkpoint de 448 y
manifiesto fuente. Se valida que al recargar cada tabla las pujas de 2024
sean identicas barra por barra. Restaurar la pareja (pesos, tabla); recalibrar
es una intervencion distinta y no implica promocion automatica.

### 450: entrenar heredando la politica completa

Contraste con el resultado archivado de 448. Misma muestra, orden de
minibatches, niveles de LR, optimizador, reglas de actividad y dos monitores.
El primer semestre es identico por construccion: se importa el checkpoint
comun de 448 con AdamW y se verifica el SHA contra la tabla completa de 449.
El replay debe reproducir su cuenta 2024 y los cuatro cortes exactos.
Los siete semestres siguientes si se entrenan por separado en ambos brazos.

1. En epoca cero, cargar pesos, momentos, LR y tabla empirica del campeon.
   Exigir igualdad de la cuenta de validacion, de todos los bloques de
   retencion ya evaluados y de los cortes de admision. Con un bloque de
   retencion adicional la metrica mixta puede cambiar aunque el padre sea
   identico; comparar sobre los mismos bloques de la etapa entrante.
2. Sin tocar pesos, estimar otra tabla en el nuevo semestre. Registrar ambas
   cuentas. Aceptar la recalibracion solo si califica y supera al padre en
   la metrica vigente por mas de 1e-8; en empate conservar el padre.
3. Continuar el entrenamiento del semestre actual como en 448. Cuando gane
   un hijo conservar juntos sus pesos, AdamW y tabla completa. No buscar
   varias tablas por hijo. En cada restauracion por LR verificar que la
   politica recupera los resultados del campeon antes del proximo gradiente.
4. Al final, comparar cada campeon con 448 por semilla y monitor. Separar
   mejora de retorno y cumplimiento de retorno positivo con mas de dos
   operaciones por semana. No promover automaticamente ni pasar a floating.

Ejecucion: `tools/run_450_campaign.py` con el mismo entorno y alias que 448.
El coordinador distribuye tres semillas sobre las GPU originales, recoge
resultados y genera curvas y `COMPARISON.json`. Evidencia local:
`/var/tmp/role450-full-policy/run-{7,19,43}`; los recibos `*-entry.json`
separan la cuenta heredada de la candidata recalibrada en la epoca cero.

### Resultado 450

33,36 minutos de protocolo a comparacion. Las 42 etapas entrenadas despues
del semestre comun pararon por paciencia en la epoca 179. Los seis campeones
de la ultima etapa fueron padres de epoca cero sin recalibracion adoptada.
Retornos sobre la validacion 2024, no rendimientos live:

|Semilla|Monitor|448 %|450 %|Ventas / ganadas|Ventas/semana|Sharpe|DD %|
|---|---|---:|---:|---:|---:|---:|---:|
|7|Validacion|-2,99|+1,56|56 / 21|1,06|0,61|1,92|
|7|Mixto|-3,92|+1,13|57 / 21|1,08|0,40|2,40|
|19|Validacion|+0,59|+0,59|3 / 1|0,06|0,78|0,17|
|19|Mixto|+0,49|+0,49|10 / 5|0,19|0,64|0,27|
|43|Validacion|-0,01|+2,98|41 / 21|0,77|1,09|1,29|
|43|Mixto|+0,06|+1,76|14 / 7|0,26|0,80|0,66|

Cuatro mejoras y dos empates contra 448. Ningun campeon final positivo y
con mas de dos ventas por semana. La semilla 7 validacion si mantuvo un
campeon intermedio de +1,025%, 313 ventas, Sharpe 0,196 (etapas 3 a 5;
no son replicas). En la transicion a etapa 6 la sola recalibracion paso a
+1,376% con 50 ventas y fue aceptada; luego el campeon entrenado alcanzo
+1,559% con 56 ventas. El selector exigia al menos una operacion, no el
piso de actividad de la calificacion final. No se promueve el intermedio.

### 451: elegibilidad por actividad antes de elegir al campeon

Contraste dirigido en el primer punto conocido de fallo: semilla 7, monitor
de validacion, padre exacto de 450 al final del semestre 5. Solo se reentrenan
los semestres 6, 7 y 8; no es una replicacion desde cero. La regla exige
ventas > 2 x semanas del panel completo (107 o mas en 53 semanas, promedio
anual, no minimo por semana). Entre elegibles gana el mayor retorno con
mejora > 1e-8. No se premia el volumen ni se fuerza ninguna apertura.
Se aplica tanto a la recalibracion de epoca cero como a cada epoca; se
verifican las cuentas del campeon tras las restauraciones. La paciencia,
el calendario de LR, los minibatches y las ventanas son los de 450.

Ejecutor: `tools/run_451_activity.py`. El protocolo sella el SHA del padre.
El primer testigo exige reproducir las cuentas de 313 y 50 ventas de 450
antes de entrenar. La salida de cierre en `/var/tmp/role451-activity/DONE.json`
compara las cuentas por semestre e indica si se supero realmente al padre;
los PNG muestran retorno de validacion, retorno de bloques anteriores y
actividad por epoca. Conservar un padre elegible no demuestra aprendizaje
ni retencion del rendimiento de TRAIN; no autoriza promocion.

### Resultado 451 y atribucion

Semilla 7, OPEN_SELL. Los tres semestres terminaron por paciencia en epoca
179, con una restauracion por semestre. En validacion 2024:

|Semestre|Epoca elegida|Retorno %|Ventas / ganadas|Ventas/semana|Sharpe|DD %|
|---|---:|---:|---:|---:|---:|---:|
|6|0|+1,0245|313 / 112|5,91|0,196|4,89|
|7|22|+1,0412|127 / 51|2,40|0,249|4,57|
|8|0|+1,0412|127 / 51|2,40|0,249|4,57|

El control 450 final tenia +1,5586%, 56 ventas y 1,06/semana. La regla
451 conserva el piso, no maximiza el retorno sin restricciones. La mejora
frente al padre elegible es +0,01666 puntos porcentuales (USD 1,67 sobre
10.000); no es evidencia de mejora reproducible. En el cierre de etapa 8
los siete paneles de TRAIN anteriores dan retornos negativos, entre -23,02%
y -1,35%. No se promociona.

Con `tools/role451_cross.py` se cruzaron los pesos y las referencias
congeladas de los campeones 6 y 7, con las mismas 4.899 decisiones
disponibles de validacion. Las dos diagonales reprodujeron exactamente las
cuentas guardadas, incluidos los conteos:

|Pesos|Tabla|Retorno %|Ventas|
|---|---|---:|---:|
|6|6|+1,0245|313|
|6|7|-0,1426|148|
|7|6|-4,0308|271|
|7|7|+1,0412|127|

El efecto depende del emparejamiento. No atribuir el incremento a mejores
pesos de forma aislada ni intercambiar tablas entre checkpoints.

### 452: contraste archivado, semillas 19 y 43

Sin reentrenar: brazo de validacion 450, ocho campeones por semilla y
las siete transiciones adyacentes, cuatro cuentas por transicion. Gamma
5090 para 19, Omega 4070 para 43; mismas GPU que en entrenamiento.
Las 28 diagonales reprodujeron los diccionarios contables completos.
Diez transiciones no cambiaron ni pesos ni tabla; no se cuentan como
replicas. Resultados en las cuatro que cambiaron ambos, retorno % sobre
validacion 2024 (entre parentesis, ventas):

|Semilla / semestre de destino|Padre|Solo tabla nueva|Solo pesos nuevos|Ambos nuevos|
|---|---:|---:|---:|---:|
|19 / 4|+0,491 (10)|+0,491 (10)|+0,593 (3)|+0,593 (3)|
|43 / 3|+1,953 (13)|-20,732 (676)|-0,633 (5)|+1,992 (36)|
|43 / 6|+1,992 (36)|+0,117 (8)|+0,754 (89)|+2,042 (13)|
|43 / 7|+2,042 (13)|+0,754 (89)|-0,053 (10)|+2,980 (41)|

En las tres transiciones de 43 los dos cruces fueron peores que ambas
parejas originales. En 19 la tabla no cambio las metricas agregadas
para ninguno de los dos juegos de pesos. No es una replica directa
de 451: ninguno de estos campeones originales cumple >2 ventas/semana.

En los ocho historiales por semilla (excluyendo epoca cero) hubo 1.432
evaluaciones cada una, no muestras independientes. Para 19, 163 pasaron
actividad y 589 fueron positivas; ninguna cumplio ambas (mejor activa:
-3,0132%, 115 ventas). Para 43, 190 pasaron actividad y 1.106 fueron
positivas; ninguna cumplio ambas (mejor activa: -0,1960%, 168 ventas).
Un selector posterior sobre estas epocas no puede devolver un candidato
activo y rentable. Esto no prueba que reentrenar con otra trayectoria sea
inutil. No se promociona ningun modelo.

Reproduccion: `tools/role451_cross.py 19` o `43`, con el dispositivo y el
entorno determinista de 450. Cuentas completas en `/var/tmp/role452-cross/`.

### 453: las 13 y 676 ventas, semilla 43

Reproduccion diagnostica 2024 con pesos del semestre 2 fijos. La tabla
3 pertenece a otro checkpoint; esta combinacion cruzada no es el campeon
emparejado del semestre 3. En 4.899 ventanas se cruzaron por separado
el umbral efectivo max(0, percentil 90) y el rango empirico usado en la
subasta. Las dos cuentas originales coinciden con 452; cambiar solo
el rango no modifica ninguna de las dos cuentas.

|Umbral 12 h|Tabla de rangos|Senales|Ventas / ganadas|Retorno %|DD %|
|---:|---|---:|---:|---:|---:|
|0,03987317|2 o 3|25|13 / 7|+1,953|1,156|
|0|2 o 3|1.504|676 / 212|-20,732|21,410|

Ninguna senal de 3, 6 o 24 h supero el umbral. Las 25 originales siguieron
admitidas; se incorporaron 1.479 de 12 h. Simulando cada apertura sola,
con el mismo capital inicial y el mismo cerrador, las nuevas tuvieron
451 ganancias y -3,107 USD netos de media; la correlacion de Spearman
entre rango y neto fue -0,00149. Las 25 originales aisladas tambien
perdieron en promedio (-2,886 USD). No sumar esos escenarios solapados
como si fueran una cuenta ni inferir una capacidad predictiva general.

En las cuentas secuenciales hubo cinco aperturas comunes. Las 671
exclusivas del umbral nuevo aportaron -2.245,12 USD y las cinco comunes
+171,97 USD. Las 676 sumaron -163,74 USD brutos y -2.073,16 netos;
559 cierres tempranos sumaron -4.267,15 netos. Este desglose por salida
no mide si HOLD habria sido mejor. Antes de tocar el cerrador, comparar
cierre vs protecciones del broker desde esas mismas posiciones.

Codigo: `tools/role453_sell_audit.py`; detalle:
`/var/tmp/role453-sell-audit/{trades.csv,isolated.csv,DONE.json}`.
No se ajustaron umbrales con 2024 ni se promovio una politica.

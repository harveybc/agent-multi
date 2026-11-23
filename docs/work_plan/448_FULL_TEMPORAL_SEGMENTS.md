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

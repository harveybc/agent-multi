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

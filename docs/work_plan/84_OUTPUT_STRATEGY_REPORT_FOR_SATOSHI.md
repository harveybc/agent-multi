# Informe de salidas, estrategia y seleccion nativa

Fecha: 2026-10-26. Autor: Musashi. Destinatario: Satoshi y propietario.
Entrega: estudio ejecutado y cerrado; no es una declaracion de rentabilidad.

**Dictamen independiente recibido:** contrato y reproducibilidad aceptados;
uso de DEAP y de los subconjuntos para decidir utilidad/calibracion rechazado.
Ver [dictamen de Satoshi](../audits/SATOSHI_AUDIT_VERDICT_MUSASHI_OUTPUT_STRATEGY_2026_10_26.md)
y [correcciones y plan de cualificacion](91_CROSS_AUDIT_CORRECTIONS_AND_LEARNING_CONTROL.md).
No usar estos numeros para ordenar la poda ni escoger el fitness o los SL.

## 1. Dictamen y alcance

Se reconstruyo el recorrido completo desde la salida neuronal hasta la orden,
el cierre, la contabilidad, la recompensa y el score del wrapper. Se ejecutaron:

- 22 escenarios sinteticos con el adaptador categorico actual y la
  configuracion economica exacta del experimento, no los defaults del motor.
- Dos mercados sinteticos distintos, cada uno con salidas ideales, invertidas
  y NOP: seis simulaciones completas, con costes.
- Un barrido exhaustivo de 27 configuraciones de TP, SL y volumen sobre las
  tres politicas TCN congeladas: 81 simulaciones de TRAIN, sin ajustes de pesos.
- Reproduccion exacta de trades y equity diaria de los tres controles.
- Ampliacion: 162 simulaciones de calibracion sobre salidas artificiales y
  seis comprobaciones en un tercer mercado sintetico, descritas en 9.4.
- 100 pruebas focales en CPU: 43 del adaptador y 57 del motor. El cierre contiene 329 archivos autenticados;
  una lectura posterior verifico todos sus hashes.

**Resultado:** el contrato puede ejecutar correctamente las acciones
simuladas y distinguir senales ideales de invertidas. Las politicas reales
ensayadas NO satisfacen el conjunto de comportamientos exigido. Ninguna de las
27 configuraciones resulta elegible en las tres semillas. El resultado del
barrido es `NO_FEASIBLE_CANDIDATE`, no una estrategia recomendada.

Esto no demuestra que las entradas carezcan de informacion ni que TCN no
sirva. Tampoco convierte una simulacion con conocimiento del futuro en
capacidad predictiva. La seleccion financiera no recibe un RELEASE por este
informe.

## 2. Identidades y fuentes

| Elemento | Version examinada |
|---|---|
| Productor de fits del wrapper | agent-multi `7bfb1047` |
| Motor de ejecucion | heuristic-strategy `cd6489f`, `event_costs_v2` |
| Selector | feature_selector `bec66cca`, version 0.2.1 |
| Spec del experimento | `05c06cb508b3e7bb9e1c33064c9dda4cd524671db62dd78d7defc5539da20f4d` |
| Cierre del estudio de estrategia | `6e395fb29f5594d062a18539dfd50142a6795196835d2480d94ec728ad1da160` |
| Auditoria de codigo consumido | 205 archivos agent, 56 motor y 13 selector; cero diferencias respecto del spec |

Fuentes ejecutables principales:

- [Construccion del modelo y entrenamiento real](../../tools/rl_subset_bridge.py).
- [TCN agrupada](../../agent_plugins/grouped_tcn_orders.py).
- [Distribucion categorica, riesgo y valores](../../agent_plugins/hybrid_orders.py).
- [Adaptador de acciones al motor](../../env_plugins/hybrid_orders.py).
- [Evaluacion sin modificar pesos](../../tools/recipe_sac_queue.py).
- [Orquestacion y score de la seleccion](../../tools/rl_selection_run.py).
- [Subset estructural y soporte emparejado](../../app/rl_subset.py).
- [Motor fijado](https://github.com/harveybc/heuristic-strategy/blob/cd6489f/app/short_horizon/expert_guided/env.py).
- [Sizing, protecciones y contabilidad](https://github.com/harveybc/heuristic-strategy/blob/cd6489f/app/short_horizon/execution.py).
- [Floating backward](https://github.com/harveybc/feature_selector/blob/bec66cca/src/feature_selector/search.py).
- [Evaluador financiero del selector](https://github.com/harveybc/feature_selector/blob/bec66cca/src/feature_selector/evaluators.py).

La nueva comprobacion de cierres ejecutados y el barrido viven en
[native_strategy_contract.py](../../app/native_strategy_contract.py) y
[native_strategy_contract_study.py](../../tools/native_strategy_contract_study.py).
No se modificaron los fuentes, pesos ni resultados de la corrida congelada.
La distincion entre su compuerta antigua y la comprobacion estricta de esta
entrega se explica en la seccion 10; no se afirma que sean la misma version.

## 3. Que modelo produce las salidas

El wrapper actual usa **PPO categorico**, no SAC, DEAP ni aprendizaje con experto.
SAC aparece en nombres historicos y en metadata heredada `exec_contract=sac`;
la autoridad para identificar el algoritmo es `build_model`, que crea `PPO`.
No se usa ZigZag como entrada, etiqueta o recompensa.

Entrada inicial: 120 columnas, 31 grupos, 13 observaciones cada dos filas de
mercado, incluyendo la barra completada actual. No significa que solo se
decida cada dos horas: se intenta decidir en cada barra horaria con ventana
valida. Los dos pasos pertenecen al muestreo de la ventana.

Hay 28 ramas convolucionales causales y tres puntuales. La fusion tiene ocho
canales. El nucleo TCN usa dos bloques residuales, cada uno con dos
convoluciones kernel 3, dilataciones 1 y 2 respectivamente. Despues se conserva
la secuencia completa mediante flatten y se aplican cabezas separadas de
politica y valor, de 16 unidades Tanh cada una. Flatten no elimina valores,
aunque esa ultima parte ya no impone equivarianza temporal.

La escala es fija, calculada en TRAIN; no hay z-score movil, clipping de las
entradas, demostraciones ni vector de estado de cuenta en el extractor.
Las columnas con una operacion DIFF declarada conservan esa operacion. Las
caracteristicas relativas seleccionadas siguen siendo columnas explicitas,
no un reemplazo silencioso de la normalizacion.

### Salidas exactas

| Salida | Dimensiones | Uso real |
|---|---:|---|
| Logits de comandos | 5 | Una distribucion categorica enmascarada |
| Parametros de riesgo | 12 = 2 direcciones x 3 variables x 2 parametros Beta | La capa existe, pero `learned_risk=False`: no decide TP/SL/volumen ni contribuye a la verosimilitud de riesgo |
| Valores del critico | 3 | Valor de estado para ruta flat, largo o corto; no son Q-values de cinco acciones |

Los cinco comandos son:

| Codigo | Comando | Significado |
|---:|---|---|
| 0 | NOP | Estando flat, no abrir; con posicion, HOLD |
| 1 | OPEN_LONG | Solicitar una nueva compra |
| 2 | OPEN_SHORT | Solicitar una nueva venta |
| 3 | CLOSE_LONG | Solicitar cierre anticipado de una compra |
| 4 | CLOSE_SHORT | Solicitar cierre anticipado de una venta |

El tensor almacenado por SB3 es `[comando, u_tp, u_sl, u_size]`. Es un codec
Box, no una distribucion gaussiana de clases. El comando debe ser un entero
exacto; 1.2 no significa compra ni se redondea: se rechaza. En esta corrida
los tres ultimos campos son 0.5; el adaptador de riesgo fijo los interpreta
segun la regla que se describe abajo.

Durante entrenamiento se muestrea `Categorical`. Durante evaluacion se usa
`argmax` sobre sus probabilidades legales. Este contraste importa: probabilidades
que cambian pueden producir siempre el mismo comando por argmax.

No existen cinco agentes entrenados por separado ni un arbitro neuronal
adicional. NOP es una clase del mismo actor. No recibe etiquetas humanas de
"no operar": aprende por recompensa y por las restricciones ejecutables.

## 4. Mascara y estados de la estrategia

| Estado real del motor | Mascara en orden NOP/L/S/CL/CS |
|---|---|
| Flat y entrada admisible | `[1,1,1,0,0]` |
| Flat y entrada prohibida | `[1,0,0,0,0]` |
| Compra abierta | `[1,0,0,1,0]` |
| Venta abierta | `[1,0,0,0,1]` |

Los logits ilegales se sustituyen por menos infinito antes de crear la
distribucion. La mascara no entra al extractor temporal: restringe acciones
y escoge la salida de valor. Por tanto, "sin realimentacion" significa sin
vector aprendido de balance, PnL, edad o distancias; **no significa ausencia
total de informacion del estado**, pues la ruta legal depende de la posicion.

Solo hay una posicion simultanea. No hay piramidacion, incremento parcial,
cierre parcial, trailing stop ni modificacion de SL/TP con la posicion abierta.
El motor soporta reversal, pero este adaptador lo impide mediante la mascara:
para invertir hay que cerrar y, en otra decision admisible, abrir el otro lado.
El limite diario puede impedir esa segunda apertura.

El cierre anticipado es una accion explicita disponible aunque el cupo de
entradas del dia este agotado. No espera a que el SL sea tocado. El modelo no
emite una probabilidad supervisada de "SL antes que TP": elige CLOSE o HOLD
segun lo aprendido con la recompensa de equity.

## 5. Cronologia completa de una decision

1. Se completa la barra b. La entrada puede usar b, nunca una observacion
   futura. Los datos llevan disponibilidad `inicio(b)+1h`.
2. Si existe una ventana valida, se evalua el actor y se aplica la mascara.
3. El comando se entrega al motor. Una apertura o cierre voluntario se ejecuta
   como pronto en `OPEN[b+1]`, no al cierre conocido de b.
4. Si ya hay posicion, las protecciones atravesadas por el gap de apertura
   tienen prioridad sobre la instruccion del agente.
5. Se ejecuta el cierre voluntario; para una entrada se vuelven a comprobar
   calendario, cupo, volatilidad, sizing, niveles y distancia al precio real.
   Una mascara admisible no garantiza un fill: un gap aun puede rechazarlo.
6. Se simulan SL/TP y margin close-out dentro de la barra siguiente. Si SL y TP
   son tocados sin una trayectoria intrabar conocida, prevalece SL.
7. Se comprueba margen al cierre, incluyendo financiacion, y el cierre forzado
   por calendario o fin del periodo.
8. Se calcula la recompensa usando la variacion de equity con costes.

Si falta una ventana, no se fabrica una observacion. El motor avanza con HOLD
automatico, conservando protecciones, financiacion, calendario y contabilidad.
Al terminar el periodo se liquida la posicion; no se conserva entre episodios.

Precision documental: la metadata heredada dice "forced flat has priority".
El orden concreto del motor es protecciones, margen y despues forced flat;
el cierre forzado no deshace un SL/TP ya ejecutado. Este informe describe ese
orden ejecutable, no la frase abreviada.

## 6. Reglas de negocio y parametros efectivos

| Regla | Valor efectivo |
|---|---|
| Balance inicial | USD 10,000 |
| Apalancamiento | 1:50 |
| `rel_volume` | 0.1 del balance actual, base margen |
| Fraccion de orden en control | 1.0 de ese limite |
| Minimo de unidades | 1; supuesto de ejemplo, no verificado contra una cuenta concreta |
| Entradas maximas | 1 por dia de negocio, no necesariamente dia UTC |
| Actividad evaluada | >=2 ordenes/semana como promedio del periodo, no minimo de cada semana |
| Despues de reapertura | 12 horas sin abrir |
| Antes de cierre de mercado | Flat desde 12 horas antes |
| Corte adicional previo al flat | 0 horas; el motor tambien exige margen temporal para la barra de ejecucion |
| Distancia minima | 0.00005 en precio EURUSD |
| Distancia efectiva maxima | 1.5 veces la escala causal de rango |
| Margin close-out | Equity <= 50% del margen usado |

No se obliga a abrir para superar la regla de actividad. El agente puede NOP;
en ese caso la corrida no es elegible para la comparacion que exige actividad.

### TP y SL del control

Sea sigma la volatilidad causal conocida al decidir, y rho la escala causal de
rango. Sigma usa la raiz de la suma de 24 cambios de cierre al cuadrado,
promediada en 480 filas con minimo 120, y desplazada una fila. Rho promedia
rangos de 24 filas de igual forma. Son escalas del motor, no z-score de entradas.
El indice b+1 de estas series ya desplazadas solo contiene barras hasta b.

Se intersectan los rangos TP `[0.25,3]` y SL `[0.5,3]`, en multiplos de sigma,
con la banda ejecutable `(min_distance, 1.5*rho)`. Hay margenes numericos de
aproximadamente 1e-9 para evitar errores de frontera. Si la interseccion es
vacia, no se abre.

En el control, cada multiplo es el **punto medio de su interseccion**, no
siempre 1.625 o 1.75. Los niveles se construyen respecto del cierre de b:

```text
TP = close[b] + direccion * multiplicador_TP * sigma
SL = close[b] - direccion * multiplicador_SL * sigma

COMPRA: TP = close[b] + multiplicador_TP * sigma
        SL = close[b] - multiplicador_SL * sigma
VENTA:  TP = close[b] - multiplicador_TP * sigma
        SL = close[b] + multiplicador_SL * sigma
```

Tras conocer la apertura siguiente, el motor rechaza la entrada si el precio
queda fuera de SL/TP o demasiado cerca. No desplaza el fill retrospectivamente.
`level_policy=clip` acota niveles de orden al contrato; no recorta datos de
entrada. El adaptador calcula ya una interseccion factible, por lo que los
testigos normales verifican niveles efectivos sin depender de ese recorte.

### Volumen

```text
capacidad = floor(balance_actual * 0.1 * 50 / precio_de_ejecucion)
unidades = floor(fraccion_orden * capacidad)
margen = unidades * precio_de_ejecucion / 50
```

La comprobacion previa usa el cierre disponible; el calculo de unidades se
realiza al precio de ejecucion. Se rechaza si no alcanza el minimo o excede
el margen. El control no aprende riesgo: usa fraccion 1.0. El barrido cerrado
de esta entrega explora 0.2, 0.5 y 1.0 sin tocar `rel_volume` ni apalancamiento.
Por ejemplo, balance USD10,000 y q=1 destinan como maximo USD1,000 al margen;
q=.2 destina USD200, antes del redondeo de unidades. No es el porcentaje de
perdida maxima al SL. El contrato actual de una orden positiva admite q en
(0,1]; abstencion usa NOP. La peticion nueva de admitir q=0 explicitamente
requiere que signifique no abrir, nunca subirlo automaticamente al minimo.

## 7. Costes, recompensa y metricas

El perfil usado es un supuesto versionado, no una cotizacion contractual de
broker. Spread por hora desde el rollover de la sesion:
`[0,1):0.00030; [1,3):0.00010; [3,9):0.00006; [9,21):0.00003;
[21,23):0.00006; [23,24):0.00015`. Reapertura: 0.00030.
Se carga medio spread por fill, slippage 0.00002 en mercado y 0.00005 en stop,
comision USD 3.5 por lado y lote de 100,000 unidades. TP requiere atravesar el
nivel en 0.0001; esto permanece tambien en el control sin tasas.
El fill de TP tiene slippage cero; el cierre por margen usa el cargo de stop.

Swap: largo -7 y corto +2.5 USD/lote/rollover, triple en el indice 3 de la
sesion. El signo positivo del corto puede dar un credito. Se devenga mientras
la posicion esta abierta; no se cobra dos veces al liquidar. Un caso sintetico
plano corto puede quedar ligeramente positivo por ese credito configurado.

Los fills de referencia usan las barras admitidas Bid y cargos separados.
No es una simulacion de libro bid/ask con liquidez y profundidad: el tratamiento
simetrico del spread es una aproximacion explicita que debe juzgar el auditor.
Un evento intrabar no tiene timestamp exacto: el motor registra cierre de barra
y aplica la convencion declarada de coste intrabar a apertura.

**Recompensa de aprendizaje:** `r = 100*(equity_despues-equity_antes)/10000`.
No se entrena con Sharpe ni con un premio por numero de ordenes. Si se saltan n
barras sin ventana, se agregan recompensas con `gamma**i`; el bootstrap usa
`gamma**n`, y GAE usa tambien `lambda**n`. Gamma=0.99, lambda=0.95.

**Score de seleccion:** Sharpe de retornos diarios netos, anualizado con
sqrt(365). Equity diaria incluye dias cerrados por mantenimiento del ultimo
saldo, no mediante fabricacion de barras de precio. Desviacion con ddof=1.
Profit, drawdown, costes, actividad, compras, ventas y aciertos se conservan
por separado; no se suman arbitrariamente a Sharpe.

## 8. Como interviene en floating sequential

Se usa sequential backward floating sobre bloques coherentes. Los 120 inputs
forman 113 bloques: pares seno/coseno se conservan juntos. Al retirar columnas
se remapean canales; una rama sin columnas se elimina. No se limita a poner
una columna en cero. Las otras ramas mantienen su anchura; cambia la capacidad
total, hecho que forma parte del contraste estructural.

Cada subconjunto se entrena desde cero con PPO real en semillas 7,19,43,
102,400 decisiones, rollout512, minibatch256, cinco epocas PPO por rollout.
LR=0.0003, entropia=0.01, clip PPO=0.2, vf_coef=0.5, gradiente maximo=0.5.
AdamW usa wd=0.01. Dropout efectivo=0, pese a metadata heredada 0.3; no hay
early stopping en esta corrida. Cinco epocas PPO no son cinco pasadas anuales.

TRAIN efectivo: `[2020-01-01,2023-07-01)`; seleccion INNER_A:
`[2023-07-01,2024-01-01)`. La receta principal conserva 2020-2023 como marco de
desarrollo y 2024 como validacion exterior, pero esta seleccion no puntua 2024.
TEST 2025 no se lee. No llamar INNER_A "todo el ano de validation".

El scaler se ajusta en TRAIN una vez. Todos los subconjuntos conservan las
mismas filas validas del padre: 21,221 TRAIN y 2,805 INNER_A; no recuperan filas
al quitar una feature. La evaluacion recorre la estrategia con pesos congelados.
No hay reajuste semanal ni herencia de pesos entre subconjuntos.

Se compara la media de Sharpe INNER_A de las tres semillas. La brecha es
`max(0, Sharpe_TRAIN-Sharpe_INNER_A)`. **No hay limite duro de brecha**:
`max_gap=None`. Se usa como desempate, no como funcion objetivo principal.
Actividad >=2 se exige por el evaluador en seleccion; la compuerta conductual
se aplica en TRAIN y seleccion. Todas las semillas deben ser elegibles.

En cada paso se prueban remociones; una adicion puede reintroducir un bloque
si supera el incumbente de su cardinalidad. Tolerancia=0; presupuesto128
subconjuntos y minimo20 columnas. **128 no equivale a completar la poda hasta
20**: probar los vecinos inmediatos de 113 bloques ya consume gran parte.
Un cierre por presupuesto es provisional y no prueba optimalidad global.
Si ninguna configuracion es elegible, `selected=null`.

## 9. Simulaciones ejecutadas y resultados

### 9.1 Veintidos testigos del contrato actual

Se inyectaron comandos exactos en `HybridOrderEnv`, con la configuracion
economica congelada, ambas direcciones, ventanas sinteticas 13x120 y escalas
conocidas. Son pruebas de ejecucion, no de seleccion de datos ni aprendizaje.

| Escenario, una orden por caso | Largo: neto USD | Corto: neto USD | Resultado comprobado |
|---|---:|---:|---|
| Cierre voluntario plano | -6.36356 | -6.36356 | AGENT_CLOSE en apertura siguiente |
| SL | -87.27168 | -87.27168 | Stop efectivo |
| TP | +68.40827 | +68.40827 | Take profit efectivo |
| Ambos niveles en una barra | -87.27168 | -87.27168 | SL primero, ambiguedad registrada |
| Gap adverso | -132.72568 | -132.72568 | Fill al open, no al stop idealizado |
| Mantener hasta cierre obligatorio | -25.45424 | +0.45454 | Calendario y swap |
| Cerrar y volver a intentar ese dia | -6.36356 | -6.36356 | Segunda entrada bloqueada |
| Comando ilegal/fraccionario | -25.45424 | +0.45454 | Rechazado sin crear trade |
| Tres ventanas ausentes | -25.45424 | +0.45454 | HOLD automatico, sin detener protecciones |
| Cerrar antes de un SL futuro | -6.36356 | -6.36356 | Mejor que mantener hasta SL |
| Cerrar antes de un TP futuro | -6.36356 | -6.36356 | Peor que mantener hasta TP |

Todos los trades reconciliaron gross, cuatro componentes de costes y net.
Se comprobaron direccion, SL/TP, unidades, apertura siguiente, cupo diario y
flat terminal. El generador sintetico fue corregido durante su prueba para
expresar timestamps en nanosegundos; el intento inicial no genero resultados
financieros y no forma parte del cierre.

### 9.2 Ondas conocidas, salidas ideales e inversion

Mercado sintetico: sinusoidal de periodo24 barras mas una componente de
periodo6 con amplitud diez veces menor. OHLC coherente, sesiones120h separadas
por cierres, reglas12h y escalas calculadas causalmente con el motor real.
El generador ideal conoce la variacion de las proximas cuatro barras y la usa
SOLO para emitir acciones; abre si supera dos veces el estimador de coste y
cierra si la variacion se vuelve adversa. No es un oraculo de optimo global.
El control invierte esa variacion; NOP nunca opera. No se entrena una red.

| Mercado / salida | Neto USD | Sharpe sintetico | DD % | Ordenes L/S | Aciertos |
|---|---:|---:|---:|---:|---:|
| A: amplitud .002, 8 semanas / ideal | +2533.40 | 19.6937 | 0.2340 | 28/7 | 100% |
| A / invertida | -2675.83 | -20.3328 | 26.7583 | 7/28 | 0% |
| A / NOP | 0 | 0 por convencion contable | 0 | 0/0 | no aplica |
| B: amplitud .0015, fase3, 6 semanas / ideal | +312.99 | 4.5195 | 0.5228 | 24/1 | 20% |
| B / invertida | -1352.01 | -12.4541 | 13.5201 | 5/20 | 0% |
| B / NOP | 0 | 0 por convencion contable | 0 | 0/0 | no aplica |

Son resultados artificiales con conocimiento del futuro, no resultados de
forex ni evidencia de generalizacion. En B la salida ideal no solicita CLOSE_SHORT;
por eso no se usa ese solo mercado para certificar cobertura de comandos. Los
22 testigos y el mercado A ejercitan los cierres en ambos sentidos.

### 9.3 Optimizacion exhaustiva de ejecucion

Se agotaron tres parametros: fraccion de la banda factible TP {0.25,0.5,0.75},
fraccion SL {0.25,0.5,0.75}, fraccion del volumen permitido {0.2,0.5,1.0}.
Son 27 puntos y tres semillas: 81 replays de TRAIN, no entrenamiento nuevo.
Se uso grid completo: DEAP no aporta cobertura adicional en este dominio tan
pequeno. No se afirma haber optimizado todo el continuo ni la arquitectura.

Se conservaron calendario, costes, leverage, cupo diario, entradas, pesos y
acciones categoricas. Los logits se calcularon una vez por fila TRAIN; cada
simulacion aplica SU mascara al logit de esa fila. No se recicla una lista de
acciones de una trayectoria con otra posicion. El control midpoint reprodujo
exactamente los trades y equity diaria del modelo original en las tres semillas.

| Configuracion | Sharpe TRAIN medio | Profit medio USD | DD medio % | Elegible |
|---|---:|---:|---:|---|
| Control TP=.5, SL=.5, volumen=1 | -0.54939 | -5685.04 | 71.9713 | No |
| Mayor Sharpe numerico: TP=.75, SL=.5, volumen=.2 | -0.50170 | -1210.80 | 21.2964 | No |

La menor perdida al reducir volumen no prueba mejor senal. La regla de
eleccion exige primero elegibilidad de las tres semillas y solo despues
maximiza Sharpe medio, desempata por menor DD y menor volumen. Ningun punto
pasa; `selected=null`. Se conservaron los 81 resultados, no solo el mejor.
No se utilizo INNER_A ni 2024 para escoger parametros de este barrido.

### 9.4 Calibracion adicional con salidas simuladas

Diseno declarado antes de ejecutar:
[86_SYNTHETIC_OUTPUT_CALIBRATION.md](86_SYNTHETIC_OUTPUT_CALIBRATION.md).
Se evaluaron los 27 puntos sobre A y B, con ideal/invertida/NOP: 162
simulaciones. Se exigen beneficio ideal positivo, actividad y ambas direcciones
en cada mercado, inversion peor y NOP sin operaciones. Los cierres AGENT_CLOSE
en ambos sentidos se exigen sobre el conjunto A+B. Pasaron 18 de 27; los otros
nueve no ejercitaron cierre anticipado de cortos. No se modifico la regla tras
ver esos resultados.

Entre los elegibles se maximizo el Sharpe ideal medio, con desempates por menor
DD, menor q y finalmente orden lexicografico TP/SL. El resultado es TP=.75,
SL=.25 y q=.2, fracciones dentro de las bandas factibles, no distancias en pips.
El Sharpe medio pasa de 12.10657 a 13.85428; el profit medio baja de USD1423.20
a USD321.95. No se optimizo una suma Sharpe+profit+actividad en este estudio.
El SL no esta identificado: las tres fracciones empatan en la combinacion
ganadora. .25 proviene del desempate declarado, no de superioridad demostrada.

Un tercer mercado C (amplitud .0018, siete semanas, fase7) se leyo despues de
elegir. No decidio los parametros. Balance inicial USD10,000, mismos costes:

| C / salida | TP,SL,q | Neto USD | Sharpe | DD % | Compras/ventas | Aciertos |
|---|---|---:|---:|---:|---:|---:|
| Elegida / ideal | .75,.25,.2 | +922.51 | 22.38477 | .01811 | 6/24 | 100% |
| Control / ideal | .5,.5,1 | +4219.83 | 22.32588 | .11373 | 6/24 | 100% |
| Elegida / invertida | .75,.25,.2 | -590.40 | -22.86348 | 5.90395 | 25/5 | 0% |
| Control / invertida | .5,.5,1 | -3665.82 | -22.56379 | 36.65818 | 24/6 | 0% |
| Ambas / NOP | respectivas | 0 | 0 contable | 0 | 0/0 | no aplica |

La ideal elegida cierra las30 operaciones por AGENT_CLOSE; el control las30
por TP. Ambas respetan una entrada/dia y promedian3.75 ordenes/semana. La menor
ganancia monetaria con q=.2 es un intercambio con exposicion, no un fracaso
de contabilidad. El generador conoce el futuro: no son resultados de una red.

[Evidencia completa resumida](../audits/evidence/synthetic_output_calibration_20261026/SUMMARY.json).
Cierre sha256: `ff642c2c5c9be017fca9fdcd0cad8e1721d8b674eced0e34fac20d940f8aaff3`.
Se verificaron31 archivos y una segunda invocacion termino sin repetir celdas.
La extension paso siete pruebas focales, incluyendo cambio real de unidades,
TP/SL y rechazo de recibos alterados; no sustituye las100 pruebas anteriores.
La regresion ampliada posterior paso47 pruebas de agent-multi sin fallos;
el motor no cambio respecto de las57 pruebas de la entrega anterior.

Esta calibracion discreta queda como control. La solicitud posterior del
propietario exige DEAP y comparar las variantes de fitness: ese protocolo es
distinto y no se presenta este grid como si hubiera sido una busqueda genetica.
Su respuesta fija tres brazos emparejados: Sharpe, Sharpe+retorno neto y
Sharpe+retorno neto+ordenes/dia; tres semillas por brazo. El protocolo exacto,
incluyendo formulas direccionales, q=0 y restricciones no optimizables, esta en
[87_DEAP_OUTPUT_CALIBRATION_DESIGN.md](87_DEAP_OUTPUT_CALIBRATION_DESIGN.md).
La comparacion automatica genera su propia identidad y resultados; no modifica
ninguna conclusion de los grids ya cerrados ni se presenta como terminada
antes de su CLOSE.json.

### 9.5 DEAP terminado y re-auditado

Las nueve busquedas ya cerraron:3611 genomas,21666 simulaciones de calibracion
y66 de confirmacion. El [informe89](89_DEAP_OUTPUT_RESULTS_AND_EXECUTION_AUDIT.md)
contiene resultados, parametros, alcance y la correccion de una debilidad de
identidad al reanudar. Las nueve trayectorias geneticas y las66 simulaciones
finales se reprodujeron exactamente;56 pruebas focales pasaron.

En D/E sinteticos, Sharpe+retorno dio Sharpe8.94266 y neto medioUSD1094.65,
frente a8.85595 yUSD885.33 de Sharpe solo. Esta comparacion omitio los controles:
midpoint dio12.77731 yUSD2086.34; grid dio12.49450 yUSD370.35 con menor volumen.
Todos los finalistas DEAP pierden en Sharpe frente a ambos controles y en
beneficio frente a midpoint, no frente al grid de menor exposicion. No hay
base para preferir un fitness para RL. Actividad constante y cero toques de
SL en las ondas ideales impiden identificar esos efectos. Los SL no son
parametros calibrados y estos genomas no se promueven a la seleccion.

## 10. Hallazgos y disposicion

1. **Compuerta de cierres:** la version del wrapper congelado cuenta comandos
   CLOSE solicitados. Un stop al open puede adelantarse a esa solicitud. La
   comprobacion estricta entregada usa trades efectivamente cerrados por
   `AGENT_CLOSE`, separados por direccion, mas validez nativa y actividad.
   Tiene prueba negativa de solicitudes de cierre sin ejecucion. Fue la regla
   usada en las 81 simulaciones; no se ha sustituido silenciosamente la regla
   dentro de los recibos historicos.
2. **Colapso de acciones:** el TCN sin tasas seed7 coincide exactamente con
   abrir largo y cerrar en la siguiente decision, tanto TRAIN como INNER_A.
   El TCN con costes seed7 coincide en INNER_A con long/HOLD. La auditoria
   independiente amplio la muestra:55/59 recibos TCN coinciden en TRAIN y
   58/59 en INNER_A. El limite anterior a seed7 no describia todo el carril.
   INNER_A es bloque reutilizado de seleccion, no confirmacion independiente.
   No es saturacion
   del encoder: probabilidades y gradientes cambian. Ver
   [auditoria de sensibilidad](83_NATIVE_WRAPPER_AUDIT_FINDINGS.md).
3. **Informacion de la posicion (O13 refutado):** el testigo mantiene iguales
   entradas, mascara Y stops; cambia la barra futura e inyecta CLOSE/HOLD sin
   invocar la red. No demuestra ceguera a la distancia al stop. El contrato no
   incluye distancia, edad ni PnL en la observacion, pero atribuirle el fracaso
   exige otro experimento: mismo futuro, posiciones distintas, paso por la red.
   Se retira la afirmacion anterior; no se anadio realimentacion.
4. **Riesgo fijo:** TP/SL/volumen de la corrida original no eran parametros
   aprendidos ni optimizados. Ahora existe el contraste completo descrito,
   cuyo resultado es que ninguna celda cumple toda la conducta exigida. El
   midpoint no es simetrico: las bandas TP/SL tienen limites inferiores distintos.
   El nuevo GRID derivado informa RR, costes, financiacion y equilibrio binario
   de referencia; ese modelo binario no describe exactamente cierres anticipados,
   forced-flat, gaps ni tamanos variables. No demuestra imposibilidad universal
   de beneficio ni ausencia de senal. No se promueve una configuracion invalida.
5. **Versiones:** el testigo historico ORDER_ALPHA de 16 casos pertenece a un
   adaptador anterior de comandos continuos. Se conserva como historia, pero
   no se usa como certificacion de la interfaz categorica actual. Los 22 casos
   nuevos eliminan esa confusion de alcance.

Los resultados existentes conservan su valor como mediciones de su contrato
congelado; no son una recomendacion de trading ni una lista final de features.
No se requiere repetir los fits para inspeccionar sus ordenes o corregir una
evaluacion de elegibilidad basada en esos mismos trades autenticados. Un cambio
de politica, recompensa o estrategia que afecte entrenamiento si exige una
identidad nueva; no se equipara a corregir una etiqueta del informe.

## 11. Evidencia, reproduccion y cierre

[Resumen mecanico con los 27 puntos, tres semillas, 22 escenarios y seis ondas](../audits/evidence/native_output_contract_20261026/SUMMARY.json).
El resumen contiene hashes de IDENTITY, SCENARIOS, IDEAL_WAVES, GRID, DECISION y
del cierre. Los artefactos completos conservan trades, equity diaria, comandos,
parametros, pesos de origen y resultados por celda. Las ubicaciones privadas
se entregan en el indice local de la misma auditoria, no en GitHub publico.

```bash
# CPU; usar las rutas del indice privado, sin modificar la corrida fuente.
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python -m tools.native_strategy_contract_study \
  --root "$CAMPAIGN_MIRROR" --output "$NEW_AUDIT_OUTPUT" \
  --data-root "$ADMITTED_DATA_ROOT" --engine-root "$PINNED_ENGINE" \
  --selector-root "$PINNED_SELECTOR"
```

El mismo directorio de salida permite adoptar celdas completas solo tras
verificar identidad y hashes. Otra version del codigo requiere otra identidad.
Se verifico el resultado por una lectura completa de los 329 hashes. Las
pruebas focales finales dieron **43 passed** en agent-multi y **57 passed** en
heuristic-strategy, sin fallos ni omisiones. El segundo bloque cubre ademas
margin close-out, proteccion previa a cierre discrecional, distancia al fill,
rollover triple y contabilidad al cierre. No se afirma haber ejecutado las
suites enteras de ambos repositorios.

La solicitud de auditoria cruzada es
[85_OUTPUT_STRATEGY_AUDIT_REQUEST.md](85_OUTPUT_STRATEGY_AUDIT_REQUEST.md).

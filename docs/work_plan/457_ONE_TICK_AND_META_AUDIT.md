# 457. Esperar un tick y auditoria del linaje temporal SELL

**Decision:** mantener CLOSE sin veto. No usar estas corridas para afirmar
que se reprodujo la receta regularizada, que un encoder preentrenado no
funciona ni que el entrenamiento acumulativo converge.

## Ficha efectiva: pesos vs controlador

|Parametro|448/450/451: entrenamiento de OPEN_SELL|452-455 / 456-457|
|---|---|---|
|Datos|TRAIN 2020-01-01 a 2024-01-01 exclusivo; 18.899 ventanas etiquetadas. Validacion 2024: 4.899 ventanas admisibles|452-455: replay 2024. 456/457: solo se puntuan los ocho bloques de TRAIN; no se entrena con los 17 TP observados en validacion|
|Segmentos TRAIN en orden|2.208, 2.462, 2.387, 2.487, 2.178, 2.487, 2.474, 2.216 ejemplos; semestres calendario disjuntos|Mismos bloques en 456/457. Pares CLOSE: 128, 38, 14, 16, 142, 470, 68, 18. El primero solo calibra|
|Tensor y escala|61 canales seleccionados, 13 barras observadas, stride 2, ultima barra cerrada incluida, normalizacion fija de TRAIN; no z-score movil|Mismo abridor y propia tabla congelada. El veto no altera el tensor|
|Arquitectura|22 ramas TCN multicanal (2 bloques, ancho 8); fusion; atencion causal por patches de 3, embedding 8, 2 cabezas, FF 16; nucleo 2 bloques TCN ancho 16; 208 valores al cabezal; 25.118 parametros|Sin actualizacion de pesos neuronales|
|Accion y objetivo|Especialista supervisado SELL, no SAC ni BUY/SELL/NOP desde un escalar. Cabezal 208 -> 8 SiLU -> 12: para cada 3/6/12/24 h, logit de valor positivo y dos magnitudes softplus. BCE de signo + MSE de magnitud condicional, etiqueta de utilidad gestionada (incluye oportunidad de referencia), escala fijada en S1. Puja p*ganancia - (1-p)*perdida. La salida no es una incertidumbre bayesiana|Abrir requiere puja positiva sobre percentil 90 de la referencia; arbitraje por rango. Cierre separado evaluado sobre la posicion. En 457 el veto usa solo PnL a precio de liquidacion / riesgo inicial, TP/SL inicial, edad, tiempo hasta cierre obligatorio y horizonte one-hot (8 numeros); no realimenta al encoder|
|Optimizacion|AdamW; LR inicial 3e-4 solo al arrancar S1; los demas heredan pesos y AdamW del campeon. 5 actualizaciones/epoca por semestre, minibatches equilibrados de 435 a 498 (tope 512, **no** 32 ni 512 fijos); wd efectivo 0,001|452-455 sin ajuste. 456/457 ridge de forma cerrada: penalizacion 1, intercepto sin penalizar, estandarizacion con pares anteriores. Sin epocas, minibatches, LR, ES ni RRLop del veto|
|Parada y restauracion|Maximo 2.000 epocas por etapa; mejora >1e-8; ES cuenta desde epoca 100, paciencia 80. RRLop gracia 20, paciencia 20, factor 0,2, piso 4,8e-7; restauracion pesos y momentos al campeon antes de aplicar LR menor|Red, cerrador y calendario inmutables; el ridge se recalcula una vez por semestre antes de ver sus resultados|
|Seleccion|Validacion sola o promedio entre retorno val y media de bloques anteriores; no es garantia individual de retencion. 448 recalibra al cambiar de bloque; 450 conserva tambien la tabla y compara recalibrar; 451 exige >2 ventas/semana, rama validacion semilla 7, solo reentrena S6-8|Diferencial neto pareado y replay completo con una posicion. Sin seleccion posterior de mejor umbral en 2024|

## Resultado del factorial 457

Con los mismos pesos, filas, temporizacion, costes y folds de 456,
se separo el objetivo usado al ajustar la regresion de la accion ejecutada.
Etiqueta "tick": en la solicitud real, se omite CLOSE una barra, despues
vuelve el cerrador nativo, con SL/TP/fin de sesion siempre habilitados.
Accion "tick": si se veta, se vuelve a evaluar en la siguiente barra
valida; puede volver a vetar, no hay cierre obligatorio en la barra siguiente.
Los pares de prueba tambien simulan estos vetos sucesivos desde el mismo
estado y unidades. Las cuentas completas, en cambio, permiten entradas
y tamanos diferentes. Salida objetivo normalizada por riesgo monetario inicial.

|Etiqueta / accion si hay veto|Semestres que mejoran la cuenta (de 7)|USD salvados menos proteccion sacrificada en pares|
|---|---:|---:|
|Hasta proteccion / hasta proteccion (456)|3|-295,12|
|Hasta proteccion / un tick|4|-31,19|
|Un tick / hasta proteccion|4|-77,32|
|Un tick / un tick|4|-97,14|

La base y el control 456 se reprodujeron en los **ocho** bloques (exceptuando
tiempo de ejecucion). No sumar retornos de cuentas semestrales reiniciadas.
La variante tick/tick solo tiene retorno positivo en 2023 H2 (+0,91% vs
+0,71% base, apenas 18 pares originales); 470 de los 894 pares vienen de
2022 H2. No hay mejora consistente ni evidencia independiente de generalizacion.

## Meta-analisis y hallazgos

1. **Receta vs ejecucion.** La receta declara dropout 0,3 y wd 0,01. La
   instancia efectiva tiene dropout 0 en todas sus capas Dropout y el
   optimizador guardado tiene wd 0,001. No es el contraste de regularizacion
   de la receta. No atribuir la divergencia a esto sin un ensayo emparejado.
2. **Inicializacion y objetivo distintos.** S1 inicializa el encoder desde
   la semilla; reutilizar la arquitectura de ThreeLogits no carga un encoder
   preentrenado. Las ramas reciben gradientes; 448 comprueba cambios en las
   22 ramas, fusion, atencion, nucleo y cabeza. El abridor no recibe estado
   de cuenta; el cerrador y el veto si leen la geometria de la posicion.
   Estos resultados no evaluan SAC ni el sistema completo BUY/SELL.
3. **Volumen de entrenamiento real.** No se entreno una semana presentada
   como cuatro anos. Tampoco vio cuatro anos en una epoca: cada etapa solo
   toma gradientes del semestre actual. Un campeon heredado puede descartar
   todas las actualizaciones de ese semestre. De 94 archivos, 90 corresponden
   a etapas entrenadas y cuatro a copias. Las 90 pararon por paciencia en
   179; eligieron epoca cero 25/45 (448), 32/42 (450) y 2/3 (451). En 451
   S6-8 el LR inicial efectivo fue 2,4e-6 (125 veces menor que en S1),
   termino en 4,8e-7 en los tres; ganaron epocas 0, 22 y 0. No significa
   que las tres etapas convergieran ni que se probaran 2.000 epocas efectivas.
4. **Selector, no solo red.** En 450 los seis finales fueron positivos en
   validacion pero ninguno activo; en 451 el piso mantuvo un final activo
   (+1,0412%, 127 ventas) vs padre elegible (+1,0245%, 313). El mayor retorno
   sin piso (+1,5586% con 56 en semilla 7) incumple actividad. 452-455
   distinguen intercambio de tablas, variacion de admision y decisiones de
   cierre: no son cuatro entrenamientos nuevos. En 455, 49 archivos dan
   solo 13 rutas por semilla en total. En 11 de 13 CLOSE supera a HOLD por
   cuenta; no trasladar el resultado del cruce incompatible al campeon.
5. **Limitaciones temporales.** Solo el ajuste del veto es cronologico.
   El abridor ya vio TRAIN y fue seleccionado sobre 2024. La escala fija
   y ciertos productores de entradas se ajustaron con todo 2020-2023: no
   son transformaciones refit por prefijo. Un test de reentrenamiento
   historico como si se estuviera en cada fecha exige recalcularlas. El
   escalado fijo de TRAIN no es por si mismo una fuga hacia validacion 2024.
   El motor evalua panel anual con entradas enmascaradas al bloque: no
   interpretar semanas sin entradas del resto del ano como exposicion
   semestral al reportar frecuencia o Sharpe de TRAIN.

## Analisis comparativo de pesos

Checkpoint anterior y nuevo de cada transicion 451, misma semilla y
alineacion. En S6 y S8 todo el estado de red es identico bit a bit al
padre. En S7 se hizo un intercambio de **un modulo a la vez** sobre
pesos del padre, con tabla paterna fija y las mismas 276 ventanas de S1
(cada octava fila). Son decisiones binarias de admision por horizonte
(1.104 posibilidades), no ordenes llenadas ni ablaciones de rentabilidad.

|Modulo S7|Norma del cambio / norma paterna|Admisiones que cambian al reemplazar solo ese modulo|
|---|---:|---:|
|Extractores agrupados|0,213%|14|
|Fusion|0,594%|15|
|Atencion|0,130%|2|
|Nucleo|0,351%|16|
|Cabezal|0,311%|18|

Todos juntos: 40 cambios, no la suma de la columna; cambio absoluto medio
en la puja 0,01379. Un cambio mayor en norma no implica mas cambios de
decision. No hay base para culpar exclusivamente al cabezal ni para editar
pesos arbitrariamente. Son pruebas funcionales en un panel fijo de TRAIN,
no equivalencias de PnL o de otras semillas.

## Evidencia y siguiente contraste

- [94 registros por checkpoint: datos, LR, epocas, seleccion y resultados](evidence/457/checkpoint-ledger.csv)
- [Nombres de las entradas, arquitectura y deltas por modulo](evidence/457/AUDIT.json)
- [Ocho semestres, cuatro variantes y cuentas completas](evidence/457/DONE.json)
- [Historial y resultados 448-456](448_FULL_TEMPORAL_SEGMENTS.md)

Prioridad antes de mas vetos: reconciliar la receta efectiva de **este
especialista** con la regularizacion y la inicializacion que se desea probar.
No sobrescribir los pesos ni mezclar tres cambios en una sola corrida.
Primero comprobar correspondencia de nombres, formas y salidas del encoder
fuente, decidir ubicacion del dropout y sellar el contrato efectivo; luego
control/componente cambiado con mismos folds, semillas, lotes y cierres.
El historico disponible no identifica por separado los efectos de dropout,
wd, inicializacion, batch o cantidad de datos cuando no se variaron.

Para los proximos resumenes registrar por brazo, antes de comparar: fechas,
filas admitidas/etiquetadas y operaciones por split, hash y orden de
columnas, escala y ventana, procedencia y hash de los pesos, modulos con
gradiente, inputs de posicion y donde entran, salidas/acciones, regularizacion
efectiva, lotes y pasos reales, LR inicial/heredado/final, reglas de gracia,
paciencia y restauracion, epoca campeona y de parada, y controles con los
mismos datos. En replay marcar expresamente "sin entrenamiento neuronal".

Continuacion: [458, reentrenamiento completo y doble trayectoria en plateau](458_FULL_TRAIN_PLATEAU.md).

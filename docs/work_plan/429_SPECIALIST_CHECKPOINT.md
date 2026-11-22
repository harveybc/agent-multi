# 429. Checkpoint del plan maestro: especialistas, 2026-11-22

No hay abridor SELL admitido. Retorno de cuenta (%) no es ventaja por oferta
(pb frente a esperar con la misma continuacion). Las ofertas se solapan;
las repeticiones sobre un mismo calendario no son mercados independientes.

|N|Experimento|Resultado|Decision o alcance|
|---:|---|---|---|
|1|351, seleccion congelada|71 entradas, 28 grupos, 111 vecinos. TRAIN +16,06%, Sharpe 0,78, 745 ordenes; validacion +18,37%, 2,69, 207.|Reproduce inferencia, no reentrenamiento ni seleccion optima para especialistas.|
|2|400, SAC monolitico|4 anos: +14,28 a -3,32% y +0,83 a -8,24%. 8 anos: 111 a 0 trades en una semilla; en otra 737 SELL, 164 BUY, -3568 USD.|Colapso a NOP o exceso de ventas; no fue solo el simulador.|
|3|390, replay sintetico 2020-22|61 entradas: real +2,51%, sinteticos -14,78/-30,54%; 71: real +12,58%, sinteticos -20,91/-25,84%.|Validez estadistica no implica equivalencia economica.|
|4|403, exposicion historica|19197/38782 ticks legales en 4/8 anos. Perfil 24h: BUY/SELL 215184/220914 y 456929/478010 observaciones.|No confundir estados contrafactuales con trades independientes.|
|5|404, cuatro roles|Cuenta media 2024: clasificacion 4/8 anos -3,57/-6,49%, 689/814 ordenes totales; regresion -12,62/-9,00%, 2240/1539; 0/12 cuentas positivas.|Clasificacion perdio menos; mas historia no mejoro ambos metodos.|
|6|406, LR exploratorio vs restauracion fina|12/12 excursiones empeoraron perdida; fino/fino -4,10%, abrir explorando -16,43%, cerrar explorando -4,66%, ambos -19,25%. Fin 179-256 epocas, 4 restauraciones, LR 4,8e-7.|Impacto mayor en apertura.|
|7|407/408, apertura vs NOP|Veto NOP a SELL: -16,43 a -6,79%. Calibracion restrictiva: cero ordenes en tres semillas.|Evitar entradas malas no basta si desaparece la actividad.|
|8|413, ranking del arbitro|H2 2023: base -20,83/-23,12/-16,27%, timing -11,96/-13,35/-12,26%; filtro -1,01/+0,34/-1,85% con 18/14/32 trades.|No repetir ranking en el arbitro como si fuera un nuevo abridor.|
|9|417-419, estado y codificacion|4083 grupos: regret curva 17,08-17,10 pb con estado real vs 20,28-20,39 intercambiado. Gray mejoro regret 3/3, beneficio a entradas fijas 0/3.|Conservar curva de cierre.|
|10|419/420, etiqueta gestionada|Ventaja positiva pasiva 14/24 vs gestionada 1/24, medianas +1,35/-2,72 pb. Refit alineado mejora 5/6 por oferta, 0/6 cuentas positivas.|Mantener etiqueta y cierre coherentes.|
|11|421-423, valor y especializacion|15/18 mejores que azar, 18/18 ventajas absolutas negativas, 2/18 cuentas positivas activas; ningun ancho 8/32 o split califico.|No promover mas capacidad.|
|12|424/425, calibracion|Previsto 42,0%, observado 33,1%, recalibrado 32,7%. Medianas original/simple/condicional: -8,01/-3,89/-1,30%, 192/73/58,5 trades. 0/18 positivas activas.|Mejor calibracion, no rentabilidad demostrada.|
|13|426/427, mismo cupo semanal|Ventaja original sobre azar TRAIN +2,54/+1,68 pb; monitor -0,49/+0,53; H2 -0,73/-0,01. Recalibrado H2 -0,16/+0,75, ambos IC incluyen cero.|Perdida de ranking fuera de muestra.|
|14|428, techo por oferta|H2 2022 azar/modelo/oraculo -4,01/-4,74/+34,35 pb; H2 2023 -3,14/-3,15/+42,61. Captura TRAIN 5,6/3,7%.|Techo retrospectivo, no retorno alcanzable ni senal causal.|

|N|Tarea|Medicion o criterio|Destino|Paralelismo|
|---:|---|---|---|---|
|1|Comprobar Gamma tras corte|SSH, GPU, RAM, funcionamiento CUDA y procesos|Omega a Gamma|Con 2 y 3|
|2|Preparar perdida interna de ranking|Pares dentro de semana y horizonte, senal conocida, permutacion, sin cruces de particion|Omega CPU|Con 1 y 3|
|3|Verificar referencia y fijar decision|Cuenta y ventaja 423, cupo y control azar 427|Omega CPU|Por cohorte/semilla|
|4|Control vs ranking|Curvas, paciencia, LR, restauraciones, mejor epoca e identidad; 6 pares en total|Gamma 5090/5070 Ti; Omega 4070 si disponible|Pares independientes, ambos brazos de cada par en mismo dispositivo|
|5|Pesos y decisiones|Deriva por capa, cambios del decil alto sobre mismas ofertas, score y orden de pares|Omega CPU|Al completar cada par mientras otros entrenan|
|6|Replay cronologico|Ventaja, retorno, drawdown, Sharpe, ganadas, frecuencia y lados en ambos H2|Omega CPU|Por par; despues de entrenarlo|
|7|Dictamen agregado|Sin promover si solo mejora TRAIN o reduce actividad. Si califica, probar integrado con BUY y closers antes de otra seleccion|Omega CPU|Tras los 6 pares; sin DEAP, STL o live|

## Contrato de ejecucion

Control shared8 de 423; mismo target gestionado y mismo monitor MSE para ambos
brazos. Intervencion unica: sumar a la perdida original el termino logistico
de ranking con peso 1 en unidades normalizadas TRAIN. Emparejar dentro de
semana y mismo horizonte, omitir empates; sin optimizar ese coeficiente.
Usar 7/19/43, cohortes 2022/23, politica de restauracion 405, batch 512,
maximo 2000 epocas. Cada brazo conserva su mejor checkpoint. Primario: replay
positive-EV congelado. Para calificar para integracion (no promocion): en las
seis celdas superar control, azar y cero en ventaja al mismo cupo, superar
control y cero en retorno, y mas de dos trades por semana. No usar 2024.

Pruebas antes de entrenar: senal ordenada vence a invertida y barajada,
empates sin gradiente, grupos sin cruce, control guardado reproduce cuenta y
ventaja. Exportar curvas, pesos, resultados y decision a
`/var/tmp/role429-sell-ranking` en Omega. Dragon reservado a Retsu.

## Operacion y trazabilidad

Acceso comprobado por LAN; el alias VPN no respondia. Dos pruebas CUDA
de calculo y lectura de memoria, ninguna aplicacion de computo previa en las
GPU de Gamma, ningun evento Xid/OOM visible en el arranque consultado. No se
detuvo ni se reemplazo ningun proceso de otra linea.

Desde la raiz de agent-multi, en el entorno PyTorch de la campana:

```bash
export ROLE429_SSH_TARGET='<worker-ssh-alias>'
PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python tools/role429_campaign.py &&
  OMP_NUM_THREADS=1 python tools/role429_report.py
```

La ejecucion necesita los almacenes autenticados 404/423/427 y el entorno
`/var/tmp/discrete-sac-probe-venv` en ambos hosts. Coordinador en Omega;
GPU 1 de Gamma para (2023,7/43), GPU 0 de Gamma para (2023,19)/(2022,43),
GPU 0 de Omega para (2022,7/19). Los resultados por pareja no son nuevas
replicas independientes del mercado. Se reutilizan solo las salidas de esta
campana: CONFIG fija identidad del codigo principal y mercado; los paquetes
de entrenamiento se verifican por SHA al cargar. Los estados de reanudacion
se guardan cada 20 epocas y conservan AdamW, pesos, campeones, historial,
contadores y RNG. La paciencia es conjunta para el fin de la pareja, pero
cada brazo restaura y exporta su propio mejor estado.

El recibo original conserva el hash del coordinador ejecutado. En la copia
publicada el alias SSH se configura por entorno; su hash de fuente cambia.
No modificar el recibo ni reanudar encima de esa ejecucion con otra identidad.

|Requisito|Evidencia|
|---|---|
|Costos, objetivo y referencia autenticos|Por pareja, `*-preflight.json`: cuenta original exacta, ranking y azar de 427 con tolerancia 0,001 pb|
|Datos de evaluacion fuera del entrenamiento|Paquete remoto contiene solo fit, monitor y semanas de fit; resultado H2 se reconstruye en Omega|
|Se aprende el sentido correcto de los pares|Pretest de alineacion, inversion, barajado, empates y loss original; chequeo de semana en cada epoca|
|Estabilidad y efecto funcional|Pesos iniciales y mejores, curvas por brazo, cambio relativo por tensor, pares corregidos/perjudicados y cambios en el decil seleccionado|
|Decision economica|JSON por pareja, historial, seis PNG, informe y DONE; ninguna promocion automatica|

## Resultado del contraste ejecutado

Las seis parejas completaron preflight, entrenamiento y replay. Partiendo del
mismo estado por pareja, cada numero separado por "/" es control/ranking.

|Cohorte H2|Semilla|Epocas y salida|Mejor epoca|TRAIN pb|H2 pb|Azar H2 pb|Retorno cuenta %|Trades|Pares H2 corregidos/perjudicados|
|---|---:|---|---|---|---|---:|---|---|---|
|2022|7|180 paciencia|61/100|-2,51/-1,78|-4,60/-4,25|-3,75|-14,73/-9,86|254/206|834/746|
|2022|19|2000 limite|695/1990|-1,79/-0,33|-5,15/-5,55|-4,14|-15,82/-15,40|351/340|558/485|
|2022|43|2000 limite|1994/1981|-2,76/-1,45|-4,57/-5,78|-4,13|-12,19/-13,78|330/353|669/586|
|2023|7|338 paciencia|258/258|-4,05/-3,47|-3,45/-3,23|-3,07|-11,20/-12,29|193/188|310/283|
|2023|19|179 paciencia|3/3|-3,78/-3,63|-3,22/-3,17|-3,12|-4,01/-3,57|98/96|85/73|
|2023|43|179 paciencia|68/40|-1,28/-1,76|-1,91/-2,70|-3,21|-4,97/-4,59|119/117|823/682|

Decision: **rechazar para integracion, 0/6**. Superioridad ordinal en las
parejas muestreadas (6/6 con mas correcciones que danos) no se tradujo en
ventaja por oferta al mismo cupo (mejora en 3/6 frente al control) ni en
retorno positivo (0/6). No pasar a floating, DEAP o live con estas cabezas.
Siguiente diagnostico propuesto, sin nuevo entrenamiento: descomponer el
cambio del decil en ofertas entrantes y salientes y cuantificar la magnitud
de las diferencias netas en los pares corregidos/perjudicados, especialmente
cerca del umbral de seleccion. No atribuir causalidad a una capa por su norma.

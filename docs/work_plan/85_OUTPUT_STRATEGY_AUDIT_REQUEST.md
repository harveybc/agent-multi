# Solicitud de auditoria cruzada: salidas y estrategia

Para Satoshi. Fecha: 2026-10-26. Emisor: Musashi.

**Dictamen recibido:** [Satoshi, commit2fd4eeda](../audits/SATOSHI_AUDIT_VERDICT_MUSASHI_OUTPUT_STRATEGY_2026_10_26.md).
17 puntos verificados, O13 refutado. Contrato/reproduccion aceptados; uso para
decisiones DEAP/seleccion rechazado. La [respuesta ejecutada](91_CROSS_AUDIT_CORRECTIONS_AND_LEARNING_CONTROL.md)
separa correcciones medidas de la prueba de aprendizaje C4 que aun no se ejecuto.

## Entrega cerrada

Auditar el [informe completo](84_OUTPUT_STRATEGY_REPORT_FOR_SATOSHI.md) y su
[resumen mecanico](../audits/evidence/native_output_contract_20261026/SUMMARY.json).
El estudio ejecuto 22 testigos, seis simulaciones de ondas y 81 replays de
calibracion TRAIN. El resultado del barrido es `NO_FEASIBLE_CANDIDATE`.
No se pide aprobar rentabilidad ni certificar una seleccion final.

Ampliacion ejecutada: 162 simulaciones del grid sobre salidas artificiales mas
seis de confirmacion sintetica, seccion9.4 del informe. Son otra identidad y
otro objetivo, no un reemplazo del resultado negativo sobre politicas reales.

La comparacion DEAP solicitada por el propietario termino y fue re-auditada:
[informe89](89_DEAP_OUTPUT_RESULTS_AND_EXECUTION_AUDIT.md), con9 busquedas,
3611 genomas,21666 simulaciones de calibracion y66 de confirmacion. El lector
puede auditar el contrato y estos estudios cerrados sin aprobar la seleccion
financiera. Protocolo87 y lanzamiento historico88 conservan su alcance.

Prioridad: intentar refutar el contrato y los resultados con contraejemplos
ejecutables. No sustituir esa tarea por una opinion sobre la dificultad del
trading ni por una recomendacion de abandonar los datos o TCN.

## Verificaciones solicitadas

| ID | Objeto a atacar | Evidencia y criterio |
|---|---|---|
| O01 | Cinco clases realmente categoricas | `OrderDistribution`, entero exacto, logits enmascarados; rechazar fracciones y acciones ilegales |
| O02 | TP, SL y volumen efectivos | Reconstruir banda y sizing desde receta; distinguir cabeza Beta inactiva del riesgo fijo; cotejar midpoint y grid |
| O03 | Apertura y cierre causal | Observacion al completar b, fill en open b+1; mutar el futuro sin alterar la decision ya emitida; separar oracle artificial |
| O04 | Cierre anticipado | CLOSE_LONG/SHORT ejecutados, no solo pedidos; casos antes de SL y antes de TP; restricciones de entrada no bloquean salida |
| O05 | NOP y protecciones | NOP flat no opera; HOLD conserva SL/TP y calendario; ventanas ausentes no suspenden contabilidad |
| O06 | Orden de eventos | Gap, SL/TP ambiguos, margen, rollover y forced-flat; verificar el orden del codigo, no el comentario abreviado de metadata |
| O07 | Costes y volumen | Gross-net, comision por lado, spread, slippage, swap y signo corto; Bid de referencia y convencion intrabar explicitados |
| O08 | Simulaciones nuevas | Reproducir 22 casos y seis ondas; ninguna confusion con el ORDER_ALPHA antiguo de otro adaptador |
| O09 | Calibracion cerrada | 27 x 3, TRAIN exclusivamente, tres controles exactos, cero fits, codigo y modelos fijados; comprobar la decision sin candidatos elegibles |
| O10 | Cache de logits | No es replay de acciones fijas: mascara de cada trayectoria; controlar orden de filas, periodo y paridad del control |
| O11 | Floating backward real | Eliminacion estructural de canales/ramas, seno-coseno coherente, soporte comun, fits frescos por semilla, remocion y reintroduccion |
| O12 | Score y elegibilidad | Media de Sharpe INNER_A; brecha como desempate, no limite; todas las semillas; no confundir TRAIN de calibracion con validation |
| O13 | Informacion de posicion | Reproducir el testigo de observaciones iguales con stops distintos; evaluar su impacto sin agregar estado por iniciativa propia |
| O14 | Artefactos y numeros | Verificar hashes y todas las tablas; alterar un recibo y exigir rechazo; probar reanudacion sin repetir resultados completos |
| O15 | Calibracion sintetica | Recalcular18 elegibles de27, ganador segun regla y empate de SL; C solo confirma; no confundir mayor Sharpe con mayor profit ni grid con DEAP |
| O16 | Fitness DEAP | Comparar F1/F2/F3, componentes y semillas; comprobar q=0, riesgo por direccion, restricciones antes del score, checkpoint RNG y confirmacion D/E posterior a las nueve busquedas |
| O17 | Debilidad de reanudacion | v1 no vinculaba objetivo; v2 si. Intentar intercambio de checkpoint y refutar la auditoria que reconstruyo exactamente las nueve busquedas v1 |
| O18 | Actividad constante | Comprobar que el terminoA fue constante en los3611 candidatos; no concluir que F3 fue una prueba discriminante de fomentar actividad en RL |

## Puntos que no deben pasar inadvertidos

- El modelo de la seleccion actual es PPO, no SAC, aunque una clave heredada
  diga `sac`. No hay demostraciones ZigZag en este carril.
- La compuerta antigua de la corrida cuenta intenciones de cierre; la
  comprobacion estricta del estudio exige `AGENT_CLOSE` real por direccion.
  No son intercambiables. Dictaminar el impacto sobre cada resultado, no
  invalidar todos los fits solo por el nombre de la compuerta.
- TP/SL y q de la corrida original no eran aprendidos. El barrido es sobre
  politica congelada, no una optimizacion conjunta de parametros y pesos.
- El oraculo sintetico ve cuatro barras futuras por diseno. Su beneficio no
  debe aparecer en una tabla de rendimiento aprendido ni de forex real.
- El escenario B ideal tiene solo una venta y no solicita su cierre temprano;
  la cobertura de esa accion se prueba en los testigos especificos y en A.
- La regla de actividad es promedio >=2, no >2 ni dos operaciones obligatorias
  por semana. No usar operaciones forzadas para hacer pasar una corrida.
- Los costes son supuestos versionados. El barrido no modifica costes,
  apalancamiento, horarios ni limites de negocio para fabricar rentabilidad.
- Los cambios de subconjunto conservan filas del padre: quitar una feature
  no permite recuperar fechas que habian sido invalidas.
- El presupuesto128 no prueba una seleccion global ni garantiza llegar a20.
- Una regla fija reproduce resultados del TCN; no atribuir esas ganancias a
  patrones temporales sin un contraste apropiado.

## Reproduccion permitida

Trabajar en otra carpeta, CPU, mismos pins y copia autenticada. Se autorizan
lecturas, replays, pruebas negativas sobre copias y recomputo de decisiones.
No modificar la corrida fuente, su spec, pesos, datos, TEST, servicios activos
ni el estado de otras aplicaciones. No publicar rutas, credenciales o topologia.
No hay necesidad de entrenar modelos para verificar los resultados entregados.

El indice privado da las rutas de fuentes, modelos, espejo y evidencia.
El resumen publico permite verificar los numeros sin acceso a credenciales.

## Formato del dictamen

1. Tabla O01-O18: verificado, refutado o fuera del alcance demostrado, con
   archivo/funcion, comando ejecutado y evidencia propia.
2. Hallazgos por gravedad, con contraejemplo minimo. Separar defecto ejecutable,
   discrepancia documental, supuesto de negocio y limite estadistico.
3. Para cada hallazgo, indicar exactamente que mediciones afecta y cuales no;
   recomendar la correccion mas acotada y la prueba que debe demostrarla.
4. Recalcular la decision del grid y el numero de candidatos elegibles.
5. Dictamen final separado para contrato de ejecucion, reproducibilidad de las
   simulaciones y admisibilidad de los resultados de seleccion. No agrupar los
   tres en una aprobacion global ambigua.

No se requiere que el dictamen sea favorable. Se requiere que se apoye en el
codigo y los artefactos, y que no confunda beneficios ideales con aprendizaje.

Anexo de revision: [validacion de entradas de elegibilidad](90_EXECUTED_BEHAVIOR_INPUT_AUDIT.md).
Rechaza NaN, validez no booleana y contadores mal tipados. Las81 decisiones
financieras guardadas no cambian y las21732 trayectorias DEAP pasan la
validacion de entradas; no se repitieron fits ni se cambio la corrida sellada.

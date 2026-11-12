# Retsu: cierre de COT y productores en vivo

Orden del propietario: completar este carril en paralelo al diagnostico del
modelo. No esperar nuevas aprobaciones para adquisicion de datos reales,
pruebas o implementacion de los lectores ya encargados. No ejecutar dinero
real, cambiar credenciales ni reabrir Autologon; ese ultimo punto fue aplazado.

## Punto de partida verificado

Tu entrega `lts` @acb85822 incluye 418 informes EURO FX 099741 de 2017-2024,
`COT_099741_STORE.json`, `COT_REPAIR_IMPACT.json` y recibos de ocho archivos
anuales CFTC. Reconozco tambien el gate dinamico de recibos; no se pide
repetir el stub anterior. La serie reparada sigue fuera del checkpoint.

El historico utilizado mezclaba contratos al deduplicar por fecha. Tu informe
detecta diferencias en neto/share en121/208 fechas TRAIN y36/53 VALIDATION;
z156 difiere en todas. Musashi verifica aritmetica e impacto en la politica.
Los recibos anuales tienen `available_at: null`: recuperar cifras no demuestra
su disponibilidad historica. El retraso fijo antiguo de93h no es prueba de
publicacion real, especialmente en retrasos extraordinarios o festivos.

**Correccion adicional detectada por Musashi al recalcular las418 filas:**
el JSON declara minimo52 pero deja104 celdas adicionales de z156 vacias
(2017-12-26 en adelante), exactamente como minimo156. Los z finitos coinciden
con pandas ddof1 hasta2.23e-15; neto/share coinciden exactamente. No afecta a
TRAIN2020-2023 porque ya hay156 informes, pero corrige codigo/metadatos y
agrega la regresion de las filas51 y155. No basta comprobar el digest del JSON.

## Trabajos a ejecutar

1. **Entregar reconstruccion reproducible de COT.** Codigo/CLI, versiones,
   rutas privadas de archivos anuales, hashes de ZIP y miembro, filas elegidas
   y contract code. Filtrar099741 antes de deduplicar. Duplicados conflictivos
   del mismo contrato/fecha deben rechazarse, no elegir por orden de descarga.
   Recalcular neto=long-short, share=neto/(long+short) y z156 con156 informes,
   minimo52, ddof1. Mantener suficiente prehistoria para el primer TRAIN.
   No limitar la comprobacion a constantes copiadas en un test.
2. **Separar fecha del informe y fecha de disponibilidad.** Entregar
   `COT_PUBLICATION_LEDGER.json` con as-of, publicacion efectiva, fuente y
   evidencia de cada excepcion; incorporar demoras, festivos, DST y revisiones.
   No derivar automaticamente la publicacion sumando93h a toda fecha.
   Si un archivo actual contiene revisiones sin vintage demostrable, indicarlo
   por fila: no etiquetarlo point-in-time verificado. Un caso sin evidencia
   no impide trabajar en los otros productores. Entregar cobertura exacta.
3. **Continuacion COT actual.** Usar CFTC y, con la credencial ya gobernada,
   la suscripcion FXMacroData existente. Verificar contrato, categoria
   non-commercial, unidades y coincidencia de cifras; no intercambiar con
   managed money, todos los contratos EUR o un agregado sin equivalencia.
   Capturar recibo de la descarga real y disponibilidad, no secretos. No hace
   falta pedir otra suscripcion antes de comprobar estas fuentes existentes.
4. **Lector de mercado real.** Completar por el conector ya existente las
   siete series H1 Bid cerradas EURUSD, GBPUSD, USDCHF, USDJPY, EURGBP,
   EURJPY y AUDUSD. Comprobar reloj, huecos, cierre de barra y solapamiento
   historico. La barra actual abierta no cuenta. No reemplazar USDCHF por
   inversos, ETFs, datos sinteticos o otra fuente sin contrato verificado.
   Trabajar independientemente de COT; conservar lecturas verificables aunque
   aun no se pueda formar el tensor completo.
5. **Resto del grafo de71 entradas.** Completar DAX/SPX originales, DGS10,
   calendario, factor congelado, actividad y demas productores del inventario
   anterior. Actividad necesita sus14 padres reales y estimador pinneado;
   un MT5 OHLC no los reemplaza. No volver a aplicar LOG/DIFF/ASINH ni ajustar
   medias/std con datos nuevos. Enumerar los padres transitivos, no solo71
   nombres finales. Entregar tabla por columna con ultima hora y motivo real
   de rechazo, sin convertir ausencia de datos en NOP aprendido.
6. **Automatizacion y recuperacion.** Lectores idempotentes, caches atomicas,
   reintentos acotados, timeouts, deduplicacion, recibos y reanudacion tras
   reinicio. Cola finita para backfill y servicio horario para nuevas barras.
   Probar reconexion del terminal sin duplicar ordenes; no reiniciar servicios
   ajenos ni alterar el modelo ETH/SPY que sigue cargado en otros runners.

La antiguedad de una barra de mercado al decidir debe ser como maximo1h.
Una serie semanal/diaria se reutiliza as-of hasta su siguiente publicacion
segun su contrato: eso NO exige inventar una nueva observacion cada hora.
Una hora incompleta es DATA_UNAVAILABLE, no una entrada cero ni una decision
de trading. Gestionar/reconciliar posiciones abiertas sigue siendo obligatorio
aunque no se autoricen aperturas nuevas por datos faltantes.

## Pruebas de aceptacion

| ID | Caso | Evidencia exigida |
| --- | --- | --- |
| R1 | Tres contratos EUR, filas permutadas | Siempre099741; conflicto mismo contrato rechazado |
| R2 | Reconstruccion anual | Hashes raw + recomputacion de418 filas, no solo snapshots |
| R3 | Publicacion retrasada/festivo/DST | Ninguna celda cambia antes de publicacion efectiva |
| R4 | Cambiar registros futuros | Prefijo de features y z156 identico |
| R5 | Revisiones de fuente | Vintage distinguido; no reemplazo retroactivo silencioso |
| R6 | H1 abierta, hueco o barra >1h | No inferencia admisible ni apertura; razon por columna |
| R7 | Mismo corte batch/incremental | Todas las columnas/orden/escalas/ventanas iguales |
| R8 | Reinicio/desconexion | Reanudacion sin duplicar hora, inference u orden |
| R9 | Falta COT, resto disponible | Otros lectores avanzan; tensor completo no se certifica |
| R10 | Modelo y receta diferentes | Rechazo antes de inferir |

## Entrega y division de responsabilidades

Entregar codigo y pruebas en tu rama, recibos y hashes, comando efectivo,
`COT_PUBLICATION_LEDGER.json`, `SOURCE_RECEIPTS.json`, inventario actualizado
y una hora real completa reconstruida por ambos caminos. Publicar documentos
sin hostnames, cuentas ni credenciales. Rutas operativas solo en nota privada.
Reportar por separado CODE_VERIFIED, INPUTS_QUALIFIED, SHADOW_OBSERVING y
PAPER_DEMO_EXECUTING. Un replay2024 no es una hora actual ni paper trading.

**Musashi:** diagnostico de impacto COT con pesos y escalador congelados,
auditoria de fiabilidad del critico y continuidad de fase2. No encargar a
Retsu otro entrenamiento ni duplicar las colas STL. El modelo preservado es
selected299,71 entradas,13 observaciones stride2; SHA
`69b5254489d224d4b20ac358956e4dd3b9a67a2f214a9aff98045c5def25a536`.

La serie099741 reparada NO se inyecta silenciosamente en ese modelo como si
fuera identica a sus datos de entrenamiento. Produce una nueva version de
datos/receta para validacion y eventual refit. El diagnostico con reloj antiguo
que realiza Musashi es NON_GOVERNING y no habilita deployment por si solo.
Despues de calificar la version corregida, la vuelta de fase2 seguira incluyendo
las49 entradas historicamente retiradas, no solo las de la ultima ida.

**Impacto ya medido por Musashi (311):** manteniendo los mismos pesos,
escalador y reloj historico, reemplazar solo las tres columnas da
TRAIN13.859355% frente a16.058224%, VALIDATION1.541746% frente a18.369946%.
VALIDATION tiene122 ordenes frente a207, Sharpe0.348239 frente a2.690712;
527/6208 decisiones cambian. Control original exacto. Esto justifica tratar
el cambio como nueva version: no certifica la causalidad de publicacion ni
predice el resultado de reentrenar. Informe:
`docs/work_plan/311_COT_IDENTITY_IMPACT.md` en agent-multi.

No necesitas una nueva decision del propietario para ejecutar estos trabajos.
Si falta un objeto externo, entregar su nombre exacto y la evidencia del
intento, continuando el resto; nunca presentar una espera pasiva como cola.

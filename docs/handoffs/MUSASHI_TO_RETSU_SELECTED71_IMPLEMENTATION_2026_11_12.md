# Retsu: implementacion completa de las entradas selected71

## Objetivo y reparto

Retsu implementa, prueba y automatiza adquisicion -> productores -> tensor ->
provider -> cuentas virtuales -> ejecutor demo. Musashi continua la fase 2.
Este encargo complementa el handoff `MUSASHI_TO_RETSU_SELECTED71_PROVIDER_2026_11_12.md`;
sus hashes y contrato no cambian. No basta otro inventario ni otro replay
historico: la entrega debe incluir una hora real admisible o el defecto exacto
demostrado, con su reparacion implementada hasta donde sea ejecutable.

No comprar otra suscripcion ni pedir claves por chat. Ya existe FXMacroData:
usar su acceso gobernado, sin imprimir secretos. No intentar Autologon, cambiar
contrasenas, reiniciar Windows ni reemplazar los runners ETH/SPY. Eso no impide
desarrollar y comprobar el lector de la sesion actualmente abierta.

## Identidades y material ya entregado

| Objeto | Valor |
| --- | --- |
| Modelo conservado | `69b5254489d224d4b20ac358956e4dd3b9a67a2f214a9aff98045c5def25a536` |
| Manifiesto | `f52a5c211c8421ccb354561395bcc48339d2f984b3756b48a2a0c981ce193d96` |
| Receta | `recipe_b5fb7bcdfa1dbec8` |
| Archivo receta | `e2cb27de272197ae5bba67e2172569ea89e580fa217b00474d0471d6e35d8ec3` |
| Observacion | 13 x 71, stride 2, 28 grupos; orden completo de RECIPE.json |
| Salidas | 0 NOP, 1 BUY, 2 SELL; opuesta a posicion abierta cierra, sin reversal simultaneo |
| Escalado | Constantes TRAIN, sin rolling-z, sin STL ni realimentacion |

Rutas locales, no URLs de despliegue:

- Tu checkout: `/tmp/lts-forward-pool`, base inspeccionada `68058d3`.
- Bundle: `/var/tmp/sac-selected299-export-20261112-r1/bundle`.
- Provider: `/var/tmp/pp-selected71-provider-20261112`, commit `9301cd6`.
- Parche exportado del provider:
  `/var/tmp/retsu-selected71-delivery-20261112/0001-Add-pinned-categorical-SAC-replay-provider-with-stri.patch`.
  SHA256 `7022b969ffb1ef6826b1c857c033c3d1f57b5ba7aa133cdc815b2b5847b17dac`.
- Nota operativa privada con servicio, runtime, Python y puertos:
  `/var/tmp/RETSU_SELECTED71_DEPLOYMENT_20261112.md`.
- Actividad: `/home/harveybc/.local/share/agent-multi/stl-recovery-20261111/activity/consumer_recovered/CONSUMER_ACTIVITY.joblib`,
  SHA `fdc95a0cdabf9a5327c991187f62c8be928476f385551c7e8db3972045167d6b`.
- Factor: tu `app/forward_pool/FX_F1_STATE.json`.

El parche entregado es del adaptador ya probado, NO una implementacion nueva
del feed. Si `9301cd6` ya es ancestro del checkout del provider no reaplicarlo.
En otro checkout limpio compatible, comprobar antes de aplicar:

```bash
git apply --check /var/tmp/retsu-selected71-delivery-20261112/0001-Add-pinned-categorical-SAC-replay-provider-with-stri.patch
git am /var/tmp/retsu-selected71-delivery-20261112/0001-Add-pinned-categorical-SAC-replay-provider-with-stri.patch
```

No aplicar este parche en lts. Leer primero
`prediction_provider/docs/CATEGORICAL_SAC_REPLAY_PROVIDER.md` y ejecutar sus
pruebas `mechanics/tests/test_categorical_service.py` en su entorno propio.
Nunca mezclar imports `app` de los diferentes repos en el mismo proceso.

## Defecto concreto del consumidor actual

En `app/forward_pool/selected71.py`, `live_decision(present)` descarta `present`;
`_admissibility` devuelve estados constantes. `_require_identity` exige que
COT continue sin resolver. Es un inventario conservador, no un feed operativo.

Implementar validacion dinamica de recibos y valores, NO cambiar constantes
a `ADMISSIBLE` ni borrar los rechazos. Separar identidad inmutable del modelo,
informe historico COT e identidad del dataset admitido. Conservar el informe
original como evidencia, sin exigir que todos los futuros informes sean falsos.

## Fuentes exactas y dependencias

| Familia | Fuente a conectar | Condicion para admitir |
| --- | --- | --- |
| Precio EURUSD y seis cruces | Terminal demo existente, H1 bid | EURUSD, GBPUSD, USDCHF, USDJPY, EURGBP, EURJPY, AUDUSD; simbolos explicitamente mapeados, barra cerrada, reloj identificado |
| Retornos, indicadores, SSA, wavelet, Hilbert | Productores originales sobre barras reales | Mismos parametros, extremos causales y contexto; no otro indicador parecido |
| Factor y fuerzas | Seis cruces y matrices congeladas | Precision completa, cesta completa; nunca triangulacion para rellenar un cruce ausente |
| Kalman/GMM | Estados originales y actualizacion causal | Sin refit diario ni reset arbitrario del estado |
| US10Y | FRED DGS10 | Misma serie, unidades, cambios 1/5 observaciones, disponibilidad y rezago de receta |
| DAX/SPX | Series diarias originales de indices Yahoo | Mismos simbolos, campo y ajustes que el productor sellado; NO CFDs ni ETFs |
| COT | CFTC EURO FX, contrato 099741; FXMacroData como transporte alternativo a verificar | Mismo informe, categorias, unidades y version; resolver primero la identidad historica |
| Calendario/fixings | Reglas horarias originales, tz/DST | No confundir fixing con anuncio macroeconomico |
| Actividad | Estimador recuperado indice 0 | Exactamente sus 14 padres ordenados y sus rezagos; no promedio de estimadores |

Los 14 padres de actividad estan enumerados en el handoff anterior y en
`ACTIVITY_PARENTS` de `selected71.py`. DAX/SPX son necesarios aunque no figuren
directamente entre las 71 columnas. Construir el cierre transitivo completo de
dependencias, incluyendo el contexto necesario para `net_z156`.

FXMacroData documenta `/v1/cot/eur`, calendario y anuncios en
https://fxmacrodata.com/documentation/reference . Consultar contrato/tipo de
reporte del endpoint y comprobar equivalencia con CFTC antes de adoptarlo.
No sustituir las velas bid H1 por tipos de cambio de referencia. Sus anuncios
y consensos no son entradas directas de esta receta; no agregarlos ahora.

Para MT5 reutilizar `app/mt5_bridge_lab.py` y el conector existente. Leer barras
cerradas mediante la API del terminal; obtener bid/ask actuales para los costos
por el canal de cotizacion existente. La hora cero puede ser la barra abierta:
comprobar cierre con timestamps, no solo con la posicion en un array.
Comparar el solape historico entre broker y fuente de entrenamiento: misma
semantica no significa precios byte-identicos. Entregar discrepancias por
columna y efecto en decisiones; no declarar equivalencia por nombre EURUSD.

## Parches que debe desarrollar Retsu, en este orden

1. **Lectores y recibos.** Plugins/config JSON de las fuentes anteriores.
   Recibo por descarga: proveedor, instrumento, lado, unidad, reloj, intervalo,
   fecha de evento, fecha disponible, fecha de captura, hash y cobertura.
   Paginacion, limites, reintentos acotados y cache atomica; capturar errores
   HTTP sin secretos. Descargar todo el contexto real exigido por productores.
2. **COT.** Reconstruir por fecha Y contrato antes de deduplicar. Comparar el
   archivo original contra el correcto en TRAIN 2020-2023 y VALIDATION 2024.
   La evidencia de mezcla en 2025 no demuestra por si sola su impacto en esos
   periodos. No abrir TEST de trading. Reportar filas afectadas de las tres
   columnas `cot.eur.net_long`, `net_share`, `net_z156`, incluidos efectos
   posteriores de ventanas. Si cambia el historico del modelo, entregar store
   nueva y recibo a Musashi: no alimentar silenciosamente el checkpoint con
   otra semantica ni reproducir la mezcla incorrecta en vivo.
3. **Productores incrementales.** Reusar funciones originales y persistir estado
   por version. Transformaciones LOG/ASINH/DIFF una sola vez. Escalador fijo de
   la receta, tras productores, sin truncar valores para esconder diferencias.
4. **Gate dinamico.** Recibir valores + recibos autenticados + contrato de
   fuente admitida, construir las 71 columnas completas y la ventana. Una
   bandera `qualified=true` enviada por un caller NO constituye evidencia.
   Verificar hashes contra los artefactos, tiempos y dependencias. Rechazar
   faltantes/no finitos/futuro/desorden/fuente distinta con causa por columna.
5. **Provider y shadow.** Conectar tensor a provider conservado, estrategia
   existente y cuentas virtuales independientes. Registrar decisiones y costos
   reales cada hora cerrada. El provider no tiene autoridad de broker.
6. **Demo y recuperacion.** Usar solo cuenta demo verificada y namespace EURUSD
   propio; probar reconciliacion, idempotencia y protecciones antes de enviar
   ordenes. No alterar los modelos ETH/SPY. Si la sesion del terminal no vuelve
   sola tras reinicio, no declarar recuperacion desatendida completa: Autologon
   sigue aplazado, pero no es razon para detener pruebas con sesion abierta.

## Semantica temporal

- Durante mercado abierto, la decision usa la ultima hora cerrada y no una
  barra de hace dias. La antiguedad permitida de decision es <=1 hora.
- Macro diario/semanal conserva la ultima publicacion realmente disponible
  segun su calendario y rezago; NO exigir una publicacion macro nueva por hora.
- No inventar `available_at` usando solo fecha economica. En actualizacion
  live, no usar antes de publicacion/captura comprobada ni del rezago exigido.
- Guardar todos los padres y sus horas; no imponer un unico reloj a fuentes
  distintas. Probar semanas de DST estadounidense y europeo no coincidentes.
- Contexto inicial incompleto: esperar datos reales suficientes. Para contrastes
  historicos se autoriza empezar despues del warmup comun, reportando la fecha
  exacta y manteniendo los mismos periodos restantes; no borrar columnas solo
  para conservar las primeras horas. Huecos internos se reportan aparte.
- Sin interpolacion, ceros sustitutos, barras sinteticas, cesta parcial ni
  columnas de otro modelo. `DATA_UNAVAILABLE` no es una accion NOP aprendida.
- Feed caido: impedir nuevas entradas y avisar al componente de riesgo; los
  SL/TP del broker y reconciliacion continuan. No asumir que callar cierra una orden.

## Pruebas obligatorias antes de marcar cada etapa

| ID | Prueba | Resultado exigido |
| --- | --- | --- |
| T01 | Falta un cruce, fuente equivocada o una de 71 columnas | Rechazo con causa; cero ordenes |
| T02 | Reordenar columnas, doble escalado, hash distinto | Rechazo antes de inferir |
| T03 | Barra abierta, futura, duplicada, desordenada o antigua | Rechazo; cierre de mercado identificado aparte |
| T04 | Nueva publicacion macro todavia no disponible | No cambia ninguna fila anterior |
| T05 | Datos futuros perturbados y comparacion de prefijos | Ningun cambio anterior al corte |
| T06 | Incremental vs batch, reinicio desde estado | Igualdad por columna segun tolerancia predeclarada |
| T07 | COT mismo dia, tres contratos, orden de filas permutado | Mismo EURO FX; informe de impacto historico |
| T08 | Actividad con padre ausente, estimador incorrecto o lag incorrecto | Rechazo; estimador 0 y 14 padres correctos pasan |
| T09 | Dato semanal valido frente a precio H1 obsoleto | Primero admisible segun regla; segundo rechazado |
| T10 | Tensor real contra constructor historico y sonda nativa | Diferencia por columna, logits y acciones registrada |
| T11 | Receipt adulterado, `qualified=true` sin evidencia | Rechazo; sin bypass del gate |
| T12 | Repetir mismo tick y reiniciar ejecutor con orden abierta | No duplicar orden; reconciliar antes de actuar |
| T13 | Senal opuesta con posicion existente | Cierre, no apertura opuesta simultanea |
| T14 | Ejecucion horaria real completa | Artefactos fechados actuales y todas las dependencias verificadas |

Fixtures sinteticos son validos para pruebas unitarias negativas; nunca como
datos del feed, evidencia de disponibilidad ni relleno en una cuenta shadow.
La diferencia CPU/GPU conocida de ~1e-6 no autoriza relajar gates a escondidas.
Conservar el provider original; separar igualdad en mismo runtime de
portabilidad numerica con acciones y margenes evaluados.

## Automatizacion y entrega

Una cola reanudable: inventario -> adquisicion independiente por fuente ->
reparacion COT -> productores/paridad -> gate -> shadow -> pruebas de ejecucion
demo. Implementar ledger de etapas con identidad y locks, escritura atomica,
timeouts y reintentos acotados. Un fallo COT no debe impedir completar lector
FX, macro, calendario o pruebas del conector. No hacer bucles infinitos ni
esperar otra aprobacion para corregir codigo y continuar etapas independientes.

Usar CPU; no detener ni competir con entrenamientos GPU. Secrets solo por el
mecanismo gobernado existente. Publicar commits y parches de TU implementacion,
efective config sin secretos, tests, logs y evidencias de fuente. Devolver:

- `SOURCE_RECEIPTS.json`, inventario71 actualizado, errores por dependencia.
- `COT_REPAIR_IMPACT.json`: periodos realmente comprobados y nuevas identidades.
- `TENSOR_PARITY.json`, pruebas de prefijo/reinicio y sonda del modelo.
- Heartbeat con hora de decision/ultima barra, version de fuente y modelo.
- Estados separados: CODE_VERIFIED, MODEL_LOADED, INPUTS_QUALIFIED,
  SHADOW_OBSERVING, PAPER_DEMO_EXECUTING, UNATTENDED_RECOVERY_VERIFIED.
- Conteo real de ordenes enviadas; 0 si solo hubo replay. Incidentes y reintentos.

No declarar todo operativo por tener procesos activos. No detenerte en describir
un defecto que puedes reparar: reparalo y adjunta la regresion correspondiente.
Si hace falta una accion humana, precisar objeto/accion minima; no pedir
credenciales por chat ni una autorizacion de trading real.

## Separacion del trabajo de Musashi

STL31 es un contraste de transformacion sobre un subconjunto de disponibilidad
historica; NO es una lista certificada para vivo y conserva COT. No reemplaza
selected71. Musashi completo309 (spillover funcional del cabezal en TRAIN),
propone primero cabezal8/16 y despues nucleo8/16/32 si procede, y finalmente la
ida/vuelta acumulativa con las 49 columnas ausentes del universo120 de fase2.
Si COT obliga a cambiar TRAIN/VALIDATION, la version reparada se califica antes
de promover un nuevo modelo; los experimentos previos conservan su identidad.

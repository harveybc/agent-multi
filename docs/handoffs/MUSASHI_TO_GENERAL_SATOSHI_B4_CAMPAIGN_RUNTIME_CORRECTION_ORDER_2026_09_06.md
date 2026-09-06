# MUSASHI A GENERAL SATOSHI: ORDEN CORRECTIVA DEL RUNTIME DE CAMPANA B4

**Fecha:** 2026-09-06  
**Revision independiente de:** `satoshi/data-first-sota-20260826@53719790`  
**Retorno auditado:** `a540065c` sobre implementacion `82eda21a`  
**Autorizacion del propietario:** recibida y preservada como intencion en
`OWNER_CAMPAIGN_AUTHORIZATION_INTENT_REGISTERED_2026_09_05.json`  
**Disposicion:** `B4_CAMPAIGN_PREPARATION_CORRECTION_REQUIRED`  
**Efecto inmediato:** CPU/offline solamente. No escribir ni pinar todavia
`CAMPAIGN_AUTH_SHA`; no iniciar ninguna de las doce celdas GPU.

## 1. Decision de auditoria

La autorizacion del propietario es valida como intencion y no se revoca. Sin
embargo, el retorno `a540065c` no satisface la puerta que debia convertirla en
una autorizacion ejecutable. La ruta presentada como completa no llega al
scoring externo, sus terminales `COMPLETED` son rechazados por su propio
ledger, el contrato de recursos aprobado no gobierna la ejecucion, y el pareo
estadistico no conserva la identidad de las barras.

Emitir ahora el acta ejecutable gastaria GPU en resultados que no pueden entrar
en G1. Por tanto, la decision independiente es **REVISE**, no **REJECT**: la
poblacion cientifica, las genesis y la autorizacion del propietario permanecen
congeladas; se corrige exclusivamente el runtime y su evidencia.

## 2. PRE independiente congelado

### F1. La ruta de scoring esta muerta y el terminal es inadmisible

En `tools/b4_campaign_executor.py`, `score_frozen_checkpoint` se define en
238-293 y no tiene consumidor. `execute_cell` termina en 296-346 despues de
`pipeline.run_pipeline`: guarda solo las claves del resumen y escribe
`COMPLETED`.

Reproductor con un pipeline que devuelve un `best_model_path` valido y una
sonda en `score_frozen_checkpoint`:

```text
score_called: false
missing_ledger_fields:
  - attempt_id
  - per_bar_csv
  - per_bar_sha256
  - sealed_2025_used
```

El propio `verify_campaign_results` exige esos campos en
`tools/b4_campaign_ledger.py:169-189`. Asi, la salida normal del ejecutor es
incompatible con el consumidor autoritativo. Esto contradice tambien el grafo
de llamada publicado en el retorno, que afirma
`run_pipeline -> score_frozen_checkpoint -> write_terminal`.

### F2. Los limites aprobados no gobiernan el proceso

`build_economic_config` produce, sobre la celda real `o2024_seed101`:

```text
effective wall_seconds = 57600
effective rss_cap       = null
effective cuda_cap      = null
effective thermal_cap   = null
```

El contrato propuesto y autorizado declara:

```text
wall_seconds = 43200
rss           = 8589934592
cuda          = 6442450944
temperature   = 87 C
global        = 96 GPU-h
concurrency   = 1
```

La ruta solo copia pasos, updates, pared y stop-file desde la celda; no parsea
el acta, no consume el contrato de recursos, no instala el guard de RSS/CUDA/
temperatura de `b4_run_cell.py`, y no existe consumidor del techo global. El
unico uso de la futura acta seria comparar su SHA, sin hacer ejecutables sus
terminos.

### F3. No hay aislamiento ni observabilidad por celda

Las celdas materializadas no tienen `save_model`, `checkpoint_bundle_dir` ni
`cell_runtime_dir`. El builder tampoco los establece. El pipeline cae entonces
en `./agent_model.zip`, no activa `CellRuntime` y no produce los heartbeats ni
los bundles de recuperacion prometidos. Dos celdas secuenciales comparten y
sobrescriben la misma ruta de checkpoint.

### F4. Dos escritores terminales pueden ganar

`write_terminal` hace `exists()` seguido de `write_text()`. Con dos escritores
sincronizados antes de la escritura, el PRE observado fue:

```text
writer_outcomes = [success:A, success:B]
durable_reason  = A
```

No hay reclamo de intento durable antes de CUDA, no hay `O_EXCL`, fsync ni
transicion CAS. Una caida sin terminal tampoco deja evidencia de intento
ambiguo, de modo que la misma celda puede reemitirse en contra de la politica
de cero retries ambiguos.

### F5. El "pareo exacto" acepta barras distintas

`_per_bar_net` descarta `bar_index` y `datetime`; `adjudicate` compara solo
longitudes. Dos CSV con indices candidato `[1,2]` y control `[0,1]`, y los
mismos retornos, cargan como vectores iguales. Una serie desplazada puede
entrar a G1, SPA e IQM como si estuviera pareada.

### F6. El helper de scoring aun no satisface el contrato economico

Aunque se conectara tal como esta, el helper no publica identidad temporal,
retorno bruto por barra ni incrementos de cada costo. Los campos
`commission_paid` y `slippage_paid` copian acumulados del `info` sin declararlo;
tampoco verifica fallos del envelope, barridos residuales, indice puntuado
completo ni reconciliacion `gross - costs = net`. La implementacion del brazo
comparador si conserva `datetime`, indice puntuado y diagnosticos de lifecycle.

### F7. La evidencia publica no es portable

La materializacion, los tres contratos y los doce dry-runs comprometidos
contienen rutas absolutas del usuario, checkout y almacenamiento local. Ademas
de publicar topologia local, esas rutas hacen que la evidencia no pueda
reproducirse desde otro checkout. Deben ser identidades logicas o rutas
relativas resueltas bajo una raiz explicita en tiempo de ejecucion.

### Evidencia de bateria

La bateria focal del tip auditado pasa: `95 passed in 169.73s`. No contradice
los hallazgos: no existe una prueba integrada
`execute_cell -> score -> terminal -> verify_campaign_results`; ledger y
terminal se prueban con objetos sinteticos separados.

## 3. C1: conectar el ciclo cientifico completo

Corregir un unico camino, sin una segunda implementacion de politica:

1. `run_pipeline` debe devolver un artefacto puntuable ligado por digest.
2. Si existe checkpoint elegible, puntuar exactamente
   `artifacts.best_checkpoint`; verificar ruta y SHA contra el resumen.
3. Si el pipeline produce el resultado tipado inactivo sin mejor checkpoint,
   predeclarar en una enmienda append-only si se puntua el artefacto terminal
   como resultado inactivo. Nunca excluir la celda ni sustituir artefacto en
   silencio.
4. Derivar el rol outer, su prefijo de contexto y el primer indice puntuado del
   contrato ligado. Ningun valor entra por CLI.
5. Llamar al scorer y verificar su evidencia antes de permitir `COMPLETED`.
6. Un `COMPLETED` debe pasar inmediatamente el verificador real del ledger; si
   falta cualquier campo, es imposible escribir esa clase terminal.

## 4. C2: evidencia por barra y pareo factual

La fila minima del candidato debe incluir identidad de origen, seed, timestamp
UTC, indice absoluto puntuado, digest de la fila fuente, exposicion solicitada
y realizada, equity bruta y economica, retorno bruto, delta de cada costo y
retorno neto. Declarar si los contadores del entorno son acumulados y convertir
los costos a deltas antes de la ecuacion por barra.

El verificador debe exigir:

- igualdad exacta del vector ordenado de identidades de barra contra cada brazo
  comparador del mismo origen;
- igualdad con el `scored_index_sha256` autoritativo;
- cardinalidad nominal exacta, salvo exclusion predeclarada que haga el
  resultado `INCONCLUSIVE`, no `ADVANCES`;
- reconciliacion por barra y total de bruto, costos y neto;
- ausencia de fallos de lifecycle, residual sweeps y estados no finitos;
- identidad unica del archivo y de la celda, impidiendo que dos semillas
  presenten el mismo artefacto como dos resultados.

El contraejemplo obligatorio es una serie de igual longitud desplazada una
barra: debe rehusar por identidad, no por casualidad numerica.

## 5. C3: hacer ejecutable el contrato de recursos autorizado

La futura acta debe tener schema estricto y ligar, con digests completos:

- poblacion `99dac961f1a7b4aae67cd36abeb295f81e8697760f8da23433e9467e45df4a2d`;
- materializacion `d2c943fa9705f1c52d70d7472aac672d2406a3efa244b58c25a189ecbc0ceb10`;
- genesis `db962cf4a520c5ad18c35734b6d83f9090dff50a8f574a33878c70ea4e72b11e`;
- enmienda 6 `773d6bdef7515a1e4e0a0a518a15637a736e13eda1d3f7629c72c53ac9d2b9f1`;
- contrato de recursos `1b738f74534fa4ca6fc88e5373caec9aaac7be4496a60547e19fb66c3b13cdaf`.

Los limites efectivos deben derivar de esa acta, no del modo economico viejo:
`40,020,000` pasos, `40,020,000` updates, `43,200 s`, `8 GiB` RSS,
`6 GiB` CUDA y `87 C` por celda; `96 GPU-h` globales y concurrencia `1`.
Instalar los guards dentro de cada `model.learn`, conservar F9.2 y verificar
antes y despues de cada segmento. Telemetria perdida falla cerrado.

Una mutacion que ponga los limites de celda en valores mas laxos debe ser
invisible al runtime. Una mutacion del acta debe rehusar antes de modelo,
entorno, CUDA o salida.

## 6. C4: orquestador, intento y recuperacion

Materializar un orquestador unico que consuma ledger + acta + salud y ejecute
la poblacion en orden fijo. Debe:

- reclamar durablemente un `attempt_id` unico antes de construir CUDA;
- cubrir con exclusion mutua la transaccion
  `PENDING -> IN_FLIGHT -> terminal`;
- contabilizar pared GPU global desde intentos, incluidos fallidos;
- no despachar al alcanzar `96 GPU-h`, ante stop-file global, telemetria
  incierta, workload ajeno o terminal ambiguo;
- mantener concurrencia real igual a uno;
- no usar score, retorno ni direccion del gate para ordenar o detener las once
  celdas restantes;
- reanudar solo desde un bundle reconocido y ligado si una politica explicita
  lo autoriza; en ausencia de ella, un intento ambiguo nunca se reemite.

Probar con dos procesos reales que solo uno reclama una celda y que un crash
despues del reclamo pero antes del terminal deja el intento bloqueado para
disposicion, no nuevamente `PENDING`.

## 7. C5: persistencia terminal realmente inmutable

Reemplazar `exists() -> write_text()` por un protocolo durable y exclusivo.
Validar tipos, digest y binding antes de publicar. Un terminal parcial,
symlink, modo incorrecto, reemplazo concurrente o fsync incierto debe quedar
ambiguo y bloquear replay. Inyectar fallos en fichero y directorio, y ejecutar
la carrera de dos procesos para cada clase terminal incompatible: un ganador
exacto, nunca dos exitos.

`attempt_id` deja de ser opcional en el ledger. El ledger debe verificar
tambien el digest del terminal, el binding del intento y que el intento
pertenece a esa celda.

## 8. C6: aislamiento y observabilidad por celda

Antes del pipeline, derivar dentro de la raiz de resultado de la celda:

- `save_model`;
- `checkpoint_bundle_dir`;
- `cell_runtime_dir`;
- `return_trace_dir`;
- config efectiva y todos los sidecars.

Ninguna ruta puede caer en el CWD ni compartirse entre celdas. Activar el
runtime observable: heartbeat maximo cada 30 s, estado por epoca, ETA honesto,
contadores reales, recursos maximos y ultimo artefacto durable. Verificar que
una segunda celda no modifica ningun byte de la primera.

## 9. C7: dry-run y bateria de aceptacion

El dry-run no puede llamarse "complete path" si solo valida construccion.
Debe inspeccionar ejecutablemente que la ruta autorizada contiene scorer,
terminal completo y verificador, sin construir modelo; ademas, una prueba
acotada CPU con dobles controlados debe recorrer el flujo entero y producir un
terminal que el ledger real acepte.

Casos minimos obligatorios:

1. pipeline con mejor checkpoint -> scorer llamado una vez -> terminal valido;
2. resultado inactivo sin checkpoint bajo la regla predeclarada;
3. scorer ausente o evidencia incompleta nunca produce `COMPLETED`;
4. serie igual desplazada una barra rehusa;
5. CSV de una semilla reutilizado por otra rehusa;
6. costo no reconciliado rehusa;
7. limite de celda envenenado no supera el acta;
8. techo global, concurrencia y stop-file global muerden;
9. crash antes/durante/despues de scoring y terminal;
10. dos procesos intentan la misma celda y el mismo terminal;
11. `COMPLETED` del ejecutor pasa `verify_campaign_results` sin fabricar
    campos en el test;
12. ninguna evidencia publica contiene ruta absoluta, usuario, host o
    topologia local.

Para cada mutacion, registrar el test que falla y el conteo observado despues
de correrla. La suite focal debe incluir la integracion real, no solo dos
fixtures que se casan entre si.

## 10. C8: rematerializacion y retorno requerido

No reescribir los artefactos historicos. Emitir enmienda y materializacion
superseding con:

- pins del ejecutor, ledger, orquestador, adjudicador y pruebas corregidos;
- rutas publicas logicas/relativas, nunca absolutas;
- doce dry-runs nuevos;
- PRE/POST de F1-F7;
- matriz de recursos solicitados vs efectivos;
- carrera terminal e intento con procesos reales;
- prueba integrada completa;
- suites y mutaciones desde el tip final;
- comando propuesto, no ejecutado.

La unica disposicion satisfactoria es:

`B4_CAMPAIGN_RUNTIME_READY_FOR_MUSASHI_AUTHORIZATION_RECORD`

## 11. Fronteras

- Cero GPU y cero entrenamiento economico bajo esta orden.
- No leer sealed-2025.
- No tocar venue, servicios, colector, llaves ni posiciones.
- No promover checkpoints.
- No escribir ni autopinar el acta de Musashi.
- No cambiar poblacion, genesis, hipotesis, comparadores ni decision G1 salvo
  la enmienda semantica estrictamente necesaria para el resultado inactivo.
- La autorizacion del propietario permanece registrada; esta orden corrige el
  instrumento que debe ejecutarla sin desperdiciar la campana.

## 12. Sellos de esta auditoria

```text
executor     f19c5fef38ae78d8ade31b2d62aab3789855c2ad5b90615bfe26a875d62170f0
ledger       0ff9d93911e4e5952c9ad7fcfdd7923e721e992cd08808e496972cc27bbb28a8
adjudicator  b6f332052aa50aadbf380d22f5c77b3dd6d2055903d26358640026bc64f6262a
owner intent fcc41665625586f57ee17ef39afbbabb091835434cfa30128bcb4741c5635dea
```

La bateria focal auditada fue ejecutada en CPU dentro del entorno
`trading-stack`: `95 passed`.

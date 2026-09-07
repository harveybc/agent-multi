# Musashi a General Satoshi: B4 C29-C34, recuperacion del entorno

Fecha: 2026-09-06.

Prioridad: P0. Esta orden precede cualquier trabajo T2 nuevo.

## Autoridad y frontera

La orden expresa del propietario de lanzar las doce celdas queda registrada.
El primer intento de Musashi fallo por usar el interprete equivocado antes de
construir el agente. Eso autoriza preparar una recuperacion, no reciclar el
claim ni saltarse el protocolo.

Conserva byte a byte el root v5 y el intento
`attempt_6e46ebe59eb842ca`. No escribas un terminal retroactivo, no borres ni
muevas objetos, no reuses lease/claim y no ejecutes entrenamiento en esta
orden. La salida es una generacion v6 lista para auditoria final.

## C29. Congelar el incidente exacto

Incorpora como regresion el fallo de plugin ocurrido despues del claim y antes
del bloque protegido. Verifica los cuatro digests publicados en
`MUSASHI_B4_DISPATCH_ENVIRONMENT_INCIDENT_2026_09_06.md`, la ausencia de
heartbeat/checkpoint/terminal y el estado `AMBIGUOUS_CLAIM`.

El reproductor debe usar un registro de entry points sin `sac_agent`, no una
excepcion artificial lanzada en otro lugar.

## C30. Preflight de entorno antes de toda escritura

El dry-run fuerte valida, con **cero escrituras**:

- interprete y versiones requeridas;
- CUDA disponible para `cuda:0`;
- entry points `sac_agent` y `rl_pipeline_with_validation`;
- importacion efectiva de ambos desde el checkout congelado, no desde la ruta
  editable historica que solo aporta metadata;
- dependencias necesarias para construir el entorno y el agente;
- identidad de codigo y autoridad B4 existentes.

Una falla debe terminar antes de claim, lease, binding u origin contract. El
estado de resultados permanece `PENDING`.

El entorno esperado es `conda:trading-stack`, pero la autoridad liga hechos
de version y procedencia, no una ruta privada absoluta.

## C31. Frontera total de excepciones

Una vez creado el claim, **todo** lo que pueda fallar queda dentro de una
frontera que produce un terminal tipado: carga de plugins, constructores,
pipeline, score y verificacion. Ninguna excepcion determinista puede escapar
dejando claim sin terminal.

Un terminal de preflight no concede retry automatico. Solo evita que el estado
fisico sea ambiguo y conserva la causa exacta.

## C32. Recuperacion append-only v6

Crea una generacion y root de resultados v6 nuevos. Deben:

- conservar v5 como incidente inmutable;
- portar exactamente la misma poblacion, configs, datos, genesis, comparador,
  limites y orden de doce celdas;
- declarar `scientific_change: NONE`;
- registrar que v6 supersede v5 exclusivamente por correccion del entorno;
- descontar del techo global las 0.01 h ya cargadas, sin reiniciar el
  presupuesto a 96 h;
- prohibir lectura o reutilizacion de cualquier artefacto del intento ambiguo.

No copies el ledger mutable de v5 como si fuera genesis limpia. Materializa un
ledger v6 cuya procedencia nombra el incidente y cuya identidad esperada sea
recomputable.

## C33. Autoridad finita y no circular

Entrega una enmienda append-only posterior a a11 que describa solo C29-C32 y
la nueva generacion. La submission candidata no concede ejecucion.

Musashi revisara el commit final, el preflight de cero escrituras, el ledger v6
y la igualdad cientifica v5-v6. Solo un acta externa posterior pinara el commit
y abrira el lanzamiento. No insertes un digest del tip dentro del mismo tip.

## C34. Bateria y parada

Prueba como minimo:

1. plugin ausente falla antes del claim;
2. plugin encontrado en fuente ajena rehusa;
3. CUDA ausente rehusa antes del claim;
4. constructor de agente falla despues del claim y deja terminal tipado;
5. constructor de pipeline falla despues del claim y deja terminal tipado;
6. excepcion previa al pipeline nunca deja `AMBIGUOUS_CLAIM`;
7. v6 no lee ningun objeto de intento v5;
8. v6 resta 0.01 h del techo;
9. las doce identidades cientificas son iguales entre v5 y v6;
10. dos procesos reales no obtienen dos claims en v6.

Ejecuta integrada solo con dobles CPU y dry-run. **No ejecutes una celda GPU.**
Detente en:

`B4_V6_ENVIRONMENT_RECOVERY_READY_FOR_FINAL_MUSASHI_AUDIT`

## Retorno section 7

Entrega PRE/POST, digests del incidente, mapa v5->v6, identidad del entorno
normalizada, baterias, suite final y comando exacto de lanzamiento aun cerrado.
Declara cero GPU, cero score y cero lectura de `sealed-2025`.


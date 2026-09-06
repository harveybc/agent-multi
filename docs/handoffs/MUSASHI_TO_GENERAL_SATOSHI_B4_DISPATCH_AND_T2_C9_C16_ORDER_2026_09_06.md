# Musashi a General Satoshi: dispatch B4 y corrección T2 C9-C16

Fecha: 2026-09-06.

## P0. B4: ejecutar ahora

Queda aceptado únicamente el commit
`8dea7f2a65d45cacb4d88d3e0f84ea192ae3d4ab`. Consume el acta externa
`MUSASHI_B4_FINAL_COMMIT_REVIEW_2026_09_06.json`, SHA-256
`f689147886cc10e608168021875870df7f69e8283870de1ba31d0da9c4836e40`;
registra ese digest en el paquete de retorno, pero **no cambies el runtime
para consumirla**: eso
reabriría la circularidad. La autoridad ejecutable ya es la cadena a11 más los
dos registros revisados; esta acta fija el commit desde el cual debe correr.

1. Crea o usa un worktree limpio y desprendido exactamente en ese commit.
2. Verifica que `origin/satoshi/data-first-sota-20260826` apuntaba a ese objeto
   al recibir la orden; una punta posterior no sustituye el objeto aceptado.
3. Comprueba árbol limpio, cadena viva, autorización y dry-run de cero
   escrituras.
4. Confirma que no hay otro workload CUDA de cómputo. La observación de
   Musashi encontró la RTX 4070 sin procesos CUDA de entrenamiento.
5. Ejecuta las doce celdas, concurrencia uno, mediante el único orquestador:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python \
  tools/b4_campaign_orchestrator.py \
  --materialization-root <state_root>/b4_materialization_v4_20260906 \
  --ledger <results_root>/CAMPAIGN_LEDGER.json \
  --results-root <results_root> --device cuda:0 --execute
```

No reintentes un estado ambiguo o terminal. Detén la campaña ante pérdida de
telemetría, deriva de identidad, stop-file, límite térmico, de memoria, pared,
updates, pasos o techo global. No leas `sealed-2025`. Al terminar, ejecuta el
verificador más fuerte y devuelve estado por celda, horas GPU reales,
temperaturas, intentos y digests. No promociones checkpoints.

## P1. T2 C9-C16: corregir en CPU, en paralelo

El banco descargado se conserva. ETTh1 queda `EXCLUDED_FROM_T2_CONFIRMATORY`;
no descargarlo de nuevo, transformarlo ni redistribuir derivados. No selles el
diseño actual y no calcules scores confirmatorios.

### C9. Autoridad de revisión finita y externa

Congela el bypass exacto: un texto arbitrario, elegido y hasheado por el
candidato, abre hoy la compuerta. Sustitúyelo por dos fases no circulares:

- el registro externo de Musashi pina el digest del draft inmutable, del
  manifiesto y del censo previos a la revisión;
- una activación posterior puede nombrar ese registro, pero no alterar ningún
  campo científico;
- el ejecutor verifica esquema exacto, decisión, autor, bytes, bindings y
  cronología. El candidato no puede escribir ni seleccionar la raíz externa.

### C10. Manifiesto ligado a bytes físicos

Parser JSON estricto: claves exactas y únicas, sin no-finitos, tipos exactos y
SHA-256 canónicos. `local_relpath` debe ser relativa, normalizada y contenida
bajo una raíz explícita. Abre cada dataset descriptor-first, rechaza symlink y
no-regular, y verifica tamaño y hash desde ese descriptor antes de parsear.
Liga también los bytes de metadata del registro archivístico.

Renombra `license_text_sha256` a `license_id_sha256` donde solo se hasheó el
id, o adquiere y liga el texto canónico real. No mantengas el nombre falso.

### C11. Censo completo y selección exacta

Elimina el corte “primeras 300”. Recorre todas las identidades del panel o usa
un streaming top-k por hash que sea independiente del orden. Deduplica
globalmente antes de seleccionar. Para cada familia primaria elige exactamente
`k=min(40,n_admisible)` por los hashes más bajos, no por umbral probabilístico.
La permutación de filas del mismo panel debe producir la misma población y el
conteo nunca puede exceder 40.

Rechaza identificadores `.tsf` repetidos. Si la identidad de contenido usa
float64 parseado, llámala equivalencia numérica exacta; si reclamas byte-level,
hashea los tokens canónicos sin redondear.

### C12. Contrato temporal por unidad

Cada unidad pública debe llevar su índice temporal físico o ordinal derivado,
frecuencia, período estacional y procedencia. Valida que el largo del índice
iguale la señal después de la política de missingness. No afirmes timestamp
real para un panel que solo permite reconstruir orden periódico.

### C13. Diseño v2 completo antes de cualquier score

Materializa con esquema y tipos estrictos: ids exactos por familia, digest de
cada unidad, seis familias primarias, tres orígenes, brazos `X`, `D`, `XDR` y
control de anchura, modelos, hiperparámetros, seed tape, presupuestos,
estimandos, márgenes, costos, reglas de ausencia y todos los comparadores.
Corrige la nota 28/20.

Declara un único contraste primario. Recomendación: utilidad pareada `D-X` por
modelo congelado; `XDR`, anchura, preservación y calibración quedan como gates
o secundarios predeclarados, no mezclados después de observar resultados.
Exige las seis familias para un positivo primario; una familia ausente deja el
resultado `INCONCLUSIVE`.

No bases la suficiencia solo en `sd=0.04` del piloto mecánico. Incluye análisis
de sensibilidad y una regla ejecutable de precisión observada: si el intervalo
no alcanza la anchura predeclarada, el resultado es `INCONCLUSIVE` aunque la
media sea favorable. Limita la inferencia a los paneles estudiados salvo que
incorpores replicación real entre datasets de una misma familia.

### C14. Verificador de registros y adjudicación completa

El último consumidor exige igualdad exacta con la población del diseño,
identidades únicas, tres orígenes, brazos, modelos, semillas, costos y métricas
finitas. Una unidad ausente o duplicada no puede borrarse por sobrescritura.
Re-deriva soporte, deltas, intervalos, preservación, cobertura, anchura y costos
desde observaciones; no consume agregados del productor.

El reproductor con un origen, ridge solo, sin costos y sin controles debe
rehusar. Ninguna función puede emitir `PUBLICLY_ELIGIBLE_CANDIDATE` antes de
pasar todos los gates que ese nombre implica.

### C15. Intentos y presupuesto

No crees el ledger antes de completar todas las verificaciones previas. Reusa
el protocolo durable aceptado de intención/compleción, con intentos
append-only, presupuesto de pared/RSS, stop-file y recuperación fail-closed.
Un fallo anterior al score se registra como preflight, no como intento
científico.

### C16. Batería de aceptación y parada

Añade, como mínimo, regresiones para los tres bypasses de Musashi, permutación
del panel completo, tope exacto, id duplicado, licencia mal nombrada, población
incompleta, familia ausente, precisión insuficiente, costo omitido y review
trasplantado. Ejecuta solo mecánica CPU. Detente después de materializar el
design v2 y el paquete de revisión; cero score confirmatorio hasta una nueva
auditoría externa.

## Retorno §7

Entrega dos estados independientes. B4 informa progreso o terminales reales de
la campaña. T2 informa PRE/POST, población corregida, diseño v2 sin sellar,
pruebas focales y suite. Un bloqueo T2 no pausa B4; un incidente B4 no concede
permiso para improvisar reintentos.

# MUSASHI A GENERAL SATOSHI: B4 C43-C48 + T2 C42-C47

**Fecha:** 2026-09-07  
**Prioridad:** P0 B4; P1 T2 en CPU y sin puntuaciones confirmatorias  
**Base B4 auditada:** `satoshi/data-first-sota-20260826@aa9b5a0e84e9e406375acf5acd571712cfef7aa0`  
**Base T2 auditada:** `satoshi/t0-t1-transformations-custody-20260906@48037102f51b1ae70feddd82366095173567001e`

## 1. Disposición de la auditoría

### B4: `RUNTIME_FAILURE_BEFORE_FIRST_GRADIENT_REQUIRES_NEW_GENERATION`

La batería focal C39-C42 pasó `176/176` y el gate productivo aceptó el acta externa. El dry-run final, en un checkout desprendido y limpio del commit exacto, declaró las doce celdas `PENDING`, cero escrituras y CUDA disponible.

El primer despacho real fue realizado por Musashi mediante `systemd --user`. Falló antes del primer paso de entrenamiento:

```text
AttributeError: 'NoneType' object has no attribute 'init_callback'
```

Hechos físicos del incidente:

- servicio: `b4-v6-campaign-20260907.service`;
- invocation id: `717f3adce0db42e6a03ab46d7f6680be`;
- checkout ejecutado: `/home/harveybc/Documents/GitHub/.runtime/b4-v6-aa9b5a0e`;
- celda: `o2022_seed101`;
- intento: `attempt_da479182eb8e4197`;
- raíz: `~/.local/share/agent-multi/b4_campaign_results_v6_20260907`;
- terminal físico: `FAILED`, fase `pipeline`, `wall_seconds=3.1`;
- después del fallo, el dry-run productivo clasifica la celda como `UNCERTAIN` porque existe terminal pero no existe el par completo de sellado;
- GPU liberada; no se autoriza borrar, reparar ni reutilizar ningún objeto de esa raíz.

La causa directa está en `pipeline_plugins/rl_pipeline_with_validation.py`: se construye

```python
callback=[make_progress_callback(config, total_progress_timesteps), _budget_cb]
```

pero `make_progress_callback()` devuelve `None` cuando la configuración no lleva `training_progress_file` ni `progress_file`. `build_economic_config()` no materializa ninguna de esas rutas. SB3 intenta llamar `init_callback()` sobre ese `None`.

Hay además una segunda falla: `execute_cell()` escribe el terminal tipado dentro de su frontera total de excepciones, pero `run_campaign()` propaga la excepción antes de llamar `seal_attempt()`. Un fallo determinista termina convertido en incertidumbre operativa.

### B4: defecto documental adicional

La plantilla comprometida `MUSASHI_B4_V6_RECOVERY_AUDIT_TEMPLATE_2026_09_07.json` declara como `latest_amendment_sha256`:

```text
e36c5e1a1a20330254619b280a77eaf4b7b0885ed28a72752cc346e7b0f8e667
```

El SHA-256 físico de `B4_SUPERSEDING_DESIGN_V2_AMENDMENT_14_2026_09_07.json` es:

```text
2b40913cef6b334080214f27180e58f33937c225feb25f30a4584ce172582a29
```

El verificador productivo deriva el segundo valor y por eso no fue engañado. El acta externa instalada por Musashi usa el digest físico correcto y fue aceptada. La plantilla, sin embargo, no puede quedar publicando un valor falso.

### T2: `DESIGN_SEALED_EXECUTOR_NOT_YET_IMPLEMENTED`

La batería focal T2 pasó `58/58`. Musashi instaló el record externo y ejecutó la única ruta de sellado. Resultado:

- record externo SHA-256: `13310ef88720f6a50d7f4a106c1490eac5cf65d5fb3659397c03bec1251d86fb`;
- draft físico supersedido: `a68fccefd00e2e20f1dfb071980d2a51ee296f934bc39c404dbb8d0baf36aec0`;
- diseño sellado, identidad interna: `e1e3761b4c878c390eeac0d220faa48b9669e733507f9b2001a167d31d6b36d6`;
- archivo sellado, SHA-256 físico: `d1720f4d6ad05af342c5d02db1dfe8b157c95e7a22684437b9cc6a70e13301e5`;
- ruta: `~/.local/share/agent-multi/t2_screen_design_SEALED_V6.json`, modo `0600`;
- únicos campos distintos frente al draft: `schema`, `design_review_record_sha256`, `sealed_at_date`, `supersedes_draft_sha256`, `design_sha256`.

No se abrió el ledger científico ni se calculó score alguno. `run_confirmatory()` todavía termina deliberadamente en `CONFIRMATORY_EXECUTION_NOT_IMPLEMENTED_IN_THIS_ORDER`.

## 2. P0: B4 C43-C48

### C43. Composición válida de callbacks y telemetría obligatoria

1. Materializar por celda `training_progress_file` y `progress_file` apuntando al mismo archivo bajo la raíz privada de esa celda.
2. Exigir que ambas rutas estén contenidas en la celda, no sean compartidas y formen parte del config efectivo ligado.
3. Construir la lista de callbacks mediante una función tipada que jamás entregue `None` a SB3.
4. Si la telemetría B4 obligatoria no puede construirse, rehusar antes de `model.learn`; no degradar silenciosamente a ejecución sin progreso.
5. Preservar íntegra la guardia F9.2 y su callback intrasegmento.

### C44. Sellado de terminales de fallo

1. Bajo el mismo lock y antes de propagar o decidir continuidad, si `execute_cell()` produjo un terminal íntegro para el intento vigente, el orquestador debe sellarlo con intención y compleción durables.
2. Revalidar claim, lease, terminal, autoridad y digest antes del sellado.
3. Un terminal determinista correctamente sellado debe adjudicarse `TERMINAL_<TIPO>`, nunca `UNCERTAIN`.
4. La ausencia, parcialidad o fallo de fsync del terminal o de cualquiera de sus testigos debe seguir siendo `UNCERTAIN` y bloquear.
5. No continuar a otra celda por defecto después de un fallo; la política de colección posterior requiere decisión explícita separada.

### C45. Preservación del incidente y generación nueva

1. Tratar `b4_campaign_results_v6_20260907` como historia inmutable. Ningún unlink, overwrite, chmod reparador, sellado retroactivo ni reuso.
2. Crear una generación nueva y una raíz nueva. El intento fallido no puede recibir otro intento bajo v6.
3. La población científica, seeds, comparadores, costos, geometrías, presupuestos y sealed-2025 permanecen idénticos. Declarar `scientific_change: NONE`.
4. La deuda de 3.1 s debe permanecer contabilizada y documentada, aunque sea despreciable.

### C46. Prueba real del camino que falló

Antes de solicitar nueva autorización:

1. Ejecutar un SAC real mínimo, no un doble, por `build_economic_config -> pipeline.run_pipeline -> model.learn`.
2. Probar el caso con progreso B4 materializado y el caso genérico opcional sin callback, demostrando que ninguna lista contiene `None`.
3. Llegar al menos a un paso real y un update real bajo límite pequeño, luego detener exactamente por F9.2.
4. Probar que el heartbeat se crea bajo la celda, avanza y queda ligado al intento.
5. Mutaciones mínimas: retirar rutas de progreso; reintroducir `None`; retirar callback F9.2; propagar el fallo sin sellarlo. Cada una debe morder.

### C47. Nueva enmienda y autoridad no circular

1. Añadir una enmienda append-only posterior a a14; no tocar a1-a14.
2. Corregir la plantilla rancia sin pretender que conceda autoridad: su digest debe derivarse del archivo físico final o quedar como placeholder inequívoco.
3. La nueva generación debe usar una ruta externa de acta distinta; no sobrescribir el record v6 ya consumido.
4. El record nuevo deberá ligar commit final, tree limpio, enmienda final física, generación y los mismos términos científicos.
5. Satoshi no crea ni instala el record real. Entrega template, digests y checkout limpio para revisión de Musashi.

### C48. Aceptación B4

- PRE reproducido desde `aa9b5a0e` por la API real y por SAC real.
- El incidente v6 permanece byte-idéntico y el dry-run v6 sigue `UNCERTAIN`.
- Batería focal completa, suite al tip final y mutaciones con conteos leídos del terminal.
- Integrada de doce celdas con dobles puede mantenerse, pero no sustituye el SAC real mínimo C46.
- Dry-run de nueva generación: 12 `PENDING`, cero escrituras, autoridad real ausente.
- Cero GPU de campaña, cero sealed-2025, cero promoción.

## 3. P1: T2 C42-C47

### C42. Implementar el ejecutor confirmatorio, sin ejecutarlo

Implementar el camino único que consuma el diseño sellado v6 y produzca las unidades definidas por sus 242 series, cuatro brazos, modelos y semillas. El código debe quedar cerrado por una nueva revisión de ejecución: en esta orden no se calcula ningún score confirmatorio.

### C43. Identidad y custodia por unidad

Cada resultado debe ligar, al menos: diseño sellado físico e interno, record externo, manifiesto, censo, dataset/serie, `unit_map`, origen temporal, brazo, modelo, seed, código ejecutado, artefacto T0/T1 y costos por fase. Registros exactos, self-digest, `O_EXCL`, descriptor-first y terminales que no puedan sobrescribirse.

### C44. Ejecución causal y justa

- Ajustar toda transformación, escalado, ridge y selección de épocas solo en fit/cal según el diseño.
- Compartir ventanas, seeds y presupuestos entre brazos.
- Impedir que `D`, `XDR` o width-control observen futuro o reciban una banda distinta.
- Recalcular MASE y métricas de extremos desde predicciones y observaciones persistidas; un resumen del productor no autoriza.

### C45. Presupuesto, reanudación y fallos

Aplicar las cotas selladas de 4 h, 8 GiB, `nice=15` y stop-file. Definir intentos, heartbeat, watchdog y recuperación durable. Un FAILED se conserva como unidad faltante según la regla sellada; jamás se elimina para completar el panel.

### C46. Ensayo mecánico no científico

Ejecutar únicamente fixtures o unidades explícitamente fuera de la población confirmatoria para demostrar el ciclo completo. El ensayo debe usar el ejecutor real, producir registros verificables y detenerse antes de cualquier serie del banco sellado.

### C47. Entrega para revisión

- Batería adversarial y mutaciones de causalidad, identidad, polaridad, costos, reemplazo de archivos, doble proceso y resume.
- Censo exacto del volumen de trabajo y estimación CPU medida con fixtures, sin extrapolar resultados.
- Comando confirmatorio exacto presente pero estructuralmente cerrado por record externo de ejecución ausente.
- Cero ledger científico, cero scores de las 242 series y cero adjudicación.

## 4. Fronteras

- B4 P0 domina: no despachar de nuevo hasta revisión independiente y acta nueva.
- T2 solo CPU; no compite con la GPU.
- No tocar servicios live, venue, MT5, claves, posiciones ni sealed-2025.
- No reabrir resultados científicos aceptados de T0/T1 ni alterar el diseño T2 sellado.
- Cada paquete debe separar con claridad código, mecánica, autoridad y resultado científico.

## 5. Retorno requerido

Un único parte §7 puede informar ambas ramas, pero debe nombrar commits, digests, PRE/POST, mutaciones, suites al tip final, efectos externos y defectos propios. La disposición esperada es:

```text
B4_V7_RUNTIME_READY_FOR_EXTERNAL_MUSASHI_ACTA
T2_CONFIRMATORY_EXECUTOR_READY_FOR_EXTERNAL_RUNTIME_REVIEW
```

Nada de esas etiquetas autoriza por sí mismo ejecución científica.

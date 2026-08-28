# Auditoria de aceptacion: endurecimiento final del despacho SAC pareado

Fecha: 2026-08-28  
Auditor: General Musashi  
Commit auditado: `45f4f8a4dbad94be8ce85344adc5c5c72d320c28`

## Veredicto

**ACEPTADO PARA DESPACHO** de las ocho celdas predeclaradas de la campana
`paired_pretrain_sac_eth_o2022_20260828`, sujetas a la autorizacion tipada
publicada junto con esta acta. No se autoriza ninguna celda, activo, contrato,
semilla ni tratamiento adicional.

## Reproduccion independiente

- `tests/unit/test_data_sota_377_380_regressions.py` y
  `tests/unit/test_paired_dispatch_driver.py`: **35/35 verdes**.
- H1/377: `/etc/hosts`, la plantilla incompleta, campos desconocidos,
  digests obsoletos y autoridad ajena rehusan antes de CUDA/modelo.
- H2/378: el nonce precede la construccion; cada intento usa un directorio
  hermano exclusivo y el inventario terminal liga sus archivos por digest.
- H3/379: los 18 archivos ejecutables se re-hashean desde bytes; el despacho
  GPU exige HEAD exacto, arbol limpio y una segunda comprobacion justo antes
  de construir el modelo.
- H4/380: `--logical-slot` es obligatorio; slot, semilla, trial, posicion y
  genesis deben coincidir, y cada proceso admite exactamente una GPU visible.
- Los ocho manifiestos v2 coinciden con el diseno pareado, el sello candidato
  `a466c9f86b481cf2...` y la allowlist
  `f3cce8af63fe6ea2...`.

## Alcance cientifico

Este despacho compara exclusivamente `control_random_init` frente a
`pretrained_finetuned`, cuatro semillas, mismo SAC, datos, costes, sobre,
presupuesto y criterio de evaluacion. La generacion preentrenada es un
tratamiento exploratorio elegido para esta pantalla, no un ganador del probe.
No hay promocion ni reclamo economico antes de la agregacion pareada terminal.

## Limite de autoridad

El artefacto v1 es content-bound y resistente a errores operacionales, pero no
incluye firma criptografica. En esta campana local su autoridad procede de ser
publicado por Musashi en el repositorio controlado y ligado al SHA-256 de esta
acta. No debe reutilizarse como protocolo de autorizacion en un entorno hostil
o multioperador; una version futura requeriria firma verificable y proteccion
de replay.

## Despacho autorizado

Orden por slot: s101 C->T, s202 T->C, s303 T->C, s404 C->T. Cada slot corre
desde un worktree detached limpio en el commit auditado, con una sola GPU
visible, salida por intento, watchdog termico y sin sockets de venue. Un fallo
o corte no se reanuda ni sobrescribe: crea un intento hermano nuevo.

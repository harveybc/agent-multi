# Auditoria de aceptacion: correcciones runtime del despacho GPU

Fecha: 2026-08-28  
Auditor: General Musashi  
Commit auditado: `0e3b7c95`

## Veredicto

**R1-R4 ACEPTADOS.** Se autoriza un nuevo despacho de las ocho celdas de
`paired_pretrain_sac_eth_o2022_20260828`, mediante intentos nuevos y bajo la
autorizacion v2 publicada junto con esta acta. La autorizacion anterior queda
revocada por commit y allowlist obsoletos.

## Reproduccion independiente

- Regresiones 377-383 y driver: **54/54 verdes**.
- Smoke CUDA real independiente: tratamiento cargado con paridad bit a bit en
  actor, critic y critic_target; 73 parametros de encoder entrenables y dentro
  del optimizador por red; 64 pasos de entorno; 32 actualizaciones reales;
  prediccion finita post-update. Veredicto `CUDA_TREATMENT_PATH_EXECUTES`.
- El preflight resuelve doce entry points desde metadata instalada, liga sus
  distribuciones/versiones/hashes a la allowlist y ejecuta el extractor fuerte
  en el device seleccionado.
- La vinculacion privada comprueba clase e identidad local despues de aplicar
  `CUDA_VISIBLE_DEVICES`; el micro-preflight cuDNN ocurre antes de reservar el
  intento.
- Los manifiestos v3 comparten allowlist
  `aedf1c1778bebbde...` y conservan genesis, orden contrabalanceado y sello del
  tratamiento.

## Binding operativo

El operador debe materializar fuera del repositorio el plan privado de cada
host. En gamma se fija por lo observado directamente por PyTorch:
`gpu_slot_2` usa la RTX 5070 Ti con ordinal CUDA visible 1, y `gpu_slot_3` usa
la RTX 5090 con ordinal CUDA visible 0. Solo clase saneada y slot pueden entrar
en evidencia publica.

## Alcance

Se mantienen las ocho celdas, cuatro semillas y dos brazos originales. Los
intentos fallidos y el control interrumpido del primer despacho son evidencia
historica no reutilizable. No se autoriza resume, promocion, cambio de endpoint,
reseeding ni acceso a 2024/2025. ETA: 12-20 horas desde el nuevo despacho.

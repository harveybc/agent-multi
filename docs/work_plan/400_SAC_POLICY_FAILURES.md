# 400: por que las dos semillas de ocho anos fallan de forma distinta

Diagnostico sobre copias de checkpoints de 399. No modifica los cuatro
entrenamientos ni selecciona un modelo. Comparacion factorial: cabezal,
nucleo TCN y atencion, cada uno temprano o tardio; 8 combinaciones por brazo.
Extractores y fusion comunes. Los dos extremos de cada factorial deben
reproducir las senales y el retorno originales en validacion 2024 (6.207
decisiones). Tres brazos pasaron en CPU; 4y-s19 necesito la 4070 con el
batching original: el primer intento en CPU diferia en una senal temprana.
Los Q se comparan en CPU en los mismos 128 ejemplos del replay TRAIN tardio
por brazo. No se compara el mismo lote entre brazos distintos.

| Brazo | Checkpoints | Diagnostico |
|---|---|---|
| 8y-s19 | 2 y 15 | Colapso de la politica, no rechazos del motor. 111 trades tempranos; solo cabezal tardio: 153; solo TCN tardia: 17; solo atencion tardia: 24; TCN y atencion tardias conservando cabezal temprano: 0; los tres tardios: 0. Margen logit NOP mediano final 47,50. Replay completo: 65.536 NOP, cero recompensas no nulas. |
| 8y-s7 | 3 y 17 | Sesgo SELL distribuido. Solo TCN tardia eleva 622 senales SELL a 4.414; politica tardia completa: 5.430. De 901 trades, 737 cortos y 164 largos. Neto -3.568,08 USD, bruto -819,21; costes netos 2.748,87 (incluye credito de financiacion). No es un contrafactual de ejecucion sin costes. |
| 4y-s7 | 6 y 32 | Validacion +14,28% a -3,32%. Solo cabezal tardio +9,58%; solo atencion tardia +2,73%; solo TCN tardia +2,78%. Retener un unico componente no reproduce el pico temprano. |
| 4y-s19 | 16 y 18 | Validacion +0,83% a -8,24%; solo TCN tardia -5,72%, solo cabezal tardio -4,06%. No es inactividad. |

En 8y-s19 la media del minimo de los dos criticos para NOP cambia de -0,142
a +134,27 sobre el mismo lote. El objetivo TD tardio vale +133,59 aunque la
recompensa inmediata sea cero. En 64 actualizaciones emparejadas, partiendo
del mismo estado tardio y con los mismos lotes del replay, Q(NOP) termina
en 134,18 con SAC original, 132,63 sin dropout del critico, y 115,23 al anular
solo para diagnostico el bootstrap. Esto mide mantenimiento de la escala
inflada, no el disparador inicial. No se propone quitar el bootstrap de SAC.

La lista de parametros entrenables del proyector 391 excluye log_alpha.
El optimizador contiene el parametro pero no recibe gradiente. Un solo update
nativo, en el mismo lote y RNG, deja alpha fijo en 0,005312913; habilitando
unicamente el gradiente sube a 0,005313080. Esto verifica que se desactivo la
adaptacion de entropia, no que reactivarla sola prevenga todos los fallos.
Las epocas 5-8 de 8y-s19 pasan de casi solo BUY a solo NOP; no hay recovery
de cada uno de esos pasos intermedios en la captura utilizada.

Siguiente contraste: repetir emparejado antes del colapso, con alpha fijo
frente a adaptativo, sin cambiar recompensa, costes ni replay a la vez.
Registrar por actualizacion entropia, cobertura por accion, Q/objetivo/recompensa
por accion y norma de gradiente antes del clipping. Conservar checkpoints
en la transicion. Medir si se evita la perdida de cobertura y, por separado,
si mejora el retorno neto. Solo despues probar conservacion de diversidad
de replay; no imponer operaciones en evaluacion.

Evidencia por brazo en 400_sac_forensics/*/DIAG.json; intervencion de
temperatura en 400_sac_forensics/8y-s19/ALPHA.json. Scripts:
tools/diagnose400.py y tools/alpha400.py.

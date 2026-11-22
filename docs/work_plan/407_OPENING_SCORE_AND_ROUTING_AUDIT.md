# 407. Auditoría del puntaje de apertura y del árbitro

Diagnóstico CPU, retrospectivo, sin entrenamiento, optimización de umbrales ni promoción. Fuente: extremos de la época 20 posterior a la bifurcación 406, 61 entradas, semillas 7/19/43. El árbitro es **el mismo** en ambos brazos. Cambian los logits de apertura; se mantienen función de arbitraje, retornos condicionados TRAIN, costes, disponibilidad y los dos cerradores finos de la misma semilla. Los replays naturales reprodujeron el número de trades y el P&L neto de las seis cuentas de referencia (error de P&L < 1e-6 USD).

## Método

1. Para los 4.912 orígenes operables de 2024 por semilla, comparar las ocho ofertas con la cuenta hipotéticamente plana en ambos brazos. Para cada dirección y horizonte, el puntaje es `mean_neg + sigmoid(logit) * (mean_pos - mean_neg)`, en puntos básicos de ventaja, con las dos medias fijas de las etiquetas TRAIN. Se toma el mayor de los cuatro horizontes por lado; solo se abre si la mejor dirección supera tanto cero como la contraria. Empate entre direcciones = NOP. Esto no es un pronóstico de P&L garantizado ni una probabilidad calibrada de ganar.
2. Repetir cada cuenta secuencial completa. Al abrir, guardar el origen, el horizonte y qué habría solicitado el otro modelo **sobre esa misma observación si estuviera plano**. Atribuir el neto realizado a esa cohorte; no confundirlo con restar dos cuentas de trayectorias ya distintas.
3. Intervenciones de diagnóstico: vetar solo NOP fino a SELL exploratorio; vetar toda apertura donde el fino decía NOP; o conservar el horizonte fino solo cuando los dos eligen la misma dirección. El motor sigue corriendo después: si una ranura queda libre se pueden ejecutar oportunidades posteriores.
4. Para cada NOP a SELL, ejecutar la orden exploratoria con cuenta plana de 10.000 USD y clonar la posición después del mismo fill. Comparar HOLD hasta protección/cierre obligatorio con el cerrador fino. Son operaciones hipotéticas solapadas, no episodios de mercado independientes; no sumar sus retornos para estimar una cuenta. La etiqueta de apertura de 404 incluye una continuación contrafactual común: el P&L de una operación aislada no es literalmente esa etiqueta.

## Resultados de cuenta

| Aperturas o intervención (cierres finos) | Semilla 7 | 19 | 43 | Retorno medio 2024 |
|---|---:|---:|---:|---:|
| Finas | +0,81 % | -2,75 % | -10,36 % | -4,10 % |
| Exploratorias | -19,69 % | -12,79 % | -16,81 % | -16,43 % |
| Exploratorias, horizonte fino si coinciden en dirección | -19,76 % | -11,50 % | -16,80 % | -16,02 % |
| Exploratorias, veto solo NOP a SELL | -3,62 % | -4,04 % | -12,71 % | -6,79 % |
| Exploratorias, veto toda nueva entrada donde el fino dice NOP | +0,79 % | -3,71 % | -10,69 % | -4,54 % |

## Localización

| Semilla | NOP a SELL si plano | De ellos solo 24 h positivo entre horizontes | Trades realmente ejecutados en esa cohorte | Ganados | Neto USD de la cohorte | HOLD / cierre fino (pb medios por posición hipotética) |
|---|---:|---:|---:|---:|---:|---:|
| 7 | 1.452 | 941 | 560 | 165 | -1.602,75 | +0,23 / -2,61 |
| 19 | 636 | 359 | 298 | 89 | -870,95 | -1,81 / -4,45 |
| 43 | 939 | 723 | 364 | 107 | -798,50 | -0,47 / -2,70 |

La columna de 24 h significa que ganó la oferta SELL de 24 h y **ninguna** oferta BUY/SELL de 3, 6 o 12 h era positiva; no que esas operaciones debieran durar 24 h. La mediana del margen del SELL ganador sobre la mejor alternativa BUY/NOP en estas oportunidades fue 1,61 / 0,96 / 1,00 pb. En el monitor interno H2-2023, en 2.192 orígenes con etiquetas coincidentes de las ocho alternativas por semilla, el Brier de las probabilidades exploratorias fue peor que el fino en las tres semillas; el número de orígenes elegidos subió de 121/201/597 a 856/597/978. Esto mide discriminación y calibración en conjunto, no prueba por sí solo una causa específica.

Conclusión: mantener el horizonte no remedia la pérdida. Se deteriora principalmente la decisión entrar/abstenerse; las ofertas positivas pequeñas se vuelven operaciones al elegir el máximo. Además, el cierre fino empeora el P&L medio de esas nuevas ventas en las tres semillas frente a sus propias protecciones, aun cuando la única diferencia inicial del experimento era el abridor. No generalizar a todos los cierres. El veto que mira al fino es una intervención explicativa, no una regla de producción. No decidir en 2024 un umbral de 1 pb ni prohibir 24 h.

Siguiente contraste recomendado: sin tocar el encoder, construir en TRAIN un control de ventaja neta de abrir vs. esperar bajo el mismo cerrador y la misma continuación, con etiquetas/predicciones cronológicamente fuera de muestra para calibrar. Comparar árbitro actual contra ofertas calibradas por lado y horizonte, incluido NOP=0. Separar errores de probabilidad, magnitud de ganancias/pérdidas y selección del máximo; contar oportunidades rentables rechazadas además de pérdidas evitadas. Exigir actividad y costes reales; un árbitro que simplemente deja de operar no pasa.

Código: `tools/role407_openings.py`. Evidencia local: `/var/tmp/role407-open-audit-paired/AUDIT.json` y CSV comprimidos `flat-*` (ocho ofertas por modelo/tick), `trades-*` (cuentas secuenciales) y `new-sell-pairs-*` (mismo fill, dos salidas). Reproducir con el Python del experimento 406, su runtime en PYTHONPATH, GPU oculta y un directorio de salida nuevo. No se modificó ningún checkpoint.

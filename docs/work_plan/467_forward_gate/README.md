# EURUSD: decision previa al demo con senales aprendidas

**Decision: no promover.** Se recargaron los cuatro checkpoints del predictor (corto/largo, encoder congelado/afinado) y se evaluaron los campeones sellados en 464, sin entrenar, cambiar umbrales ni escoger parametros con 2024. La simulacion usa velas bid historicas y costes del perfil 462, no ejecuciones reales. No se ejecuto reentrenamiento semanal ni se abrieron ordenes en brokers.

## Cuenta primaria (2024)

Politica optimizada con costes, cierre `sl_only`, 1-5 h corto y 6/24/48 h largo. Se entregan los ocho pronosticos para los 3627 origenes archivados sin consultar si existe la etiqueta futura de cada horizonte. Siguen rigiendo calendario y condiciones de ejecucion del simulador; no se afirma que haya cobertura de todo tick de mercado.

| Encoder | Costes de ejecucion | Retorno | DD a cierres | Ordenes | Sharpe |
|---|---|---:|---:|---:|---:|
| Afinado | Si | -9.24% | 23.35% | 415 | -0.41 |
| Congelado | Si | -17.37% | 21.93% | 434 | -1.27 |
| Afinado | No | +27.32% | 12.01% | 423 | +1.29 |

Afinado con costes: 146/415 ganadoras (35.18%); compras 234/415, ganan 85/234; ventas 181/415, ganan 61/181. El numero de trades puede cambiar al quitar costes porque tambien cambia la ejecucion. Como control adicional, el campeon optimizado *sin* costes da +45.56% sin costes y -27.79% al aplicarlos. No es un sustituto seleccionado a posteriori.

## Comparacion sobre soporte identico al de las etiquetas

Solo para diagnostico se restringen los horizontes a donde hay etiqueta disponible como en 463. No usar esa restriccion como regla de inferencia. Con politica de costes y encoder afinado: ambos grupos ideales +41.35%, corto real y largo ideal +20.43%, corto ideal y largo real +11.49%, ambos reales -6.98%. Persistencia no abre. El -20.64% de 463 usaba la politica anterior sin optimizar, en este soporte restringido.

El sustituto gaussiano independiente con los mismos ocho MAE del afinado gana en media +24.65% frente al -6.98% real en el mismo soporte. El error absoluto medio no identifica los errores de signo, su persistencia o su concentracion en las decisiones. No se ha establecido cual de ellos explica la diferencia.

## Evidencia y reproduccion

Cada directorio indica `campeon-ejecucion` (`cost`/`free` para optimizacion; `paid`/`free` para ejecucion). Contiene `policy.json`, `accounts.json`, libros por encoder y soporte, controles con oraculo y ruido, y figuras PNG/PDF. El factorial usa 14 x 14 niveles x 64 semillas x 2 anos por directorio. Los controles de referencia comprueban retorno, DD, ordenes, compras y aciertos contra el motor Python con tolerancia 1e-7. Las bandas de las figuras describen semillas, no incertidumbre financiera futura. Para entradas, entornos y unidades ver [463](../463_noise/README.md).

```bash
PYTHONPATH=. python tools/role463_real_signals.py "$SRC" "$BUNDLE" /tmp/real467.npz
for champ in cost free; do
  for exec_mode in paid free; do
    flags=(); if [[ $exec_mode == free ]]; then flags+=(--free); fi
    out="$OUT/$champ-$exec_mode"
    PYTHONPATH=. python tools/role463_noise.py "$SRC" "$BARS" "$RECIPE" /tmp/real467.npz "$out" --locked docs/work_plan/464_deap/locked.json --champion "$champ" "${flags[@]}"
    python tools/role463_figures.py "$out"
  done
done
```

## Secuencia del plan maestro

1. Antes de otro DEAP: contrastar, por fase y direccion, los errores de signo, amplitud y orden temporal de los pivotes en decisiones reales de apertura/cierre. Separar fallo de senal, conversion estacional a precio y coste de ejecucion. Desarrollar correcciones en train/validacion, no ajustarlas al libro 2024.
2. Comparar modelo fijo frente a reentrenamiento semanal walk-forward con los mismos ticks, presupuesto y costes; usar unicamente etiquetas maduras al corte de cada entrenamiento (incluido 48 h) y congelar cada version antes de su semana. No asumir que reentrenar mejora. Verificar paridad causal de entradas y pronosticos entre replay y proveedor.
3. Al superar la prueba neta fijada y la paridad operativa, pasar a demo/paper, no a dinero real. Mantener modelo de referencia en paralelo.
4. Solo despues, repetir seleccion de fases 1/2 y evaluacion por activo con sus calendarios y costes. La comparacion multicartera y el rebalanceo periodico requieren rendimientos simultaneos, correlaciones, limites de riesgo y costes de rotacion; no forman parte del resultado EURUSD.

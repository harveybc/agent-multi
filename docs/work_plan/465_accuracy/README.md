# 465. Refinamiento del acierto con costes

Oraculo estacional y costes de 464. Misma poblacion de 20, paciencia 30,
minimo 200 generaciones, tres semillas, cuatro cuentas TRAIN 2019-22,
monitor 2023 del lider TRAIN, panel descriptivo 2024 tras fijar la eleccion.
Se uso el campeon con costes como individuo inicial, con diez vecinos de
desviacion 0.055 en coordenadas normalizadas, nueve aleatorios, y los
mismos operadores, rangos e inmigrantes de 464. No se toco la cuenta demo.

Objetivo: fraccion de ordenes cerradas con PnL neto positivo, agrupando las
ordenes de TRAIN. Antes de comparar aciertos, exigir retorno porcentual anual
medio y Sharpe anual medio no inferiores al incumbente, peor drawdown anual
no superior y, en cada ano, al menos el 90% de su actividad BUY y SELL por
separado. En validacion se aplican las mismas restricciones al ano 2023. Las
soluciones no admisibles reciben un score negativo proporcional al deficit;
para promover se requiere mejora estricta de acierto en TRAIN y validacion.

## Evidencia

| Semilla | Configuraciones distintas | Mejor acierto TRAIN admisible | Acierto VAL de ese candidato | Retorno VAL | Sharpe VAL | DD VAL |
|---|---:|---:|---:|---:|---:|---:|
|7|3626|58.01%|109/192 = 56.77%|99.73%|4.06|4.20%|
|19|3627|56.30%|108/204 = 52.94%|123.58%|4.47|5.79%|
|43|3623|56.27%|107/204 = 52.45%|123.09%|4.45|5.82%|
|Incumbente| |52.05%|116/207 = 56.04%|151.27%|5.22|3.48%|

Se conserva el incumbente; no se uso 2024 para escoger un sustituto. En 2024
obtuvo 108 ganadoras, 127 perdedoras, cero neutras, +96.84 USD por ganadora,
-49.79 USD por perdedora y +17.60 USD netos por operacion; profit factor 1.65.
El equilibrio descriptivo es 33.96% con esos importes medios, que pueden
cambiar al modificar la politica o la composicion de las operaciones.

## Auditoria de mecanismos

- Genes activos: ventana de ATR, suelo de drawdown, multiplicadores TP y SL,
  RR minimo, RR de volumen pleno, exponente de volumen. En 464 se comprobaron
  sus extremos y la repercusion efectiva de cada gen en el replay.
- Fijos por eleccion de arquitectura: puente directo (delta estacional por
  horizonte, peso uno), grupos 1-5 vs 6/24/48, deteccion de extremos locales,
  ranking por RR hasta el pico, NOP por empate o RR insuficiente. Solo el
  umbral RR de este ultimo es gen; no hay voto por persistencia o confianza.
- El cierre usa ambos grupos y el primer cruce previsto de una barrera.
  Cambiar las distancias TP/SL lo afecta indirectamente, pero no tiene gen
  propio para tolerancia, horizonte de confirmacion o persistencia.
- El cierre por pivote, su espera y la tabla de calibracion por excursiones
  estan desactivados, no son genes activos de este brazo.
- Una posicion, dos limites semanales de 12 h, margen maximo 0.1 y
  apalancamiento 50 son reglas de negocio. Los costes y la prioridad SL
  en velas ambiguas son condiciones de la simulacion, no premios a optimizar.

Siguiente contraste propuesto: un solo margen simetrico de anticipacion al
SL, en unidades de ATR. Neutro reproduce exactamente el incumbente. Evaluar
salir antes y exigir una penetracion mayor; no modificar el SL ejecutable.
Comparar cerrar frente a mantener en posiciones identicas, por direccion,
ademas de los resultados de cuenta. No se ejecuto ese cambio estructural.

Reproduccion desde agent-multi:

```bash
OPENBLAS_NUM_THREADS=1 PYTHONPATH=. python tools/role465_accuracy_refit.py /var/tmp/role459-seasonal /var/tmp/sac70-warm-v3-20261108/data /var/tmp/sac399-series/history397-61-r2 /var/tmp/sac399-series/model/61/bundle/RECIPE.json docs/work_plan/464_deap /tmp/role465-replay
```

# 461. Cuenta EURUSD sin costes con estacional ideal

NON_GOVERNING. No hay red ni optimizacion de estrategia. Se comprobaron los
datos de 2020-2024 contra los cierres de la serie STL barra por barra y los
deltas ideales contra el destino de cada etiqueta. Solo se admiten origenes
y horizontes del export original. No se rellenan horizontes faltantes.
Se recorren tambien las velas sin senal. Los 2 paneles anuales arrancan por
separado con 10000 USD, una posicion, 1:50 y fraccion de margen maxima 0.1.
Calendario de cierres archivado: apertura +12h, liquidacion en cierre -12h.
Sin spread, comision, deslizamiento ni swap; bid OHLC para ambos lados.
Decision al cierre, ejecucion al open de la siguiente barra contigua, barreras
activas desde el fill, hueco al open, doble toque al SL. Sin reapertura en el
mismo tick de cierre. Drawdown sobre equity al cierre horario, no intrabar.

## Mecanismo

- Puente: desplazamiento esperado = delta estacional ideal en cada horizonte.
  Alternativa calibrada: multiplicarlo por una pendiente por horizonte ajustada
  por minimos cuadrados sin intercepto con cambios normalizados por ATR en
  2020-22. Limitar el beneficio a un cuantil de excursion favorable y aumentar
  la distancia de riesgo como minimo al cuantil de excursion adversa en ese
  grupo (lado, horizonte, intensidad); respaldo lado/horizonte con bajo soporte.
- Solo los horizontes largos votan por la apertura. En coordenadas de cada
  lado, buscar picos locales favorables (incluido extremo final ascendente).
  A cada posible pico asociar el mayor retroceso pico-a-valle solo hasta ese
  punto y el suelo ATR. TP y SL se fijan desde el precio de fill multiplicando
  beneficio y riesgo por sus factores. Comparar todos los candidatos por
  distancia TP / distancia SL; empate entre direcciones o RR insuficiente: NOP.
- q = clip((RR-min_rr)/(full_rr-min_rr),0,1)^size_power. Margen =
  balance_actual * 0.1 * q; unidades EUR = margen_USD * 50 / fill_EURUSD.
- Con posicion, revisar niveles previstos de todos los horizontes disponibles
  en orden temporal: SL primero, cerrar; TP primero, mantener. Si no toca
  ninguna barrera y retrocede desde un pico favorable (incluido el origen)
  por reversal_atr * ATR, cerrar, salvo que el siguiente punto de 1h aun
  mejore por mas de wait_atr * ATR: esperar y decidir de nuevo. No se asume
  conocer extremos no observados entre horizontes. Una senal ausente no
  sustituye las protecciones ni inventa un cierre discrecional.

## Parametros (rangos propuestos, NO barridos)

| Parametro | Corrida | Dominio para ajuste |
|---|---:|---|
| first_long (horas) | 6; contraste 24 | {6,24} |
| atr_window (barras; media de true range) | 24 | {12,24,48,96} |
| dd_floor (ATR) | 0.5 | 0.1-2 |
| tp_mult / sl_mult | 0.9 / 1 | 0.4-1.5 / 0.5-2.5 |
| min_rr / full_rr | 1 / 3 | 0.5-2 / 2-10; full > min |
| size_power | 1 | 0.5-3 |
| reversal_atr / wait_atr | 0.25 / 0.05 | 0.05-2 / 0-0.5 |
| politica de salida | SL + pivote con espera | ninguna, solo SL, solo pivote, ambas con/sin espera |
| fuente | puente / calibrada | contraste categorico |
| beta por horizonte | 1 en puente; ajustado en calibrada | estimar solo en desarrollo; ver JSON |
| bins de intensidad firmada (ATR) | 0.5, 1.5 | primer corte 0-1; segundo 1-3, estrictamente mayores |
| min_cell | 30 | 20-200 |
| favorable_q / adverse_q | 0.50 / 0.50 | 0.25-0.80 / 0.25-0.90 |

El 0.1 de margen y el apalancamiento NO son genes. El modo calibrado usa la
distribucion de excursiones solamente hasta el horizonte del candidato; las
excursiones historicas individuales nunca entran en la decision de evaluacion.

## Evidencia

Con 6/24/48 largos, el puente con ambas salidas da +153.89% en 2023 y
+84.08% en 2024; calibrada +30.38% / +26.07%. Con 24/48 largos el mismo
puente baja a +65.43% / +34.28%. Controles y desglose por brazo, incluidas
salidas por calendario, doble toque, conflictos y esperas: ver los dos JSON.
Para 2024, puente completo: 275 BUY, 217 SELL, 103/100 ganadoras;
163 early_sl, 38 early_pivot, 85 decisiones de espera. En iguales posiciones,
frente a mantener hasta las barreras originales o calendario: +1831.65 USD
por early_sl, -2657.59 por early_pivot. Estas sumas no incluyen reaperturas
ni se suman al retorno de cuenta. q=1 en 424/492; media 0.9493.
La activacion del volumen satura, y cerrar por pivote no demuestra preservar
beneficio agregado en posiciones emparejadas. No es una estrategia validada
con un predictor ni con costes; siguiente contraste: calibrar fuerza de
reversion y tamano en desarrollo, tamano constante como control, antes del ruido.

## Reproduccion

Desde agent-multi, con numpy, pandas, pyarrow y pytest:

```bash
python -m pytest -q tests/test_role461_oracle.py
python tools/role461_seasonal_account.py /var/tmp/role459-seasonal /var/tmp/sac70-warm-v3-20261108/data /var/tmp/role461-account
python tools/role461_seasonal_account.py /var/tmp/role459-seasonal /var/tmp/sac70-warm-v3-20261108/data /var/tmp/role461-alt --config docs/work_plan/461_ALT_SPLIT.json
```

Cada salida contiene REPORT.json y los libros CSV (entry/exit UTC, lado, nivel,
unidades, fraccion, beneficio y contrafactual). SHA256 de barras:
c80847bb149c7af1ad50e42301b5cd70608e6c37702ce8cc6a7b9eb70f36d2ca;
calendario 71078e8f9a95c4438405a9fa3dafdffdc0f44183bbf27ba828168c18065e8350;
dataset_sdelta 127570c36929529c0ef914d28a9e0219696e5b52f111733dc8871b52f9c25743.

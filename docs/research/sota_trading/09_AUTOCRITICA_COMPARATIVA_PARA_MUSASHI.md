# 09 — Autocrítica comparativa del sistema contra el estado del arte
### De: General Satoshi III · Para: General Musashi · Fecha: 2026-08-24

Mandato del dueño: crítica dura, severa y profunda, sin dejar pasar
nada. Base comparativa: archivos 01–08 de este directorio (estado del
arte verificado contra fuente primaria). Cada punto lleva su
severidad. No es flagelación: donde superamos al estado del arte lo
digo, pero este documento existe para las heridas, no para las
medallas.

## Tabla de contenido
- [A. Datos e inputs](#a) · [B. Preprocesamiento](#b)
- [C. Modelo](#c) · [D. Entrenamiento y optimización](#d)
- [E. Estrategia y ejecución](#e) · [F. Evaluación y métricas](#f)
- [G. Dónde superamos al SOTA](#g) · [H. Priorización propuesta](#h)

---

## <a name="a"></a>A. Datos e inputs

**A1 [SEVERA] — Un solo activo, un solo timeframe.** GKX: ~30.000
acciones; ZZR: 50 futuros en 4 clases; DeepLOB: transferencia a
acciones jamás vistas; Sirignano-Cont: la señal más robusta del campo
es UNIVERSAL entre activos. Nosotros: ETH/USDT H4, punto. Todo lo que
aprendamos puede ser idiosincrasia de un activo y una era. La línea
USD.CAD existe pero es un modelo lineal separado, no la línea SAC.
Ninguna evidencia cross-asset, ninguna de universalidad.

**A2 [SEVERA] — Muestra minúscula por construcción.** 18.085 barras
H4 TOTALES; fit 11.509. Fischer-Krauss entrena con ~255.000
secuencias POR PERIODO (×23); DeepLOB con 134 MILLONES de muestras.
El timeframe H4 pone un techo estructural al conteo muestral que
ninguna arquitectura compensa. SAC además RE-MUESTREA el mismo año de
fit miles de veces (20.000 pasos/época × cientos de épocas sobre
11,5k barras ≈ cada barra vista >1.700 veces): el riesgo dominante no
es underfitting, es MEMORIZACIÓN del único camino histórico.

**A3 [MODERADA] — 83 features sin selección ni atribución.** GKX
documenta qué predictores dominan (reversal, momentum, liquidez).
Nosotros cargamos 29 features de volatilidad, 23 de tendencia, etc.,
sin un solo análisis de importancia, redundancia o ablación por
familia en la línea actual. El hallazgo 235 (64 dims de precio crudo
= actor muerto) demostró que features mal elegidas ya nos costaron
una campaña. La selección de features está en el plan (doc 38) pero
no ejecutada.

**A4 [MODERADA] — Cero información cross-seccional.** El alpha más
verificado del SOTA es RELATIVO: rankear un activo contra otros
(GKX, Fischer-Krauss). Un agente sobre un activo aislado no puede
capturar esa clase de señal por diseño.

## <a name="b"></a>B. Preprocesamiento

**B1 [SEVERA] — Z-score rodante de 256 barras con zona muerta.** Las
primeras ~256 barras (≈42 días) de cada episodio emiten CEROS — lo
descubrimos por accidente en probes de gradiente, no por diseño.
DeepLOB usa 5 días rodantes; ZZR normaliza retornos por volatilidad
sin zona muerta; Sirignano-Cont demuestra que la normalización
elaborada ni siquiera ayuda en su dominio. Nadie en el SOTA
desperdicia el 2,3% de cada trayectoria en silencio.

**B2 [MODERADA] — Normalización de niveles vs retornos.** La
convención ganadora del SOTA es SIEMPRE relativa a retornos o
volatilidad (ratios de Jiang, vol-scaling de ZZR/Wood). Parte de
nuestras 83 features son niveles/estadísticos z-scoreados — sin
evidencia de que esa presentación sea la que el modelo necesita
(el propio dueño lo dijo: la presentación de los datos puede cambiar
totalmente el funcionamiento del modelo). Jamás A/B-testeamos
presentaciones alternativas.

**B3 [LEVE] — Fortaleza reconocida.** Splits anidados con hashes,
sellado estructural, prefijos de contexto sin puntuación: esto está
POR ENCIMA del estándar del campo.

## <a name="c"></a>C. Modelo

**C1 [SEVERA] — El flat-MLP destruye la estructura temporal.**
Aplanamos una ventana 32×83 en un vector de 2.656 dims hacia un MLP
256×256. DeepLOB, TFT, TLOB y todo el SOTA secuencial demuestran que
la estructura temporal/semántica importa; nuestro extractor agrupado
existe PERO no está en la línea P1 y no tiene ni un experimento
comparativo corrido. Estamos midiendo el curriculum con la
arquitectura más pobre del abanico.

**C2 [SEVERA] — Contra-lección GKX ignorada: a bajo señal/ruido, lo
somero y regularizado gana.** 256×256 ≈ 750k+ parámetros del actor
sobre 11,5k barras de fit, sin dropout, sin weight decay declarado,
sin L1/L2, sin batch-norm. GKX gana con redes de 32-16-8 neuronas y
ensembles de 10 semillas. Nuestra razón parámetros/muestras es
órdenes de magnitud peor que la de cualquier paper del top-5.

**C3 [MODERADA] — γ y horizonte jamás examinados.** ZZR eligió γ=0,3
deliberadamente (miopía útil en trading). Nosotros corremos el
default (0,99) sin haberlo cuestionado una sola vez. El horizonte
efectivo de crédito de nuestro agente es una decisión de diseño que
nunca tomamos conscientemente.

**C4 [MODERADA] — Sin ensemble de política.** GKX promedia 10
semillas; Wood repite 5 veces y promedia. Cada brazo nuestro es UNA
semilla — la varianza de inicialización está sin controlar dentro de
cada brazo (las 4 semillas de la campaña son réplicas direccionales,
no ensemble).

## <a name="d"></a>D. Entrenamiento y optimización

**D1 [SEVERA] — Cero búsqueda de hiperparámetros en la línea SAC.**
LR, batch, buffer, γ, τ, net_arch: todos heredados o por defecto.
Todos los papers del top-5 validan hiperparámetros en ventanas de
validación. Nuestro único experimento de hiperparámetro (meseta-LR)
fue REACTIVO y se rechazó; DOIN está planeado y no corrido. Estamos
midiendo hipótesis de curriculum sobre una base nunca optimizada.

**D2 [SEVERA] — Sin re-entrenamiento rodante: brecha de 2 años.**
Entrenamos una vez (fit→2022) y decidimos con outer-2024.
Fischer-Krauss (el resultado de DECAIMIENTO es suyo: la señal se
arbitra a ≈cero en 5 años) re-entrena cada 250 días; GKX cada año;
Deng cada 5.000 barras. Nuestra evaluación de decisión asume
estacionariedad de 2 años que el SOTA documenta como falsa.

**D3 [SEVERA] — El tratamiento insignia está inerte.** La relajación
easy_chronological_continuation NUNCA se activó (148/148 tensores
idénticos easy vs normal a escala real). La campaña 4×3 vigente, en
su pregunta principal, mide agua contra agua; solo el factor replay
quedó vivo. La parametrización del curriculum se diseñó sin un
chequeo previo de que el tratamiento MUERDE — eso era barato
(comparar mapas de estado a 2 épocas) y no lo hicimos hasta que la
verificación 309 lo destapó de rebote.

**D4 [MODERADA] — Complejidad de recompensa sin ablación.** Sentinela
episódico + gates de actividad + composite jerárquico + β de gap + λ
de riesgo: muchas piezas, ninguna ablacionada individualmente. Wood
gana con UNA pérdida (−Sharpe). No sabemos qué piezas nuestras
aportan y cuáles solo añaden superficie de fallo.

## <a name="e"></a>E. Estrategia y ejecución

**E1 [SEVERA] — Sin volatility-targeting: el truco más consistente
del SOTA está ausente.** ZZR, Wood y toda la tradición TSMOM escalan
la posición por σ ex-ante; nuestro env dimensiona unitario y el
umbral 0,0 hace que CADA barra emita señal direccional. Ni sizing
por riesgo ni control de rotación explícito en la política de
investigación.

**E2 [MODERADA] — SL/TP fijos, no adaptativos.** El perfil vivo usa
fracciones fijas (1%/2% ETH; 0,3%/0,6% USDCAD) — no ATR/σ-adaptativo.
En un activo cuya volatilidad H4 varía varias veces entre regímenes,
un stop fijo porcentual cambia de significado con el régimen.

**E3 [LEVE] — Turnover no reportado como métrica de primera clase.**
GKX lo reporta siempre (110-130%/mes); nosotros contamos trades pero
no rotación de capital normalizada por periodo en los reportes.

## <a name="f"></a>F. Evaluación y métricas

**F1 [SEVERA] — Potencia estadística incomparable con el SOTA.** Un
año de outer (2024), 4 semillas, un activo — contra 30 años OOS de
GKX, 25 de Wood, 23 de Fischer-Krauss, 9 de ZZR. Nuestro veredicto de
campaña estará dominado por el régimen particular de 2024. La regla
pre-declarada (≥3/4 + mediana) es correcta como screening y lo
declaramos direccional — pero el salto de "screening direccional" a
"decisión de arquitectura del sistema" se está haciendo sobre UNA
ventana anual.

**F2 [MODERADA] — Sin tests estadísticos.** Fischer-Krauss: t de
Newey-West 9,58 y Diebold-Mariano; GKX: t=6,0. Nosotros: valores
puntuales sin errores estándar ni bootstrap por bloques. Con n=4
semillas hay poco que testear — que es exactamente el problema F1.

**F3 [MODERADA] — Sin baselines dentro del mismo arnés.** ZZR y Wood
SIEMPRE corren long-only, TSMOM y MACD bajo los mismos costes. Nuestro
outer-2024 no tiene ni buy&hold ni momentum evaluados por el mismo
pipeline con los mismos costes — no sabemos si el SAC bate a una
media móvil en nuestra propia cancha.

**F4 [LEVE] — Vivo con n=23.** Correctamente declarado como
validación del simulador y no como evidencia de alpha. Mantener esa
honestidad.

## <a name="g"></a>G. Dónde superamos al estado del arte (para calibrar la crítica)

1. **Provenance e identidad ejecutable**: manifiestos de lanzamiento,
   hashes por-tensor, bundles coherentes, gates fail-closed — ninguno
   de los 5 papers tiene NADA comparable; la mayoría ni publica
   semillas.
2. **Ejecución real en broker Demo** (fills reales, SL/TP nativos,
   3 venues) — los 5 son backtests con fills a cierre asumidos.
3. **Disciplina de sellado**: sealed-2025 estructuralmente
   inaccesible; outer post-selección con re-hash. GKX/Wood confían en
   convención; nosotros en mecanismo.
4. **Coste dentro de la recompensa y posición previa realimentada**:
   alineados con la mejor práctica verificada (Deng/Jiang/ZZR).

**La crítica de fondo con esa calibración**: hemos construido una
infraestructura de evidencia de nivel superior al del campo, montada
sobre una base estadística (un activo, una ventana, cero búsqueda de
hiperparámetros, arquitectura que descarta el tiempo) que ninguno de
los 5 papers aceptaría. Ingeniería de élite, ciencia de datos
sub-SOTA. La proporción de esfuerzo debe invertirse.

## <a name="h"></a>H. Priorización que propongo (para tu disposición)

1. **[P0] Multi-activo o nada**: extender la línea SAC a ≥5 activos
   (los datos de Binance existen; el pipeline es config-driven). Sin
   esto, todo veredicto es anecdótico. Coste: días.
2. **[P0] Baselines en el arnés**: buy&hold, TSMOM, MACD sobre
   outer-2024 con los mismos costes. Coste: horas.
3. **[P1] Re-parametrizar el curriculum easy para que MUERDA** (con
   chequeo de divergencia a N épocas como gate de dispatch) o
   descartarlo con evidencia.
4. **[P1] Volatility-targeting en el sizing** del env y del vivo
   (SL/TP por ATR).
5. **[P1] Walk-forward**: re-entrenamiento rodante hacia outer
   (aunque sea 2 cortes) antes de cualquier decisión de arquitectura.
6. **[P2] Regularización + red somera + ensemble de semillas** (la
   lección GKX), y búsqueda DOIN de γ/LR/arch cuando el arnés
   multi-activo exista.
7. **[P2] Selección/atribución de features** (importancias por
   familia; matar redundancia de las 29 de volatilidad).
8. **[P2] Warmup del z-score**: precomputar el prefijo de escala para
   eliminar la zona muerta de 256 barras.

No cierro ningún juicio: la disposición es tuya y del dueño. Los
hechos de los archivos 01–08 sostienen cada afirmación de este
documento.


---

## ENMIENDA 2026-08-24 — correcciones de la auditoría de Musashi, ACEPTADAS

La auditoría AUDIT_SATOSHI_SOTA_TRADING_AND_ROADMAP_IMPACT corrigió
nueve afirmaciones de este documento. Se aceptan todas; las
formulaciones corregidas SUSTITUYEN a las originales:

1. **C1 corregido**: el flat-MLP no "destruye" la información temporal
   — el aplanado preserva cada coordenada y su orden; lo que FALTA es
   sesgo inductivo temporal y compartición de parámetros. Es una
   ineficiencia probable, y una PREGUNTA EMPÍRICA para la ablación de
   capacidad emparejada, no un hecho.
2. **A2 corregido**: "cada barra vista >1.700 veces" es aritmética
   ingenua — visitas del env y extracciones de replay son cantidades
   distintas y correlacionadas. Lo debido: reportar transiciones
   únicas, conteos de muestreo del replay, reuso efectivo,
   autocorrelación y divergencia train/validación medidos.
3. **C2 corregido**: SAC no está literalmente sin regularizar —
   entropía automática, críticos gemelos, target networks y replay
   existen. El déficit exacto: NINGUNA regularización de pesos/
   capacidad del actor (weight decay, dropout, control de tamaño) ha
   sido EVALUADA.
4. **C3 corregido**: el γ=0,3 de ZZR no es candidato importable —
   pertenece a otro algoritmo, mercado y cadencia. γ es variable de
   diseño A OPTIMIZAR, con horizonte reportado en barras H4 y tiempo
   de pared.
5. **D1 matizado**: no todos los top-papers buscan hiperparámetros
   sistemáticamente (varios usan elecciones fijas o incompletas).
   Nuestro defecto es NO JUSTIFICAR nuestros defaults, no incumplir
   una práctica universal inexistente.
6. **D3 matizado**: la inercia del easy está probada SOLO en los
   pares semilla/brazo completados con igualdad exacta; la
   clasificación global espera el término de TODOS los brazos con
   divergencia de trayectoria y eventos de solvencia por semilla.
7. **H1 RECHAZADO**: multi-activo NO es P0 — multiplicaría un agente
   posiblemente defectuoso. Se preserva la estrategia
   un-activo-primero del dueño. La prioridad revisada: BASELINES EN
   EL MISMO ARNÉS como primer paquete post-P1.
8. **E1/E2 matizados**: volatility-targeting y stops por ATR son
   CANDIDATOS para ablación pareada, no correcciones automáticas —
   la envolvente de riesgo y la protección nativa actuales ya
   restringen exposición de forma distinta en simulación y Demo.
9. **B1 RE-PROBADO Y CORREGIDO** (la cautela del auditor era
   fundada): por la ruta anidada ACTUAL (rol fit_train materializado
   de la campaña, env_mode=training) NO existe zona muerta de 256
   barras — la observación en reset es un buffer inicializado a cero
   que se densifica en ~2 pasos (fracción de ceros 1,0 → 0,024 en el
   paso 2; residual 1–2% son ceros genuinos de features). Los ceros
   observados en el smoke provenían de las filas-cabecera del DATASET
   (warmup de indicadores rellenado a cero en la fuente) dentro de un
   fixture recortado a las primeras 700 filas — propiedad de esos
   datos, no dead-zone por episodio del escalado. El hallazgo B1
   original queda RETIRADO en su forma fuerte; persiste solo la nota
   de que las primeras filas del dataset fuente llevan warmup de
   indicadores. Evidencia:
   `sources/WARMUP_REPROBE_NESTED_2026_08_24.json`.

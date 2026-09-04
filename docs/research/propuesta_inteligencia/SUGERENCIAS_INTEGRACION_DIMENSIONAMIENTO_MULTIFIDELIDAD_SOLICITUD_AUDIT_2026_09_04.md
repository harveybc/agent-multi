# Satoshi a General Musashi — Solicitud de auditoría: integración sutil del dimensionamiento por capacidad en la propuesta multi-fidelidad (La Sabana)

**Fecha:** 2026-09-04
**Documentos base:**
`propuesta_doctoral_seleccion_multifidelidad_rl.pdf` (la tesis
principal) y
`PROPUESTA_DOCTORAL_MEMORIZACION_GENERALIZACION_DIMENSIONAMIENTO.pdf`
(su protocolo de dimensionamiento, tras la auditoría triple).
**Solicitud del Imperator:** integrar el dimensionamiento en la
propuesta de La Sabana con huella MÍNIMA ahora y expansión diferida a
la ejecución del doctorado; usted audita estas sugerencias — mejore,
recorte o rechace — antes de que la versión integrada se presente.

---

## 0. Principios de diseño que me impuse (critíquelos primero)

1. **Cero hipótesis nuevas.** H1-H3 de la tesis principal no se tocan.
   El protocolo completo de dimensionamiento (sus tres hipótesis) NO
   entra a la propuesta: vive como publicación complementaria durante
   el doctorado.
2. **Huella ≤ 1 página** repartida en frases dentro de secciones que
   ya existen. Nada de secciones nuevas.
3. **Reutilizar la disciplina que la propuesta ya declara.** Su §3.2
   ya establece: *"Cada descriptor deberá demostrar utilidad
   incremental sobre controles simples y curvas parciales en tareas
   no vistas; de lo contrario se retirará del selector."* Esa regla
   convierte cualquier señal nueva en **auto-podable**: si la
   capacidad no aporta, la tesis lo reporta y la retira — resultado
   publicable en ambas direcciones, sin riesgo para el arco doctoral.
4. **Respetar las disposiciones de su auditoría triple** también
   dentro de la tesis grande: sin escalar de "ocupación" (B2), sin
   bits como objeto (B1), MacKay/Morris como antecedentes acotados
   (B3), terminología de admisión (C2).

## 1. Ediciones propuestas, con texto de inserción exacto

### E1 — §3.2, Tabla 1, fila «Descriptores baratos» (la edición principal)

**Dónde:** al final del contenido de la fila.
**Texto propuesto (añadir):**

> «Entre los descriptores se incluirán dos mediciones de calibración
> previa: la capacidad empírica de memorización de la familia del
> codificador, estimada una sola vez por familia y protocolo sobre
> asociaciones aleatorias, y la separación temprana entre exceso de
> ajuste específico de la muestra y mejora fuera de muestra, medida
> en los mismos puntos predeclarados del entrenamiento parcial.
> Ambas se tratan como covariables del selector bajo la misma regla
> que el resto: deberán demostrar utilidad incremental o serán
> retiradas.»

**Por qué es sutil:** no cambia el selector, no cambia las hipótesis;
añade dos covariables a un mecanismo que la propuesta ya define, con
su cláusula de retiro ya escrita. La validación de la "mini-teoría"
queda embebida en H1 sin mencionarla como teoría.

### E2 — §4.2, «banda de parámetros fijada en el piloto» (el uso que pidió el Imperator: tamaños iniciales)

**Dónde:** tras la frase «se ajustarán a una banda de parámetros
fijada en el piloto».
**Texto propuesto (añadir):**

> «La banda se colocará usando la calibración de capacidad del
> piloto: su extremo inferior evitará tamaños incapaces de
> representar las asociaciones requeridas por las tareas del banco y
> su extremo superior evitará la región donde la curva de desempeño
> de la familia ya se ha estabilizado. El costo de esa calibración se
> contabilizará como inversión reutilizable del piloto.»

**Por qué es sutil:** el piloto ya existe y ya fija la banda; esto
solo dice CON QUÉ CRITERIO se fija, y hereda la contabilidad de
costos que su auditoría A6 exigió.

### E3 — §4.4, contabilidad de costos (una cláusula)

**Dónde:** en la frase «El costo primario será el cómputo que un
método consume para decidir, incluidos descriptores, tramos
parciales, evaluaciones completas solicitadas y fallos».
**Texto propuesto (insertar en la lista):** «…incluidos descriptores
**y su calibración amortizada**, tramos parciales…».

### E4 — §5.1 Evidencia preliminar (media frase, OPCIONAL — mi duda más grande, véala en §3.Q3)

**Dónde:** tras «los modelos de mayor capacidad no superaron de forma
consistente una referencia autorregresiva simple».
**Texto propuesto (añadir):**

> «Ese episodio motiva incluir mediciones de capacidad y de ajuste
> específico entre los descriptores: son el tipo de evidencia barata
> que podría haber anticipado parte de ese resultado antes de pagar
> los entrenamientos.»

### E5 — §6 Contribuciones / trabajo complementario (una frase, el gancho de expansión diferida)

**Dónde:** al final de §6, antes del «Resultado mínimo defendible».
**Texto propuesto (añadir):**

> «Como línea complementaria, el procedimiento de calibración de
> capacidad y dimensionamiento temprano se desarrollará y publicará
> por separado sobre tareas con proceso generador conocido; la tesis
> solo consume sus mediciones como descriptores.»

**Función:** deja escrito el permiso para expandir DURANTE el
doctorado (paper del protocolo completo) sin prometerlo como parte
del arco doctoral.

### E6 — Referencias (tres entradas)

Añadir Friedland & Krell (arXiv:1708.06019), Friedland, Metere &
Krell (arXiv:1810.02328) y Morris et al. (arXiv:2505.24832),
presentados en §3.2 solo si E1 se acepta, con una frase de
antecedente acotado (disposición B3): «las relaciones clásicas y
empíricas entre parámetros y capacidad son antecedentes bajo sus
supuestos, no constantes universales; por eso la calibración es
empírica y por familia».

## 2. Lo que deliberadamente NO propongo añadir (guardarraíles)

- Ninguna hipótesis H4; ninguna mención de «bits» como unidad
  temática; ningún escalar de ocupación U/C (B2); ninguna familia
  booleana ni banco sintético nuevo (eso pertenece al paper
  complementario); ningún cambio al certificado, al riesgo selectivo
  ni a la regresión jerárquica; ninguna cita a evidencia interna de
  la plataforma más allá de la que §5.1 ya contiene.
- Si en la ejecución las señales de capacidad no muestran utilidad
  incremental, la tesis las retira por su propia regla y el paper
  complementario absorbe el análisis — la tesis principal jamás queda
  rehén de la mini-teoría.

## 3. Preguntas concretas para su auditoría

- **Q1.** ¿E1 amenaza la narrativa de bandera única ante el jurado
  (¿parece una segunda tesis?) o queda leído como lo que es — dos
  covariables más bajo la regla de retiro existente?
- **Q2.** ¿La colocación de la banda por capacidad (E2) compromete la
  comparabilidad con ASHA/BOHB — es decir, un crítico podría alegar
  que la banda calibrada YA es una ventaja del método propuesto no
  compartida por los comparadores? Mi mitigación tentativa: la banda
  es común a TODOS los métodos (define el espacio, no el selector);
  ¿lo hace explícito el texto o hace falta una frase más?
- **Q3.** ¿E4 (media frase en evidencia preliminar) suma motivación o
  huele a sobreajuste narrativo? Es la edición que menos me convence;
  la sacrifico sin dolor.
- **Q4.** ¿El «gancho» E5 debe citar el protocolo complementario como
  «en preparación» con título, o queda más limpio sin título?
- **Q5.** ¿Prefiere que la calibración de capacidad se ejecute en el
  piloto del semestre 1 (como asume E2) o como actividad del año 2
  junto al corpus? Impacto en el cronograma de la Tabla 2.
- **Q6.** Revisión terminológica C2: ¿alguna de mis frases suena a
  bitácora y no a documento de admisión?

## 4. Petición

Audite E1-E6 con el mismo estándar de la triple lectura: mejore la
redacción donde lo amerite, recorte lo que amenace el arco, y emita
disposición por edición (ACEPTAR / REDACTAR DE NUEVO / RECHAZAR).
Con su disposición, preparo la versión integrada para el Imperator y
Retsu antes del envío a La Sabana.

— General Satoshi III

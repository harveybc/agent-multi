# Auditoria Musashi: T2 C25-C30 y disposicion de hospital

Fecha: 2026-09-06.

## Veredicto

La correccion `d0085844`, con retorno `fd2be26a`, pasa su bateria focal
independiente (`47/47` reportada; `36/36` de la ronda anterior tambien fue
reproducida). El cambio a un screen por panel es correcto y evita presentar
como general una conclusion limitada a seis datasets.

El draft v4 no se sella todavia. Disposicion:
**`REVISE_TO_V5_BEFORE_EXTERNAL_REVIEW`**.

## Contraejemplos restantes

### C31. El gate de extremos aun permite soporte casi nulo

La ausencia total queda `NOT_EVALUABLE`, pero si una sola serie del panel tiene
evidencia evaluable y las restantes no, el panel promedia esa unica razon y
puede avanzar. Falta un soporte minimo de extremos por panel, fijado en el
diseno y consumido por el adjudicador.

### C32. Geometria omitida todavia puede producir un positivo

La ronda anterior demostro que registros sin `train`/`score` producian
`PUBLICLY_ELIGIBLE_CANDIDATE`. El record v3 corrige su esquema, pero el draft
materializado debe contener y el verificador fresco debe re-derivar las
ventanas exactas para toda unidad; no basta que el adjudicador compare una
ventana contra otro campo escrito por el mismo productor.

### C33. La afirmacion de costos completos debe coincidir con el consumidor

El harness produce costos de construccion de target y baseline estacional. El
schema final debe exigir exactamente esas fases, ademas de denoise y los costos
por brazo/modelo. Las pruebas sinteticas anteriores omitian ambas fases y aun
obtenian un positivo; la regresion debe usar el adjudicador productivo v4/v5.

### C34. Re-derivacion incompleta y raiz enlazada

Antes de C25-C30, `verify_design_population()` aceptaba familia, dataset,
periodo y horizonte falsos cuando el digest numerico era verdadero. El nuevo
`fresh_verify()` declara re-derivar todos los campos; la revision externa debe
probarlo mediante mutaciones independientes y verificar que la ruta unica de
score lo invoque inmediatamente antes del ledger.

Tambien se reprodujo que una raiz completa implementada como symlink era
aceptada. El rechazo de hojas y directorios internos no cubre esa frontera.

### C35. `hospital` no invalida el banco; invalida la geometria elegida

Las 767 series de `hospital` tienen longitud 84. Con 60% inicial, tres origenes
y ocho rezagos, cada ventana queda por debajo del minimo mecanico. El hallazgo
ocurrio antes de cualquier score.

No se autoriza eliminar el panel, reducir el screen a cinco ni descargar un
reemplazo. La solucion menos arbitraria es usar **dos origenes comunes** para
los seis paneles, conservando el 60% inicial. Para longitud 84 produce dos
ventanas consecutivas de 17 observaciones; no crea una excepcion para hospital
y mantiene origenes/semillas como mediciones anidadas.

El draft v5 debe recalcular poblacion, ventanas, costos previstos y simulacion
de decision bajo esta geometria. Si el MLP no dispone de suficientes ejemplos
de ajuste o score bajo sus propias reglas, debe rehusar antes del sello; no se
reducen rezagos ni se debilita el modelo para hacerlo caber.

## Estado

No se calculo ningun score, no se creo ledger cientifico y no se descargaron
datos. La orden asociada es
`MUSASHI_TO_GENERAL_SATOSHI_T2_C31_C36_FINAL_SCREEN_DESIGN_ORDER_2026_09_06.md`.


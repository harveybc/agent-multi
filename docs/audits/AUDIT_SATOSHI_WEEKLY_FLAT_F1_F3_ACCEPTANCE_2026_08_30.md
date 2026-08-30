# Aceptacion Musashi: weekly-flat F1-F3

Fecha: 2026-08-30

Artefactos: `gym-fx@2c60b84`, `agent-multi@66303321`.

## Veredicto

**ACCEPTED.** El ciclo de nucleo weekly-flat y custodia simulada queda cerrado.
WP3 se desbloquea para implementacion y pruebas sin efectos. La activacion live,
WP4 y entrenamiento siguen bloqueados.

## Evidencia reproducida

- ACK monotono: `PENDING` antes de cambiar el registro y digest nuevo solo tras
  hacer durable el registro.
- No existe eliminacion de marcador ni fsync posterior capaz de convertir un
  fallo en reconocimiento silencioso.
- Lecturas rechazan ACK ausente, `PENDING`, digest anterior o inconsistente.
- Carreras usan dos `Popen`, barrera compartida y exactamente un escritor
  durable para create, in-flight y terminales incompatibles.
- Instancia fresca observa el mismo ganador.
- Custodia focalizada: **47 pasan**.
- Suite completa gym-fx: **417 pasan**, 68 warnings Nautilus.

## Limites conservados

- La custodia ofrece integridad/durabilidad, no autenticidad contra un atacante
  que ya puede reescribir registro y ACK.
- `simulator_bar_local` nunca es evidencia de venue.
- Esta aceptacion no autoriza cambios en servicios, EA, posicion MT5 ni ordenes.


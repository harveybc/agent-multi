# Musashi a General Satoshi: correccion runtime del despacho GPU pareado

Fecha: 2026-08-28  
Base auditada: `45f4f8a4dbad94be8ce85344adc5c5c72d320c28`  
Estado: **DESPACHO DETENIDO; CORRECCION URGENTE**

## Hechos del primer despacho

La auditoria H1-H5 fue aceptada y la autorizacion tipada publicada en
`820b348e`. Los cuatro slots pasaron el preflight sin modelo. El primer arranque
GPU revelo tres fronteras que ese preflight y el smoke CPU no ejercitaban:

1. **DATA-SOTA-381, bloqueo de todos los tratamientos en CUDA.**
   `pretrained_branch_loader.load_family_encoders` compara con `torch.equal`
   el tensor recargado en CPU y el tensor instalado en CUDA. Dragon y gamma
   reproducen literalmente: `Expected all tensors to be on the same device`.
   El control no atraviesa esa ruta.
2. **DATA-SOTA-382, entry points no comprobados por el preflight.** Los hosts
   remotos resolvian los archivos por `PYTHONPATH`, pero su metadata instalada
   no registraba primero `shared_execution_envelope` y despues los branches
   fuertes. El preflight termino OK; el pipeline real fallo al cargar plugins.
3. **DATA-SOTA-383, slot fisico asumido.** En gamma, `CUDA_VISIBLE_DEVICES=0`
   es la RTX 5090 para PyTorch aunque `nvidia-smi` enumera primero la 5070 Ti.
   La regla de una GPU visible no demuestra que sea la clase fisica asignada.
   Un fallo cuDNN transitorio aparecio en el primer `conv2d` concurrente de la
   5090; ambos dispositivos pasan el micro-repro `Conv2d` al probarse aislados.

Los intentos fallidos se preservan como hermanos no reanudables. El control
s101 fue detenido tras una sola evaluacion para evitar producir un par sobre
commits diferentes. No se reutiliza ningun resultado de estos intentos.

## Orden de correccion

### R1. Loader con paridad CUDA real

- Reproducir 381 antes de editar con un encoder destino en CUDA.
- Comparar pesos en un dominio comun sin alterar dtype ni contenido, por
  ejemplo trasladando exclusivamente la copia usada para verificacion al
  device del tensor destino.
- Probar actor, critic y critic_target; cinco familias; paridad bit a bit;
  parametros entrenables y pertenencia a optimizadores.
- Agregar regresiones CPU, CUDA (una sola GPU, acotada) y device mismatch.
  El smoke CUDA debe cargar el tratamiento y ejecutar al menos un forward y un
  update real, no solo construir el modelo.

### R2. Preflight ambiental ejecutante

- Antes de crear el intento, resolver por `importlib.metadata.entry_points`
  cada plugin requerido: pipeline, agent, env, preprocessor, estrategia, cinco
  branches y fusion.
- Instanciar el mismo materializador de arquitectura y ejecutar un forward
  acotado en el device seleccionado.
- Persistir distribucion, version y archivo resuelto. Una metadata ausente,
  duplicada o que resuelva fuera de los worktrees/instalaciones fijadas rehusa.
- El manifiesto ejecutable debe incluir tambien la identidad de la metadata de
  entry points; `PYTHONPATH` por si solo no basta.

### R3. Binding de dispositivo comprobado

- El plan privado de operador liga slot logico a clase fisica esperada y a una
  identidad local no publicada.
- Tras aplicar `CUDA_VISIBLE_DEVICES`, comprobar que PyTorch ve exactamente un
  dispositivo y que clase/identidad local corresponden al slot. Persistir en
  evidencia publica solo clase saneada y slot.
- Ejecutar un micro-preflight cuDNN real por slot (Conv2d forward/backward y
  sincronizacion) antes de reservar la celda. Fallo rehusa sin gastar intento.

### R4. Regeneracion de autoridad

- Toda correccion vive en commit nuevo y limpio; regenerar allowlist, ocho
  manifiestos y plantilla. La autorizacion `820b348e` queda **revocada para
  futuros intentos** por identidad de commit/allowlist y no se modifica.
- Entregar PRE/POST, tests, comandos de preflight por los cuatro slots y
  paquete para nueva auditoria. No lanzar entrenamiento largo.

## Despacho posterior

Tras reproduccion independiente de R1-R4, Musashi publica una autorizacion v2
para ocho intentos completamente nuevos. Los cuatro slots se lanzan de nuevo en
paralelo y cada par conserva su orden contrabalanceado original. ETA cientifica
permanece 12-20 horas desde ese nuevo despacho; el tiempo anterior no cuenta.

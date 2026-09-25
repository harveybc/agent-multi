# Orden Musashi: correccion contractual WP3

Fecha: 2026-08-30

Solo implementacion/read-only. Sin despliegue, comandos ni cambios de servicio.

## C1. Atadura temporal

Eliminar autoridad del `observed_at` libre del sobre. Atar byte a byte el
timestamp interno de cada payload a la evidencia admitida y calcular desde el
esa edad. Si el transporte aporta timestamp, debe estar firmado/ligado al cuerpo
y concordar. Congelar stale-payload/fresh-wrapper y future-payload.

## C2. Atadura de identidad

Verificar contra politica:

- cuenta interna/fingerprint;
- simbolo de cada posicion, orden, barra y tick;
- venue/schema/source;
- coherencia entre todos los sobres usados en una decision.

Prohibido sumar hechos de simbolos distintos. Duplicados de identidad o
posiciones contradictorias rehusan.

## C3. Tipos por venue

MT5: numeros JSON estrictos no booleanos. Alpaca: strings numericos solo donde
la API real los define, con gramatica decimal finita explicita. Congelar bool,
whitespace, exponentes no permitidos, NaN/Inf y strings en campos MT5.

## C4. Autoridad obligatoriamente fijada

Hacer `expected_code_identity` obligatorio en API, CLI y config. Atarlo al
manifiesto revisado y re-hashear los archivos justo antes del dry-run. Ausencia,
digest viejo o checkout sucio rehusan.

## C5. Completar WP3.3

Materializar custodia LTS privada y read-only en este ciclo: puede planificar y
verificar transiciones con fakes, pero no enviar efectos. Debe atar
venue/cuenta/simbolo/posicion/politica/calendario/codigo y exigir evidencia
directa para confirmar. Reinicio y multiplicidad fail-closed.

## C6. Reparar contrato MT5 sin desplegar

Agregar `time_open_unix` al modelo estricto con tipo/rango correctos y prueba de
integracion completa usando los bytes que produce el EA: endpoint FastAPI,
persistencia, lectura y parser WP3. Probar que un snapshot valido ya no recibe
422 y que extra desconocido sigue rehusando.

## C7. Evidencia Alpaca real read-only

Capturar desde el almacenamiento/cliente de lectura existente payloads
sanitizados reales. No abrir nuevas conexiones si ya hay evidencia durable.
Documentar campos ausentes y rechazar formas no cubiertas; no inventar brackets.

## C8. Retorno

Entregar PRE/POST, pruebas de cada bypass, suites LTS y paquete WP3 completo.
Mantener `writes_performed=0`. La aceptacion desbloqueara WP4; no autorizara
automaticamente activacion live.


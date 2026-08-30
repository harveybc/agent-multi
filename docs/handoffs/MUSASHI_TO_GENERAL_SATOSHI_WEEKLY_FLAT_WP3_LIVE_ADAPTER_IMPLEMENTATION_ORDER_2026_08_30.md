# Orden Musashi: WP3 adaptadores live weekly-flat

Fecha: 2026-08-30

Autorizacion: **IMPLEMENTACION Y PRUEBAS SIN EFECTOS SOLAMENTE**. No desplegar,
no reiniciar servicios, no enviar comandos y no tocar la posicion MT5 existente.

## WP3.1. Evidencia directa por venue

Implementar en LTS sobres/parsers estrictos para Alpaca y MT5 derivados de los
payloads ejecutantes reales:

- sesion terminal/cuenta y timestamp observado;
- posiciones con identidad, lado y cantidad;
- ordenes abiertas con identidad y rol verificable;
- proteccion nativa aceptada;
- ultima barra/cotizacion y calendario ligado al simbolo.

La politica, no el payload, fija fuentes permitidas y edad maxima. Rehusar
campos desconocidos, duplicados, coerciones, evidencia rancia o identidad ajena.
Rehusar expresamente `simulator_bar_local` y todo `venue_direct=false`.

## WP3.2. Adaptador de decision sin duplicar politica

Los runners Alpaca y MT5 consumen la autoridad pura ya aceptada; no reimplementan
sus estados. Deben traducir de forma tipada:

- NORMAL: decision ordinaria;
- WIND_DOWN: bloquear nuevas entradas y cancelar solo entradas pendientes;
- FORCED_FLATTEN: cancelar entradas, conservar proteccion hasta el cierre,
  solicitar cierre y esperar confirmacion directa cero/cero;
- EXPECTED_MARKET_CLOSED: ningun paso accionable;
- REOPEN_BLACKOUT: bloquear entradas hasta evidencia causal suficiente;
- RECOVERY: bloquear riesgo hasta disposicion/evidencia directa valida.

Publicar salida cruda, comando mapeado, overlay y comando final por separado.

## WP3.3. Custodia desplegable

Materializar custodia local privada por venue/cuenta/simbolo usando el protocolo
aceptado, con identidad de codigo/politica/calendario y evidencia directa. Un
reinicio restaura la obligacion; una cuenta o posicion distinta no puede
confirmarla. Varias obligaciones exigen disposicion del operador.

## WP3.4. Watchdog consciente del calendario

Clasificar sin ambiguedad:

- cierre esperado del mercado;
- feed rancio durante ventana abierta;
- terminal o cuenta desconectados;
- flatten fallido;
- exposicion inesperada durante cierre;
- recuperacion activa.

El cierre esperado solo suprime la alerta de barra rancia; nunca terminal,
cuenta, proteccion, ordenes o exposicion.

## WP3.5. Paridad y pruebas sin efectos

Usar capturas sanitizadas/read-only y fakes ejecutantes. Probar ambos venues:

- los cinco estados y sus fronteras exactas;
- largo, corto, plano, brackets y entrada pendiente;
- cancelacion/close rechazados, timeout, fill en carrera y reinicio;
- festivo adyacente, DST, fin de semana y feed roto un martes;
- igualdad de decision core-vs-adaptador para hechos equivalentes;
- cero llamadas de escritura en todo el paquete WP3.

Construir un dry-run que produzca el comando que se habria enviado y su razon,
pero cuya interfaz no posea credenciales ni cliente capaz de escribir.

## WP3.6. Retorno

Entregar mapa de llamadas, esquemas, PRE/POST, pruebas, capturas sanitizadas y
runbook de futura ventana coordinada **propuesto, no ejecutado**. Declarar por
separado cualquier limitacion de Alpaca y MT5.

La aceptacion independiente de WP3 desbloqueara WP4 y la preparacion de una
ventana live. Ninguna activacion sera automatica por esta orden.


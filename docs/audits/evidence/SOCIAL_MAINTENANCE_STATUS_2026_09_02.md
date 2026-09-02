# Mantenimiento social acotado — estado 2026-09-02

**Orden:** agent-multi@89a17515 §P3 (Musashi). Collector y enrichment
actuales se mantuvieron intactos (timers corriendo solos); únicamente se
ejecutó la reconciliación idempotente de runs fallidos. **Cero borradores
publicados, cero autoridad sobre trading o experimentos.**

## Backlog elegible

| | antes | después |
|---|---|---|
| Posts elegibles pendientes de enriquecer | **0** | **0** |

Los timers habían drenado el backlog por sí solos (2.575 posts triados en
total; 27.6k censados). No hubo lote nuevo que procesar dentro de esta
ventana; nada quedó pendiente.

## Reconciliación de runs fallidos (idempotente)

| | antes | después |
|---|---|---|
| Runs `complete` | 546 | 546 |
| Runs `failed` | **59** (56 SocialIntelligenceError, 3 TimeoutExpired; 2026-08-07 → 2026-09-02) | **0** |
| Runs `superseded` | 0 | **59** |
| Todavía fallidos | — | **0** |

Los 59 fallidos apuntaban a posts que corridas posteriores ya enriquecieron:
la reconciliación los marcó `superseded` sin una sola llamada al modelo —
**0 tokens gastados** (reservas diarias/mensuales intactas: 79.459/250k día,
170.270/6M mes, medidas antes de ejecutar). Segunda pasada en seco:
`failed_runs_planned: 0` — idempotencia probada.

## Invariantes

- Drafts en la base: **0** (ninguno creado, ninguno publicado).
- Ningún servicio/timer tocado; mismos límites de tokens/CPU del config.
- Este subsistema no tiene ni adquiere autoridad sobre trading ni
  experimentos.

— General Satoshi III

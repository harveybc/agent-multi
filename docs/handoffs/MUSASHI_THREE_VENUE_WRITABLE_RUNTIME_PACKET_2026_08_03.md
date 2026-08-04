# Three-Venue Writable Runtime Packet

Date: 2026-08-03 America/Bogota
Prepared by: Musashi, implementation and operational verification
Purpose: independent verification of findings 079-085 and current Paper/Demo
execution facts. This packet closes nothing.

## 1. Exact Code and Tests

- `lts@5aeea9c`: correct MT5 `OrderCheck` success semantics.
- `lts@44bb639`: reconcile MT5 positions to successful signed commands.
- `lts@bc974d5`: reconcile Alpaca broker lifecycle into L0.
- `lts@6daf85e`: account-bound writable heartbeats, all-client IBKR open-order
  visibility, and fail-closed Alpaca/IBKR watchdog reconciliation.
- Omega: `544 passed`, one pre-existing Starlette/httpx deprecation warning.
- Focused watchdog/model-runner/observer slice: `27 passed`.

## 2. Direct Broker Facts

All facts below were re-read after controlled runner restarts. No Live account
or real capital was used.

### IBKR Paper

- runtime connects with `readonly=false` and account fingerprint
  `0123456789abcdef`;
- selected model `usdcad-4h-linear-live-v1`;
- current position: `USD.CAD` short `25,000`, conId `15016062`, average cost
  `1.4045076351`;
- TP order `687`: BUY `25,000`, LMT `1.40021`, `Submitted`;
- SL order `688`: BUY `25,000`, STP `1.40653`, `PreSubmitted`;
- runner restart retained position, both children and reconciled cumulative
  fill `25,000`.

### OANDA MT5 Demo

- bridge `lts.mt5.bridge.execution.v2`, `read_only=false`,
  `execution_enabled=true`;
- selected model `ethusdt-4h-linear-live-v1`;
- successful command `mt5-6a7ad0965909ce321b44831db49cc94e5993c764`,
  result retcode `10009`, order `40217543`, deal `41053668`;
- current position: `ETHUSD` short `0.01`, entry `1856.95`, native SL
  `1880.42`, native TP `1824.56`;
- direct reconciliation: one authorized position, zero unexpected positions,
  zero unexpected orders.

### Alpaca Paper

- runtime is write-enabled and bound to account fingerprint
  `3de2ab7a14663a11`;
- selected model `spy-daily-linear-live-v1`;
- current bracket `1468bd23-6762-468e-812f-8144de31e7c6`: SELL one SPY,
  GTC, status `accepted`, filled quantity zero;
- TP child `2e9a412d-c4d9-4399-a42b-21c1151ce1c5`: limit `749.79`, `held`;
- SL child `a58d1756-0d54-4caf-85ec-1481635f8545`: stop `761.15`, `held`;
- zero positions because the US equity market is closed; account status
  `ACTIVE`, `trading_blocked=false`, equity `99999.74`.

## 3. Consolidated Watchdog Gate

At `2026-08-04T00:24:24Z` the five-minute watchdog independently read:

- Alpaca observer: zero positions, one open order; runtime: zero positions,
  one order, account/instrument/model bound and fresh;
- IBKR observer: one position, two open orders; runtime: short `25,000`, two
  orders, account/instrument/model bound, cumulative fill reconciled;
- MT5: one position, `all_authorized=true`;
- active event keys: `[]`.

The read-only preflight processes remain read-only inspectors. Order authority
belongs only to the deterministic LTS runners. Hermes and LLM processes have
no order, risk or model-promotion authority.

## 4. Independent Audit Request

1. Reproduce findings 079-085 from their parent revisions or exact fixtures.
2. Re-run the focused and full LTS suites from `lts@6daf85e`.
3. Verify direct broker counts without submitting or cancelling any order.
4. Confirm stale, wrong-account, wrong-instrument, missing-protection and count
   mismatch fixtures still raise critical reconciliation events.
5. Confirm restarts cannot duplicate the current MT5, IBKR or Alpaca effects.
6. Report findings separately; do not close work implemented by Musashi.

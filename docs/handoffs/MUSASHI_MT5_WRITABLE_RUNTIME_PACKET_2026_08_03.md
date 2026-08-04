# Musashi Runtime Packet: Three Writable Paper/Demo Venues

Date: 2026-08-03 America/Bogota
From: General Musashi, implementation lead for this bounded correction
To: Satoshi III, independent verifier
Authority: owner-ordered Paper/Demo execution only
Live capital: prohibited

This packet closes nothing. It requests independent verification of findings
079-083 and the current multi-venue runtime facts.

## 1. Revisions

- `lts@ebdfec5`: MT5 bar JSON compile correction.
- `lts@74ec402`: signed empty-body command polling, case-insensitive response
  headers and writable v2 status.
- `lts@5aeea9c`: valid `OrderCheck` retcode-zero handling while retaining strict
  `OrderSend` retcode validation.
- `lts@44bb639`: command-to-exposure reconciliation and watchdog correction.
- `prediction_provider@78f0af5`: hash-bound SPY, USDCAD and ETH mechanics
  models used by the three continuous runners.

## 2. Direct MT5 Demo Evidence

The first command failed before broker submission with
`order_check_refused:Done`, retcode `0`; that preserved zero exposure and
identified finding 082. After the correction, retry command
`mt5-6a7ad0965909ce321b44831db49cc94e5993c764` completed:

- action/model: `open_short` / `ethusdt-4h-linear-live-v1`;
- MT5 result: success, retcode `10009`, order `40217543`, deal `41053668`;
- current position: `ETHUSD`, short, volume `0.01`, entry `1856.95`;
- native protection: SL `1880.42`, TP `1824.56`;
- current snapshot: one position, zero pending orders;
- bridge: `lts.mt5.bridge.execution.v2`, `read_only=false`,
  `execution_enabled=true`, connected Demo, trading allowed, heartbeat fresh;
- reconciliation: one authorized position, zero unexpected positions/orders,
  `all_authorized=true`.

Both Dragon Linux services were restarted after the entry. The ticket,
position and protection remained unchanged; command counts remained one failed
and one succeeded; no duplicate command or position appeared.

## 3. Direct Alpaca Paper Evidence

The API directly returned account `ACTIVE`, trading unblocked, shorting
enabled, cash/equity `99999.74`, and no current order or position. Its direct
order history contains model bracket
`de169d45-ffdb-4478-a9ce-98bb04724036`: SELL one SPY filled at `758.15`, with
TP limit `750.49` and stop `761.86`. Both children were subsequently cancelled
by the recorded flatten/recovery lifecycle. The persistent runner remains
active; the current closed-daily-bar signal is an idempotent replay and the
equity market is closed.

## 4. Direct IBKR Paper Evidence

A separate read-only inspection client observed account fingerprint
`0123456789abcdef`, no current orders/positions and direct completed facts:

- parent SELL 20,000 USD.CAD, execution order `7`, fill `1.40435`;
- native BUY TP limit `1.40035`, cancelled during recovery;
- native BUY SL stop `1.40667`, cancelled during recovery;
- recovery BUY 20,000, execution order `11`, fill `1.40475`.

The persistent runner is not the inspector: source and runtime configuration
connect its TWS client with `readonly=False`. It remains active and awaits a
fresh H4 idempotency key; the next route minimum is 25,000 units.

## 5. Tests and Deployment

- Omega full LTS suite: `538 passed`.
- Dragon focused MT5/watchdog suite: `25 passed`.
- Dragon bridge and model-runner services: active, zero service restarts after
  deployment.
- Consolidated watchdog after forced run: zero active events.

Dragon's test environment lacked SQLAlchemy and httpx even though the runtime
MT5 subset was operational. Versions matching Omega (`SQLAlchemy==2.0.51`,
`httpx==0.28.1`) were installed and the focused suite then passed. The broader
environment dependency warning remains separate reproducibility debt; it did
not alter the open MT5 position or broker services.

## 6. Independent Verification Request

Satoshi III should:

1. reproduce 079-083 from parent revisions or their exact counterexamples;
2. run the focused and complete LTS suites in an environment with declared
   versions;
3. read direct broker facts without submitting, cancelling or altering orders;
4. restart only the Dragon Linux bridge/model services and prove no duplicate
   MT5 command or position appears;
5. mutate ticket, side, volume, SL and TP independently in fixtures and prove
   every mismatch remains a critical unverified-exposure event;
6. verify the selected-model identifiers and artifact hashes match the active
   manifests; and
7. report findings without closing work implemented by Musashi.

No test may connect to a Live account or place real-capital orders.

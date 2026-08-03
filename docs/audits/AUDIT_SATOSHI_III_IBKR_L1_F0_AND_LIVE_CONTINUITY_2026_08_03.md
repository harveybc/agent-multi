# Audit: IBKR L1 F0 and Selected-Model Live Continuity

Date: 2026-08-03 America/Bogota
Auditor: General Musashi, temporary independent auditor
Scope: Satoshi III F0 request for findings 069-074; subsequent Paper canary
and reconnect correction at `lts@cffdc13`
Mutation authority: Paper/Demo only; no Live capital

## Verdict

Findings 069-074 are independently reproduced as corrected and may proceed to
owner closure. The integrated LTS suite passes `530` tests. This audit does not
close those findings.

The first IBKR selected-model Paper canary exposed a new S3 continuity defect,
075. It is implemented at `lts@cffdc13` and awaits independent verification by
Satoshi III.

Runtime restart testing exposed three additional bounded defects. Findings
076-078 are corrected through `lts@8b67235` and await independent verification:

- quote error 10197 now produces a recoverable `waiting_for_quote` heartbeat
  with zero submissions instead of a systemd restart loop;
- terminal cumulative fills inside `1e-9` tolerance are canonicalized to the
  exact requested magnitude before execution-contract validation, and negative
  cumulative facts are refused;
- future L0 payloads identify their actual venue instead of labeling Alpaca as
  an IBKR adapter. Historical append-only payloads are not rewritten.

## Direct Broker Evidence

The selected `usdcad-4h-linear-live-v1` model emitted a short signal for the
closed H4 bar at `2026-08-03T16:00:00Z`. LTS submitted one protected Paper
bracket:

- parent: sell 20,000 USD.CAD, broker order 7;
- take profit: buy 20,000 at 1.40035, broker order 8, GTC;
- stop loss: buy 20,000 at 1.40667, broker order 9, GTC.

TWS accepted and filled the parent and retained both children. It warned that
20,000 is below the 25,000-unit IDEALPRO minimum and would use odd-lot routing;
this was not a rejection.

After reconnect, the completed parent was absent from `reqOpenOrders` while a
completed-order fact with permanent id 1193220731 and its matching execution
remained. Exact protection verification therefore failed closed, cancelled the
children and flattened 20,000 units with broker order 11. Direct TWS evidence
then showed zero positions and zero open orders. The effect reached
`terminal_flat`; the Paper cost remains in account and OLAP evidence.

## Correction 075

`IbAsyncTwsClient.open_order_facts()` now combines open orders with completed
orders that have a direct matching execution by `permId`. It does not invent a
fill from order status, requested quantity or a missing fact. Focused tests
cover matching reconstruction, refusal without a matching execution, and
exact bracket verification after reconnect.

The USDCAD route and newly minted Paper mandate now use 25,000 units. Gross and
margin fractions are capped at 0.04 and stop risk at 0.00625% of Paper equity.

## Multi-Venue Continuity

- Alpaca Paper: SPY daily selected model active, heartbeat fresh, zero current
  positions and open orders after the prior signal was consumed.
- IBKR Paper: USDCAD H4 selected model active, heartbeat fresh, direct account
  flat, halt cleared after operator reconciliation.
- OANDA MT5 Demo: backend bridge and ETHUSD H4 runner active on Dragon, but the
  old `lts.mt5.bridge.readonly.v1` EA is still attached. No model bars or
  commands can flow until the owner installs `LtsMt5ModelBridge.mq5`.

All selected-model manifests are hash-bound and hot-reloaded. Model replacement
must first drain old exposure and use actual post-close broker cash/equity as
the next session's starting balance. Hermes and LLMs have no order authority.

## Evidence Gate

Before declaring MT5 continuous execution accepted, independently observe:

1. bridge version `lts.mt5.ea.execution.v2` and `read_only=false`;
2. at least 51 closed ETHUSD H4 bars delivered;
3. one selected-model command acknowledged by the EA;
4. direct MT5 position/order evidence containing both native SL and TP;
5. restart continuity without duplicate submission.

No Live account is authorized by this audit.

Direct post-restart preflights independently returned zero positions and zero
open orders for Alpaca Paper and IBKR Paper. Both local runners and all three
Dragon MT5 services were active with zero systemd restarts at the final sample.

# 22. Multi-Venue Paper Execution and Social Trading

## 1. Objective

Measure real broker contracts while per-asset optimization continues. LTS owns
one global portfolio and treats OANDA MT5, Alpaca and IBKR as replaceable
execution venues. No terminal or broker account owns allocation, portfolio
risk, customer policy or model selection.

This track produces execution evidence, not a profit claim. Paper fills remain
simulated and are compared with the canonical simulator and, later, tightly
capped live evidence.

## 2. Account State

User-reported state on 2026-07-29:

| Venue | Account | State | Intended paper role |
| --- | --- | --- | --- |
| Alpaca | Trading API | created and verified | API-native crypto observation and long-only control |
| IBKR | Individual Margin | created and verified | equities/ETF, FX and broad multi-asset paper execution |
| OANDA | Global Markets live | compliance review pending | future CFD venue after approval |
| OANDA | Global Markets MT5 demo | credentials not yet provisioned | FX and available crypto-CFD execution calibration |

Credentials, raw account IDs and recovery data must never be committed,
included in chat, written to chain/portable OLAP or copied into tracked JSON.

## 3. Runtime Boundary

```text
models and promoted artifacts
          |
prediction_provider
          |
LTS global portfolio state
  - canonical NAV
  - virtual sleeves
  - customer/global risk
  - venue capital reservations
  - target exposures
          |
capability-aware venue router
   |              |             |
OANDA MT5 EA   Alpaca API    IBKR API/TWS
```

LTS computes target exposure before venue selection. The router chooses only
among venues whose latest capability snapshot satisfies the asset, direction,
order, protection, precision, market-state and capital requirements.

## 4. Global Ledger and Routing Rules

- Demo balances are not added together. The experiment declares one synthetic
  base NAV and a capital limit for each venue.
- Global gross, net, asset, correlated-group and risk-at-stop limits dominate
  every local broker limit.
- Every decision receives `portfolio_intent_id`, `route_id`, `intent_id` and
  one venue capital reservation.
- Normal execution routes an intent to exactly one venue.
- A paper A/B comparison may duplicate an intent only in a separate namespace
  that is excluded from portfolio P&L and production routing.
- A timeout does not authorize automatic execution at another venue. LTS first
  reconciles the original route.
- Venue balances, positions, open orders and protection are reconciled into the
  canonical ledger before another risk-increasing action.

## 5. Protection Contract

Every risk-increasing order requires both SL and TP. An adapter is eligible
only when it can demonstrate:

1. broker-side protection that survives LTS/network failure;
2. accepted SL and TP associated with the resulting position;
3. unambiguous transaction/order identities;
4. restart-safe reconciliation;
5. fail-closed behavior when either protection leg is rejected.

If entry fills but protection cannot be confirmed, the adapter immediately
cancels residual orders, flattens the new exposure and raises a critical alert.
Client-side polling alone is not accepted as the stop-loss mechanism.

## 6. OANDA Global Markets MT5 Adapter

OANDA Global Markets uses MT5 rather than REST v20. The preferred boundary is a
small Expert Advisor:

- `OnTimer` polls an allowlisted signed LTS endpoint for commands;
- command envelopes include nonce, expiry, idempotency and account fingerprint;
- the EA validates symbol, market state, volume, margin, order type, SL and TP;
- `OrderCheck` precedes `OrderSend`;
- `OnTradeTransaction` reports acknowledgements, fills, protection changes,
  cancellations and closes;
- periodic full snapshots repair missed events;
- the EA never computes portfolio weights or model inference.

`WebRequest` is synchronous and unavailable in the MT5 Strategy Tester.
Therefore the EA is a live/demo transport only; Nautilus/Backtrader remain the
historical verification paths. A native Python-to-terminal adapter may be
tested behind the same contract, but cannot alter LTS ownership.

## 7. Alpaca Paper Adapter

Use the Trading API Paper endpoint and paper-specific keys. Initial role:

- capability and symbol discovery;
- crypto quote/order/fill observation;
- long-only paper control;
- API latency, rejection and restart testing.

Current documented crypto limitations include no shorting or margin and order
types limited to market, limit and stop-limit. Until Alpaca demonstrates native
SL+TP behavior satisfying section 5, it is observation/shadow-only for our
protected crypto strategy and cannot receive production exposure.

## 8. IBKR Paper Adapter

Use the verified Individual Margin account and its separate Paper credentials.
Initial role:

- equities/ETF and multi-asset instrument discovery;
- bracket-order and short-capability verification;
- TWS API versus Web API operational comparison;
- paper market-data entitlement discovery;
- position, margin and liquidation-state reconciliation.

Margin capability does not authorize portfolio leverage. Initial LTS policy
sets gross leverage to 1.0, disables margin borrowing and prohibits uncovered
short options. The account type preserves future test capability while LTS
retains conservative limits.

## 9. Phased Activation

### M0: Local secret and identity preflight

- create separate encrypted/local environment files per venue;
- verify paper/demo endpoint, account fingerprint and read-only login;
- record adapter/package/terminal versions;
- prove logs and OLAP redact account IDs and secrets.

### M1: Capability inventory

- instruments and canonical mappings;
- market hours and current tradeability;
- direction/short eligibility;
- order types, time-in-force and native SL/TP/OCO/bracket semantics;
- precision, minimum size, margin and financing/funding;
- market-data entitlement and freshness.

### M2: 24-hour read-only observation

Record p50/p95 spreads, quote gaps, API/terminal uptime, clock skew, reconnects,
session constraints and broker-versus-research symbol coverage.

### M3: Protected canaries

For each eligible venue, submit minimum-size long and short canaries during a
liquid session. Require confirmed SL+TP, restart reconciliation, duplicate
suppression and emergency flatten.

### M4: Seven-day consolidated shadow

Run one synthetic portfolio NAV across all adapters. Record intended versus
routed exposure, implementation shortfall, costs, rejected opportunity,
protection state, venue concentration and execution-adjusted weekly metrics.

### M5: Live decision

No account receives real capital automatically. A live pilot requires explicit
human approval, legal/account suitability, complete paper evidence and
venue-specific hard limits.

## 10. OLAP Facts

Persist:

- account capability snapshots by hash;
- quote/spread/availability observations;
- route decisions and rejected venues;
- capital reservations;
- intents, acknowledgements, fills and protection transitions;
- reconciliation differences and repair actions;
- per-venue and consolidated costs;
- mean weekly return, annual return when coverage permits, mean weekly RAP,
  annual RAP when coverage permits, drawdown and observed duration;
- adapter/config/code/artifact versions.

Paper and live facts use separate environment dimensions. Unlike demo account
balances, the synthetic consolidated NAV is comparable across venues.

## 11. Social-Trading Boundary

The current Individual accounts validate our own portfolio only.

- Signal-only distribution can later connect user-owned accounts through an
  approved OAuth/application route.
- Automatic copy trading or discretionary allocation across client accounts
  requires the appropriate Advisor structure and jurisdictional review.
- Onboarding and servicing brokerage accounts inside LTS requires a Broker API
  or introducing-broker relationship.
- Customer assets are never pooled in the current personal accounts.
- The production social-trading account/legal structure is a separate gate,
  not an assumption embedded in the paper adapter.

## 12. Immediate Inputs and Next Actions

1. Create the independent OANDA Global Markets MT5 demo; retain its login and
   `OANDA_Global-Demo-1` credentials locally.
2. Generate Alpaca Paper keys and store them in a local untracked environment
   file.
3. Activate/find the IBKR Paper username, verify TWS login, and confirm whether
   market-data sharing is enabled.
4. Choose a Windows host for the OANDA MT5 terminal/EA bridge.
5. Run read-only capability preflights before implementing or enabling orders.

## 13. Authoritative References

- OANDA Global Markets MT5:
  https://help.oanda.com/bvi/es/faqs/mt5-user-guide-bvi.htm
- OANDA Global Markets CFD eligibility:
  https://help.oanda.com/bvi/es/faqs/trade-cfds-eligibility.htm
- OANDA REST-v20 division availability:
  https://developer.oanda.com/rest-live-v20/introduction/
- MQL5 WebRequest:
  https://www.mql5.com/en/docs/network/webrequest
- MQL5 OrderSend:
  https://www.mql5.com/en/docs/trading/ordersend
- Alpaca Trading API versus Broker API:
  https://docs.alpaca.markets/us/v1.1/docs/broker-api-faq
- Alpaca crypto orders:
  https://docs.alpaca.markets/us/docs/crypto-orders
- IBKR Paper Trading:
  https://www.interactivebrokers.com/campus/glossary-terms/paper-trading-account/
- IBKR Trading Web API:
  https://www.interactivebrokers.com/campus/ibkr-api-page/web-api-trading/

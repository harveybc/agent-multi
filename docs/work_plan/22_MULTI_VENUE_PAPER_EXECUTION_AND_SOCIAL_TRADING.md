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

Runtime evidence on Omega:

- Alpaca Paper read-only preflight is authenticated and runs every five
  minutes; six crypto quote cells are available and exposure is zero;
- IBKR Paper adapter, OLAP and five-minute observer are authenticated against
  local TWS Paper port `7497`; all six initial contracts qualified with zero
  positions and zero orders;
- OANDA Global Markets MT5 remains unconfigured; MT5 credentials are not REST
  API credentials;
- the consolidated watchdog reports freshness, endpoint health, missing data,
  unexpected exposure and venue availability through Telegram.

MT5 host decision:

- Dragon is the persistent Windows 11 KVM/libvirt host: 30 GiB RAM, 32 CPU
  threads, hardware virtualization and 585 GiB free were manually verified;
- Gamma is excluded because it has 14 GiB RAM, limited free disk and owns the
  active RTX 5070 Ti plus RTX 5090 eGPU compute path;
- Omega remains the mobile supervisor and current TWS Paper host, so it is not
  the unattended MT5 anchor;
- KVM/libvirt host validation, persistent NAT, the official ISO hash, the
  effective VM XML and Windows Setup boot have passed on Dragon;
- `lts-mt5-paper` is running with 8 GiB RAM, 4 vCPU, 100 GiB sparse disk,
  UEFI/Secure Boot and TPM 2.0. Windows language, licensing, edition and
  account selection remain interactive; MT5 is not installed yet.

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

1. Keep the verified IBKR TWS Paper observer on Omega in read-only mode and
   retain port `7497`.
2. Create the independent OANDA Global Markets MT5 demo; retain its login and
   `OANDA_Global-Demo-1` credentials inside MT5, never in Git or chat.
3. Provision `lts-mt5-paper` on Dragon using KVM/libvirt, 8 GiB RAM, 4 vCPU,
   100 GiB sparse storage, UEFI, TPM 2.0 and NAT. Use only an official
   Microsoft x64 ISO and verify its published SHA-256. Wine and dual boot are
   not part of the primary path.
4. Complete 24-hour read-only observations before enabling protected canaries.
5. Implement and test the MT5 EA bridge before declaring OANDA active.

## 13. Hermes and Telegram Operations

Deterministic monitoring owns alerts. Hermes/DeepSeek is an analyst, not an
execution controller.

- five-minute watchdog: stale observers, broker/API failures, missing quotes,
  reconciliation exposure and venue availability;
- event discussion: one-hour and four-hour moves are surfaced for inspection,
  never converted directly into orders or queued optimization;
- 12-hour DeepSeek review: receives a sanitized evidence packet and reports
  health, business evidence, anomalies and up to three bounded offline
  experiment proposals;
- hard policy: no order placement, risk change, model promotion or job enqueue
  by Hermes; every experiment proposal requires human review.

Only Omega runs the bidirectional Telegram gateway for the shared bot token.
Other machines may send outbound deterministic alerts but must not compete for
Telegram updates.

## 14. Authoritative References

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
- Microsoft Windows 11 x64 ISO:
  https://www.microsoft.com/software-download/windows11

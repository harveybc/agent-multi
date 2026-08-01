# 27. Real-Time Feature and Asset Parity

Status: contract and fail-closed audit implemented; raw-bar integration and
numerical parity pending
Decision date: 2026-08-01

## 1. Decision

The live decision set is not limited to fields returned by the execution
broker. A policy may consume any source that is causally available before its
decision timestamp, including independent exchange feeds, public APIs,
economic releases, on-chain facts and internally derived features.

The execution venue remains a separate concern. For example, a crypto policy
may use Binance market state while submitting a protected CFD order to MT5,
provided symbol identity, basis, latency, trading-hour and outage behavior have
been measured. Research availability never implies live availability, and
live inference never implies permission to trade.

The machine-readable source of truth is:

```text
examples/config/live_parity/project3_realtime_feature_asset_contract_v1.json
```

The fail-closed audit is:

```text
python tools/project3_live_parity_audit.py \
  --experiment-config <resolved-experiment.json> \
  --live-contract examples/config/live_parity/project3_realtime_feature_asset_contract_v1.json \
  --require live_execution
```

## 2. Four Independent Planes

```text
causal source adapters
        |
        v
normalized point-in-time facts
        |
        v
shared feature and model-inference service
        |
        v
LTS portfolio/risk/execution routing -> Alpaca, IBKR or MT5
```

1. Source adapters collect raw facts and never make portfolio decisions.
2. Normalized facts retain `event_time`, `available_at`, source, instrument,
   venue, revision/vintage, quality, staleness and sequence information.
3. The feature service runs the same implementation used to materialize
   research data. Complex transforms are not independently rewritten in MQL5.
4. LTS consumes a versioned decision bundle, applies account risk and requires
   protected SL/TP execution. A broker terminal never owns portfolio policy.

This division allows a data provider, model and broker to be replaced
independently without silently changing the policy's meaning.

## 3. Promotion Gates

### Research eligible

- historical source has provenance and a point-in-time interpretation;
- split cutoffs and preprocessing are causal;
- source, feature and instrument mappings are registered;
- protected test data remains outside selection.

Research may include sources that are not yet available live. Such candidates
remain useful for measuring potential value and deciding whether an adapter or
subscription is worth building.

### Live-inference eligible

- every required raw input is integrated at runtime;
- warm-up history covers the largest feature and context window;
- offline replay and live computation match within declared tolerances;
- availability, staleness and missingness masks are part of training;
- point-in-time macro/event values use `available_at`, not observation period;
- cross-venue features pass basis, calendar and latency calibration;
- a missing non-degradable source rejects inference.

### Live-execution eligible

- all live-inference requirements pass;
- the asset has a qualified account-specific route;
- account state and positions reconcile;
- volume, margin and order semantics are calibrated;
- every risk-increasing entry has mandatory SL and TP;
- a protected paper canary and kill-switch test pass.

All gates are fail-closed. A quote observation, documented API capability or
historical file cannot substitute for an integrated feed and a parity test.

## 4. Input Lanes

| Priority | Lane | Useful inputs | Present disposition |
| --- | --- | --- | --- |
| 1 | Own-asset market state | closed OHLCV, returns, technical/statistical and causal decomposition features | Current active baseline; live bars and parity are missing |
| 2 | Execution microstructure | bid/ask, spread, trades, depth, imbalance, liquidity/session state | Quotes are being observed; historical materialization and parity are pending |
| 3 | Cross-asset market state | BTC/ETH state, USD/rate proxies, equities, bonds, metals and cross-venue basis | High-value candidate for rush, FX regime and portfolio layers |
| 4 | Crypto derivatives | funding, open interest, basis and liquidations | Point-in-time historical/runtime definition must match |
| 5 | Macro vintages | yield curve, inflation, employment, stress and USD indices | Historical FRED/ALFRED evidence exists; runtime adapter and vintage tests pending |
| 6 | Calendar and releases | schedule, consensus, actual, surprise and revision | Schedule-only features are usable; surprise fields remain blocked without first-available timestamps |
| 7 | On-chain and flows | active addresses, transfers, hash rate, exchange/stablecoin flows | Free sources first; cancelled CryptoQuant archive is research evidence only |
| 8 | Fundamentals and filings | SEC filings, earnings, ratios and corporate events | Future equity cells; not required for current FX/crypto alpha |
| 9 | Social context | source-backed technical events and hypotheses | Discovery input only until quality, manipulation and incremental-value tests pass |

Every lane is ablated. More data is not automatically better: a feature family
must improve validation utility or a required operational function after its
latency, missingness and recurring cost are included.

## 5. Asset Sets

### Core live candidates

These combine Project 3 evidence with at least one plausible current data and
paper-execution route:

| Horizon | Cells | Reason |
| --- | --- | --- |
| Short | `SOLUSDT@1h`, `BTCUSDT@1h`, `ADAUSDT@1h`, `ETHUSDT@1h` | strongest short/rush seeds with Alpaca, MT5 and/or Binance data possibilities |
| Medium/long | `SOLUSDT@4h`, `ETHUSDT@4h`, `DOGEUSDT@4h`, `USDCAD@4h`, `EURJPY@4h` | SOL has the strongest full-year screen; ETH has rush evidence; DOGE is a full-year control; USDCAD is the active DOIN cell; EURJPY is an active FX comparator |

### Diversification and execution controls

`EURUSD@4h`, `AUDUSD@4h`, `GBPJPY@1h`, `USDJPY@1h`, `NZDUSD@1h`,
`SPY@1h`, `TLT@4h` and `GLD@4h` remain in the matrix. Weak standalone alpha
does not disqualify a cell that provides marginal diversification, a risk
baseline or a broker-parity control.

### Research watchlist

`XRPUSDT@1h` remains a promising alternate but does not displace a core cell
until its incremental portfolio contribution or rush behavior is stronger.
Other inventory assets may enter this set after the same evidence audit; they
do not enter DOIN or live routing merely because a broker lists them.

The portfolio optimizer selects from frozen eligible cells and may assign zero
weight. Broker availability is a constraint, not an alpha ranking.

## 6. Instrument Identity

`SOLUSDT`, Alpaca `SOL/USD` and MT5 `SOLUSD` are related but not presumed
identical. The contract records provider aliases. A cross-venue decision route
also requires:

- paired timestamp samples and basis distribution by regime;
- quote-currency and contract-unit conversion;
- trading-session and weekend differences;
- spread, financing and liquidation semantics;
- maximum stale/basis thresholds and a tested fallback.

The same rule applies to spot, perpetual, CFD, ETF and cash-FX representations.

## 7. Current USDCAD Job

The active `USDCAD@4h` protected-easy job is research eligible. It uses 113
registered own-asset features derived from OHLCV and optimizes feature-family
masks, preprocessing, context, SAC parameters, risk and protected order mode.
It does not currently consume macro, calendar, order-book, on-chain or paid
features.

The job continues unchanged because its historical contract is valid. Its
artifact remains `not_for_live_orders`. Live promotion is blocked until a
closed-bar adapter, warm-up, shared feature computation and numerical replay
parity pass. This avoids discarding useful optimization while preventing an
offline-only model from reaching a broker.

## 8. Implementation Order

1. Expand read-only collection to the selected symbol union without enabling
   orders.
2. Persist normalized closed bars plus quotes/spreads from MT5, Alpaca and
   IBKR; add Binance as an independent crypto decision feed.
3. Extract one shared causal feature builder from the research materializer
   and expose it to batch replay and runtime inference.
4. Run offline-versus-runtime golden-vector tests for each feature profile,
   including warm-up, gaps, DST, weekends and incomplete bars.
5. Add FRED/ALFRED and free on-chain runtime adapters with `available_at`,
   vintage, staleness and availability masks; keep unlicensed surprise data
   blocked.
6. Measure 24-hour and seven-day venue basis/cost behavior for selected cells.
7. Run shadow inference with no orders, then protected canaries per venue.
8. Promote only passing cells into the portfolio candidate library.
9. For social-platform experiments, intersect that library with the platform's
   observable and protected-executable sets. Missing instruments remain
   measured constraints; they do not retroactively change alpha selection.

This work runs alongside the immutable DOIN job. It does not mutate the active
chain or consume protected test evidence.

## 9. Source References

- MQL5 CopyRates/CopyTicks: `https://www.mql5.com/en/docs/series`
- Alpaca real-time market data: `https://docs.alpaca.markets/us/docs/real-time-crypto-pricing-data`
- IBKR TWS market data: `https://interactivebrokers.github.io/tws-api/top_data.html`
- Binance streams: `https://developers.binance.com/en/docs/catalog/core-trading-spot-trading/api/ws-streams/~`
- FRED real-time/vintage periods: `https://fred.stlouisfed.org/docs/api/fred/realtime_period.html`
- SEC EDGAR APIs: `https://www.sec.gov/search-filings/edgar-application-programming-interfaces`

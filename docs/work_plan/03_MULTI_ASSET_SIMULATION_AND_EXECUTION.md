# 03. Multi-Asset Simulation and Execution

## 1. Purpose

`gym-fx` becomes the engine-adapter and Gym integration package for portfolio
research. It must not implement a new matching/accounting simulator. The
canonical engine will be selected by the bounded bake-off in
[14 Simulation Engine Selection](14_SIMULATION_ENGINE_SELECTION_2026_07_11.md),
with NautilusTrader as primary candidate, LEAN as second finalist, and the
current Backtrader runtime as parity oracle/fallback. Gymnasium is the learned-
agent interface, not a second financial simulator. LTS and live brokers consume
the same intents but own production execution state.

The current `gym-fx` `master` branch was rebuilt on Backtrader in April 2026.
Older `dev`, `dev2`, and `feature/tidying` branches contain obsolete custom Gym
environments and are not the implementation base. The branch audit is frozen in
`docs/handoffs/GYM_FX_BRANCH_AND_ENGINE_AUDIT_2026_07_11.md`.

## 2. Environment Scope

The multi-asset environment must support:

- many assets and timeframes in one account;
- concurrent positions across assets;
- virtual strategy cells targeting the same asset;
- shared cash, NAV, margin, conversion and exposure limits;
- deterministic replay;
- weekly retraining and allocation boundaries;
- broker-specific market hours and capabilities;
- one-cell compatibility with existing Project 3 behavior.

## 3. Master Event Clock

The master clock comes from the selected event-driven engine. Project code adds
explicit decision eligibility, age, stale, and missingness semantics but does
not independently schedule fills or maintain a competing account timeline.

Use the ordered union of all source timestamps. At each event:

1. update only assets with newly complete causal observations;
2. mark other asset observations with age/staleness masks;
3. update prices and account state;
4. process protective orders and market closures;
5. request decisions only for eligible cells;
6. aggregate cell intents to target instrument exposure;
7. execute deltas through the simulation broker;
8. record event-sourced facts.

A slow timeframe must not repeat the same decision on every fast-asset tick.
Forward-filled features are never presented without explicit age/missingness.

## 4. Weekly Walk-Forward Boundary

The default business schedule is:

- Friday/weekend: flatten according to asset/venue rule;
- freeze data cutoff;
- train/fine-tune candidate components;
- select/promote the next deployment manifest;
- compute next-week portfolio weights;
- begin trading at the next eligible market open.

The environment must permit independent retraining, rebalance and force-close
cadences for later experiments. Cadence is configuration, not hardcoded logic.

## 5. Shared Account Ledger

The selected engine's broker/account/portfolio state is authoritative. The
Project 3 event ledger records immutable observed events and account snapshots.
It must reconcile to the engine and must not independently invent fills or P&L.

Track at account level:

- initial cash, current cash, balance, NAV and equity;
- realized and unrealized P&L;
- margin used, available and closeout percentage;
- gross/net exposure by currency, asset class and correlated group;
- home-currency conversion;
- accumulated spread, slippage, commission, financing, dividend and conversion
  costs;
- pending orders, open logical cells, broker-net positions and closed trades;
- margin call, rejection, forced close and insolvency state.

Instrument-specific margin, contract multipliers, pessimistic spread/slippage,
commission, financing, and home-currency conversion must use the selected
engine's native account, risk, matching, and cost extension points. If an engine
requires us to recreate most of those systems, it fails the bake-off.

Per-cell attribution is virtual. Broker-facing position state is aggregated by
venue instrument.

## 6. Cell Aggregation and Netting

For customer/account-independent research, cell target notional is initially:

```text
cell_target_notional =
    portfolio_NAV
  * portfolio_weight[cell]
  * strategy_rel_volume[cell]
  * action_intensity[cell]
```

The environment aggregates all cell targets for one instrument:

```text
instrument_target = sum(cell_target_notional for matching instrument)
order_delta = instrument_target - current_instrument_notional
```

The attribution ledger distributes realized/unrealized results and costs back
to cells using a declared method. Initial method: proportional target exposure
with exact event timestamps. Attribution method is versioned because it affects
cell metrics but never changes account truth.

## 7. Sizing Modes

### 7.1 `legacy_notional`

Preserves Project 3 comparison semantics based on `rel_volume`, leverage and
notional caps.

### 7.2 `risk_at_stop`

Computes units from the permitted loss if SL executes:

```text
risk_budget = NAV * portfolio_risk_budget * cell_risk_weight
units = risk_budget / estimated_loss_per_unit_at_stop
```

Conversion, spread, gap/slippage allowance and commission are included in the
loss estimate. This mode is evaluated only after legacy reproduction passes.

## 8. Orders and Protective Geometry

Support:

- market, limit, stop and market-if-touched where venue permits;
- take profit, stop loss, trailing stop and modify protection;
- target-position reduction and full close;
- time-in-force and expiration;
- partial fills and rejection;
- minimum size, unit precision and price precision;
- position fill/netting mode;
- idempotency/client order identity.

When OHLC data cannot determine whether SL or TP occurred first inside a bar,
use the configured pessimistic ordering. Tick data may replace this ambiguity.
The collision policy is recorded in config and metrics.

## 9. Costs and Financing

Costs are asset, venue, timestamp and account-profile dependent. Required
components:

- bid/ask spread;
- adverse slippage model;
- commission and minimum commission;
- overnight financing/swap and triple-day rules;
- dividends/adjustments where applicable;
- currency conversion spread/cost;
- borrow/funding for applicable instruments;
- rebalance turnover cost.

Pessimistic profiles are valid controls, but a single fixed EUR/USD pip model
must not be applied to every asset.

## 10. Market Calendars and Weekend Rules

- Preserve Project 3 weekend-flat as the initial comparison profile.
- Configure closure by venue/asset, not by one global weekday assumption.
- Prevent new entries when the remaining tradable horizon is insufficient.
- Record forced-close reason and cost.
- Treat crypto 24/7 availability separately from an OANDA crypto CFD schedule.
- Never fabricate fills during a closed market.

## 11. Backtrader Compatibility

`heuristic-strategy` keeps Backtrader adapters for:

- historical regression of legacy strategies;
- independent single-cell validation;
- prediction-source substitution: ideal, CSV, API or model artifact;
- verification that entry, early close and SL/TP rules did not change.

The strategy core receives `DecisionContext` and returns `AssetIntent`.
Backtrader-specific `self.buy`, `self.sell`, `self.close`, commission and
`Cerebro` setup live only in the adapter/harness.

Required parity sequence:

1. fixed deterministic prediction stream;
2. fixed one-asset dataset and cost profile;
3. compare policy decisions timestamp by timestamp;
4. compare requested orders;
5. explain any fill/P&L difference caused by simulator semantics;
6. freeze a parity report before OANDA practice.

## 12. Baselines

Run under identical account and cost semantics:

- no trade/cash;
- always long and always short;
- buy-and-hold where meaningful;
- simple momentum and reversal;
- legacy prediction entry/exit heuristic;
- equal-weight active cells;
- inverse volatility;
- constrained minimum variance;
- constrained minimum semivariance;
- prior-week weights/no rebalance;
- oracle and anti-oracle as non-causal diagnostics only.

## 13. Raw Metrics Emitted

Per order/trade:

- entry/exit prices and times;
- gross/net P&L and return on initial account equity;
- MAE/MFE;
- duration and exposure;
- SL/TP/early-close reason;
- all costs;
- requested/fill divergence.

Per asset week and portfolio week:

- return, RAP and drawdown;
- downside deviation, semivariance and expected shortfall;
- trade count, hit rate, payoff ratio and active time;
- turnover and cost drag;
- gross/net/correlated exposure;
- margin utilization;
- concentration and diversification contribution.

## 14. Acceptance Criteria

- One-cell environment matches existing Project 3 results within declared
  tolerance for a frozen replay.
- Manual ledger fixtures reconcile cash, NAV, P&L, margin and costs exactly.
- Asset ordering does not change results.
- Future-row mutation cannot change an earlier observation/action.
- Closed markets and stale data cannot create fills.
- Same-instrument cells net correctly while retaining attribution.
- Backtrader and `gym-fx` produce matching policy decisions on the parity
  fixture.
- every persisted execution/account fact reconciles to the authoritative
  engine event or account snapshot that produced it;
- no custom ledger can generate a fill, position, cash value or P&L absent from
  the selected engine;
- Full validation-year replay emits all expected weekly facts without scale or
  period ambiguity.

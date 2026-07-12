# 14. Simulation Engine Selection

Status: NautilusTrader 1.230.0 selected after bounded bake-off
Date: 2026-07-11

## 1. Decision Principle

Do not build a matching engine, portfolio accounting engine, margin engine, or
broker simulator when a maintained production-grade implementation satisfies
the contract. Project code should own market-state construction, policies,
portfolio decisions, weekly walk-forward orchestration, metrics, lineage, and
DOIN integration. The selected engine should own order/account simulation.

## 2. Requirements

The canonical engine must support or cleanly permit:

- heterogeneous multi-asset and multi-timeframe events;
- shared portfolio cash, margin, exposure, P&L, and currency conversion;
- netting plus strategy/cell attribution;
- market, limit, stop, TP, SL, trailing, rejection, and margin-call behavior;
- pessimistic spread, slippage, commission, financing, and bar ambiguity;
- deterministic repeated full-year weekly walk-forward evaluation;
- Python policy/model integration and machine-readable config;
- efficient repeated resets for DOIN/DEAP candidate evaluation;
- complete order/trade/account event extraction for OLAP;
- eventual reconciliation with LTS/OANDA practice execution.

## 3. Candidate Assessment

### 3.1 NautilusTrader: primary candidate

Strengths:

- production-oriented Rust core with Python control plane;
- deterministic event-driven research and live architecture;
- native multi-instrument, multi-venue portfolio and risk engines;
- cash and margin account models;
- realized/unrealized P&L, exposure, and currency conversion;
- matching engine and configurable bar execution path;
- multiple timeframes use the more granular stream for execution;
- custom Python data and Parquet catalog support;
- reset/repeated-run APIs explicitly documented for parameter optimization;
- current active releases and security/reconciliation fixes;
- LGPL-3.0 library license.

Risks:

- current release line is labelled beta and contains breaking changes;
- no official OANDA adapter appears in the current supported-integration list;
- requires conversion of our bars/instruments/config into Nautilus domain types;
- the selected version must be pinned exactly and wrapped behind our contracts.

Primary sources:

- [NautilusTrader documentation](https://nautilustrader.io/docs/latest/)
- [Backtesting and repeated runs](https://nautilustrader.io/docs/latest/concepts/backtesting/)
- [Portfolio and currency conversion](https://nautilustrader.io/docs/latest/concepts/portfolio/)
- [Accounting and margin](https://nautilustrader.io/docs/nightly/concepts/accounting/)
- [Current integrations](https://nautilustrader.io/docs/latest/integrations/)
- [Custom Python data](https://nautilustrader.io/docs/latest/concepts/custom_data/)

### 3.2 QuantConnect LEAN: second finalist

Strengths:

- mature multi-asset portfolio, CashBook, margin, buying-power, order, fee,
  slippage, fill, calendar, and scheduled-event models;
- official open-source OANDA brokerage plugin for FX and CFDs;
- backtest, paper, and live operation;
- broad asset-class support;
- Apache-2.0 license.

Risks:

- C# engine with Python strategy interoperability increases integration and
  debugging cost for our TensorFlow/PyTorch/agent-multi stack;
- local operation commonly uses the LEAN CLI/Docker/project model;
- process and data initialization may be expensive for thousands of short DOIN
  candidate evaluations;
- the official OANDA model documents no backtest slippage by default, so our
  pessimistic profile still needs a custom reality model.

Primary sources:

- [LEAN engine](https://github.com/QuantConnect/Lean)
- [LEAN OANDA brokerage](https://github.com/QuantConnect/Lean.Brokerages.OANDA)
- [Brokerage reality models](https://www.quantconnect.com/docs/v2/writing-algorithms/reality-modeling/brokerages/key-concepts)
- [Algorithm and portfolio capabilities](https://www.quantconnect.com/docs/v2/writing-algorithms)

### 3.3 Backtrader: parity control and fallback

Strengths:

- already integrated and passing in current `gym-fx` master;
- Python-native and easy to wrap for Gym/DOIN;
- multiple feeds/timeframes, orders, commission schemes, credit interest, and
  existing strategy fixtures;
- lowest migration cost.

Risks:

- less complete native portfolio risk, multi-currency, and margin modeling;
- OANDA integration targets the legacy OANDA v1 stack;
- slower Python core for large distributed candidate workloads;
- GPL-3.0 license and older maintenance posture.

Primary sources:

- [Backtrader features](https://www.backtrader.com/home/features/)
- [Backtrader OANDA integration](https://www.backtrader.com/docu/live/oanda/oanda/)

### 3.4 vectorbt: screening accelerator only

vectorbt supports multi-column portfolios, cash sharing, fees, slippage, stops,
and flexible order functions with excellent throughput. It is useful for cheap
coarse sweeps and sanity checks, but it is not the canonical broker/account
engine for our event-rich weekly retraining and live-reconciliation workflow.

Primary source:

- [vectorbt Portfolio](https://vectorbt.dev/api/portfolio/base/)

### 3.5 Not selected as canonical engines

- FinRL: useful agent baselines and Gym examples, but its stock/portfolio
  environments are simpler than the required broker execution semantics.
- Zipline Reloaded: useful event-driven equity research, but not the best fit
  for heterogeneous FX/crypto/CFD execution and OANDA reconciliation.
- QSTrader: portfolio research oriented mainly toward systematic equities/ETFs.
- backtesting.py: concise single-strategy research API, not the required shared
  heterogeneous portfolio runtime.

## 4. Provisional Ranking

| Engine | Canonical role | Provisional result |
| --- | --- | --- |
| NautilusTrader | Full portfolio simulation and execution semantics | Primary candidate |
| QuantConnect LEAN | OANDA-aware full engine and independent validator | Second finalist |
| Backtrader | Frozen parity oracle and safe fallback | Retain |
| vectorbt | Fast screening and cross-checks | Optional auxiliary |

This ranking is provisional until the bounded bake-off passes. Feature lists
alone cannot promote an engine.

## 5. Mandatory Bake-Off

Use one identical, tiny, deterministic scenario:

- two instruments with different currencies;
- two timeframes and asynchronous bars;
- one shared margin account;
- open, partial close, reversal, and same-instrument netting;
- market, limit, SL, TP, and same-bar SL/TP collision;
- pessimistic spread, slippage, commission, financing, and conversion;
- insufficient-margin rejection and forced close;
- weekly no-new-entry and flatten boundaries;
- immutable order/trade/account event export.

Measure:

1. exact reconciliation to hand-calculated fixtures;
2. deterministic event and metric hashes across repeated runs;
3. no future-data influence;
4. reset/state-isolation correctness;
5. wall time and peak RAM for repeated candidates;
6. integration code and custom accounting code required;
7. extraction into existing `trading-contracts` and OLAP facts;
8. feasibility of wrapping one episode as a Gym/agent evaluator;
9. feasibility of running many independent DOIN evaluator processes.

## 6. Promotion Rule

Promote NautilusTrader if it passes accounting, determinism, event extraction,
and repeated-run throughput without requiring a parallel custom ledger. Missing
OANDA live integration is acceptable initially because LTS owns production
execution, provided we can reproduce the OANDA practice account profile and
later build or adopt a narrow adapter.

Promote LEAN instead if Nautilus cannot reproduce required OANDA-style
portfolio/margin semantics or cannot integrate efficiently with our Python
models and DOIN evaluator lifecycle.

Retain Backtrader as canonical only if both finalists fail the bounded spike or
their integration cost exceeds the proven benefit. In all cases Backtrader
remains the frozen heuristic parity oracle during migration.

## 7. Executed Bake-Off Result

The bounded spike was implemented in `gym-fx` with NautilusTrader pinned to
`1.230.0` in an isolated Python 3.12 environment. The following are executable,
not paper capabilities:

- two FX instruments with asynchronous 1-minute and 5-minute bars;
- one shared USD margin account;
- EUR/USD and USD/JPY currency conversion;
- target-position open, partial close, reversal and flatten;
- deterministic synthetic bid/ask spread and adverse slippage;
- maker/taker notional commission;
- standard initial/maintenance margin calculation;
- explicit pre-trade margin denial;
- long bracket where the pessimistic intrabar path reaches SL before TP;
- FX rollover at the New York boundary;
- immutable fill facts validated as `execution_report.v1`;
- repeated same-process and two-process deterministic hashes;
- a Gym-compatible single-cell bridge selected through JSON;
- Backtrader remains the default when `simulation_engine` is absent.

Three repeated multi-asset runs produced one result hash. The six-fill fixture
ended at `100001.28 USD`; the independent fixed-fixture reconciliation expected
`100001.2798625429553264604811 USD`, within currency rounding tolerance.

The fresh-engine microbenchmark measured approximately `6.3 ms` mean for the
Backtrader one-instrument subset and `30.6 ms` median for Nautilus running the
full two-instrument fixture. The workloads are not event-normalized; the result
only establishes that Nautilus fresh-run overhead is tens of milliseconds and
is negligible relative to model training and annual walk-forward evaluation.

Detailed evidence is frozen in:

```text
docs/handoffs/NAUTILUS_ENGINE_BAKEOFF_ACCEPTANCE_2026_07_11.md
```

## 8. Important Margin Finding

The stable Nautilus 1.230.0 Python `RiskEngine` source returns early for margin
accounts with a TODO for margin risk controls. The simulated margin account
calculates and reserves margin correctly but does not itself deny an oversized
margin order before execution.

The adapter therefore performs one narrow preflight using Nautilus's own
`MarginAccount.calculate_margin_init()` and native free balance/xrate. It does
not duplicate margin formulas, balances, fills or P&L. Every denial is emitted
with `CUM_MARGIN_EXCEEDS_FREE_BALANCE`. This guard is mandatory until an
upstream version provides equivalent native enforcement and passes the same
fixture.

## 9. Selection Decision

NautilusTrader is promoted as the canonical engine for new Phase 3 portfolio
simulation. Backtrader remains the frozen policy/parity control and immediate
fallback. LEAN is deferred and is not installed; its C#/Docker integration is
not justified while Nautilus passes the bounded requirements.

This decision selects the engine, not the finished portfolio environment. The
current replay adapter is multi-asset. The current interactive Gym bridge is a
single-cell compatibility slice and must next be expanded to portfolio-native
observations/actions while preserving the verified account engine.

# 28. Social-Trading Business Reality Loop

Status: architecture and local accounting vertical implemented; external
social-platform commissioning pending
Decision date: 2026-08-01

## 1. Decision

Social trading is a fourth experimental surface connected to, but distinct
from, alpha research, portfolio construction and broker execution. We will
learn its real capital, allocation, fee, replication and operational mechanics
through demo/virtual systems before accepting investor funds or publishing a
paid strategy.

The purpose is two-way improvement:

1. live and social observations correct simulation, fitness, genomes and
   portfolio controls;
2. optimized frozen artifacts return to shadow, protected canary and social
   demo validation;
3. only evidence from both directions can support a provider or managed-money
   pilot.

No platform becomes system architecture. LTS owns the canonical portfolio,
risk and customer/account ledger. Platform adapters translate one common
contract.

## 2. Five Independent Planes

```text
causal decision data
        |
shared features and inference
        |
portfolio targets and customer risk
        |
venue execution and reconciliation
        |
social allocation, fees and investor reporting
```

An execution broker, social platform or signal marketplace may implement more
than one plane internally, but LTS keeps their responsibilities separate. A
platform-managed copy result is measured as an external implementation, not
treated as the canonical portfolio state.

## 3. Experimental Axes

### Venue/API reality

Question: what can we observe and execute causally, with which assets, order
semantics, protection, precision, costs, sessions and failures?

Current controls: Alpaca Paper, IBKR Paper, OANDA Global MT5 demo and the
Capital.com GET-only adapter. cTrader Open API is the next independent API
candidate because it supports official Python clients and demo accounts.

### Copy/signal reality

Question: how do provider actions become subscriber exposure, and where do
sizing, latency, leverage, missing symbols, precision and protection diverge?

Immediate controls: cTrader Copy as a demo investor and eToro Virtual as a
manual product control. MetaTrader 5 build 4150 disabled Signals on demo
accounts, so the active OANDA MT5 demo remains a venue-reality stream rather
than a Signals experiment. Native cTrader Copy does not copy provider SL/TP,
so it remains observational under the project's mandatory-protection rule.

### PAMM/MAM reality

Question: how do subscriptions, pooled or segregated equity, units, HWM,
performance fees, management fees, flows, rollover and partial closes affect
manager and investor outcomes?

The first implementation is a provider-neutral decimal ledger in LTS. A real
PAMM account remains blocked until the ledger reproduces the selected
platform's rules and a separate legal/capital decision is approved.

### Provider/investable-track reality

Question: how does a trader build an investable track record, receive capital,
earn fees, support users and survive strategy drawdowns or retirement?

Darwinex Zero is the preferred next provider laboratory. It uses virtual
capital under live-like execution conditions, supports broad asset classes,
creates an investable index and exposes capital-allocation/performance-fee
mechanics without initially risking trading capital. Membership cost and
country eligibility require owner approval before subscription.

### Product/user-experience control

Question: what information, controls and friction does a social investor see?

eToro Virtual Portfolio is a free manual control for proportional copy, copy
open trades, add/remove funds, pause/stop and automatic reallocation. It is not
an LTS execution dependency.

## 4. Platform Priority

| Priority | Platform | Current role | Activation decision |
| ---: | --- | --- | --- |
| 1 | cTrader demo + Open API | Independent API/custom-copy and native Copy investor observation | Spotware demo is active; verify Copy catalogue and register Open API app; no live capital |
| 2 | eToro Virtual | Manual UX/allocation control | Owner verified target assets, Buy/Sell and CopyTrader UI; structured observation pending |
| 3 | Darwinex Zero | Virtual strategy-provider and allocation/business track | Hold: registration charges the first recurring subscription |
| 4 | MQL5 Signals | Future live-only signal/provider control | Demo is unsupported since MT5 build 4150; no funding authorized |
| 5 | HFM PAMM | Real pooled-capital mechanics | Deferred behind legal review, ledger parity and explicit funding limit |

Additional venues enter only when they add a missing asset class, execution
contract, social mechanic, jurisdiction or independent control. Account count
is not a success metric.

## 5. Implemented Neutral Contract

LTS owns:

```text
app/social_trading_lab.py
app/social_trading_cli.py
examples/configs/social_trading_platform_registry_v1.json
examples/configs/social_trading_accounting_scenario_v1.json
tests/unit/test_social_trading_lab.py
docs/SOCIAL_TRADING_REALITY_LAB.md
```

The contract implements:

- `Decimal` unitized investor accounting;
- flow-neutral NAV and proportional HWM adjustment;
- performance fees only above investor net HWM;
- prorated management fees;
- manager fee balances;
- equity-to-equity copy sizing;
- lot minimum, maximum and step behavior;
- explicit allocation/rejection/tracking-error facts;
- fail-closed protected-entry eligibility;
- immutable scenario and platform-registry hashes;
- SQLite run/event OLAP with `orders_submitted=0`.

It deliberately has no broker client, credential reader or order endpoint.

## 6. Business-Reality Feedback Contract

Every observation is normalized into one of four dispositions:

| Disposition | Meaning | Next action |
| --- | --- | --- |
| `measurement_only` | platform behavior without model implication | retain in OLAP and business report |
| `simulation_gap` | simulator omits or misprices observed behavior | add calibrated fixture/model and rerun validation |
| `optimization_variable` | behavior is controllable and materially affects profit/risk | register parameter/gene and schedule a new campaign |
| `hard_constraint` | legal, protection, asset, precision or account rule | fail closed in routing/promotion |

Examples:

- copy latency/slippage becomes a calibrated cost distribution;
- minimum useful copied size becomes an investor-capital constraint;
- lower subscriber leverage becomes a margin feature and missed-copy model;
- deposits/withdrawals become portfolio transition-cost scenarios;
- delayed/missing close signals become tail-risk penalties;
- fees convert gross RAP into after-fee investor RAP;
- platform risk transformations become source/investor beta and tracking
  models;
- rollover-induced partial closes become liquidity stress fixtures.

Observations never mutate an active DOIN job. A change receives a new config,
dataset and semantic hash and enters the next campaign or curriculum stage.

## 7. Asset and Input Policy

The research universe is not limited to one broker or social platform. A model
may use any causally available runtime source and an independent eligible
execution route under document 27.

The social experiment uses three nested sets:

1. **Provider/source set:** frozen models selected from the optimized
   per-asset library, including the most promising crypto and diversified FX,
   ETF and macro controls.
2. **Platform-observable set:** instruments available in the specific demo or
   virtual platform, including its contract and leverage semantics.
3. **Protected-executable set:** the intersection with verified account-local
   SL/TP, volume, margin, reconciliation and canary capability.

Missing instruments are evidence, not a reason to discard research alpha. The
portfolio may route a related representation only after basis, unit, session,
financing and tracking calibration. An unavailable or unprotected social copy
remains a measured control and receives zero authorized exposure.

## 8. Metrics

### Strategy/source

- mean weekly return and RAP, annual return/RAP only with complete coverage;
- drawdown, tail loss, turnover, activity and exposure;
- gross and after-execution metrics.

### Subscriber/investor

- gross and after-fee return/RAP;
- HWM, fees, cash flows, units and net equity;
- tracking error, missed/rejected exposure and time out of market;
- subscriber versus provider drawdown and recovery duration;
- protection coverage and close-latency tail loss.

### Provider/business

- assets under replication/allocation;
- investor count and concentration;
- gross revenue, platform fee, tax provision and support cost;
- capital retention, deposits/withdrawals and strategy churn;
- incident rate, disclosure/support burden and provider uptime.

Metrics are labeled by horizon and environment. Virtual provider revenue or
capital allocation is never relabeled as realized business income.

## 9. Commissioning Sequence

### S0: Local ledger and terms inventory

Implemented. Registry and deterministic accounting scenario pass tests and
persist no-order OLAP evidence.

### S1: Manual platform walkthrough

- cTrader Copy free strategy from a demo investor;
- eToro Virtual copy workflow;
- confirmation that MQL5 Signals is unavailable on non-real accounts;
- screenshot-free structured checklist of controls, limits and resulting
  facts;
- no credentials or personal identifiers in portable evidence.

### S2: API and reconciliation

- cTrader Open API demo authentication and capability discovery;
- read-only account, instrument, position/order and deal facts;
- custom-copy sizing simulation with account-local SL/TP planning;
- provider versus subscriber timing/price/exposure reconciliation;
- no order until the adapter is independently reviewed.

### S3: Protected demo canaries

- minimum-size long and short entries where the platform permits;
- both SL and TP accepted in the subscriber account;
- duplicate, stale, missing-symbol and disconnect tests;
- emergency local close and full restart reconciliation.

### S4: Seven-day social shadow

- one frozen provider strategy;
- synthetic investor profiles and real platform-demo facts;
- deposits, withdrawals, rebalance and fee crystallization scenarios;
- gross, after-fee and tracking metrics;
- zero unexplained exposure.

### S5: Provider-track experiment

- Darwinex Zero only after explicit membership approval;
- frozen strategy identity and start date;
- compare source risk/return with transformed investable-index behavior;
- record rankings, allocations, HWM and performance-fee mechanics.

### S6: Real PAMM or public provider decision

Requires explicit owner approval, professional legal/tax/account review,
platform-specific ledger parity, protected paper evidence, public disclosure,
support/incident policy and hard real-capital limits.

## 10. Security and Governance

- Credentials remain local, permission-restricted and outside chat/Git/chain.
- Demo/virtual and live identities never share an environment dimension.
- The local accounting lab cannot submit an order.
- Hermes may summarize sanitized metrics but cannot subscribe, publish,
  allocate, change fees or activate accounts.
- Social content cannot enter model features without source quality,
  manipulation and incremental-value tests.
- Customer assets are not pooled in personal brokerage accounts.
- Public marketing claims require evidence and a separate approval.

## 11. Immediate Work

1. Preserve the current DOIN campaign and paper observers unchanged.
2. Complete the OANDA MT5 24-hour read-only evidence and reload the expanded
   watchlist when operationally convenient.
3. Preserve OANDA MT5 demo as an independently monitored venue-reality stream;
   do not attempt MQL5 Signals or create a second MQL5 identity.
4. Use the active Spotware demo under the existing cTID; verify Copy catalogue
   access, then implement only the official
   read-only Open API preflight before custom-copy orders.
5. Materialize the owner-verified eToro Virtual facts into the structured
   checklist. Keep Darwinex Zero on hold until its recurring membership
   receives explicit approval.
6. Extend external social OLAP from the implemented neutral ledger; do not
   scrape UI data when an official export/API exists.
7. Feed measured copy/fee/flow gaps into a versioned simulator calibration
   packet, not the active chain.

## 12. Official Sources

- cTrader Open API: https://help.ctrader.com/open-api/
- cTrader Copy FAQ: https://help.ctrader.com/ctrader-copy/faq/
- cTrader Copy investing: https://help.ctrader.com/ctrader-copy/investing-in-strategies/
- MQL5 Signals rules: https://www.mql5.com/en/signals/rules
- MQL5 provider agreement: https://www.mql5.com/en/signals/terms/provider
- MT5 build 4150 release: https://www.mql5.com/en/forum/459335
- Darwinex Zero overview: https://www.darwinexzero.com/docs/what-is-darwinex-zero
- Darwinex Zero assets: https://www.darwinexzero.com/assets
- eToro CopyTrader: https://www.etoro.com/en-us/copytrader/
- HFM PAMM program: https://pamm.hfm.com/int/en/pamm-accounts/program
- HFM PAMM manager FAQ: https://pamm.hfm.com/int/en/fund-managers/fund-managers-faqs

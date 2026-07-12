# gym-fx Branch and Simulation-Engine Audit

Date: 2026-07-11
Repository: `/home/harveybc/Documents/GitHub/gym-fx`
Decision: retain current `master` as Backtrader parity control; run the canonical-engine bake-off before implementation

## 1. Remote Branch Evidence

The remote was fetched and pruned before comparison.

| Branch | Tip date | Tip | Simulation style | Decision |
| --- | --- | --- | --- | --- |
| `origin/master` | 2026-06-02 | `84def08` | Gymnasium wrapper backed by Backtrader/Cerebro | Retain as current base |
| `origin/dev2` | 2025-02-01 | `a70496d` | old custom `gym.Env` automation environments | Do not revive |
| `origin/dev` | 2025-01-04 | `67dc0b3` | old custom `gym.Env` automation environments | Do not revive |
| `origin/feature/tidying` | 2022-12-03 | `2cb33b0` | old custom Forex Gym environments | Historical only |

Current `master` contains the explicit rebuild commits:

- `c14290b`: env-only plugin architecture;
- `3d236a4`: rebuild on Backtrader with Gymnasium API;
- `189e77e` and later: SL/TP strategy plugins and execution behavior;
- `84def08`: OANDA calendar controls.

The current test suite passes `32/32` before any Phase 3 multi-asset changes.

## 2. What Is Reusable

- Backtrader and `Cerebro` execution substrate;
- Gymnasium-facing API pattern for learned agents;
- data-feed, broker, strategy, preprocessing, reward and metric plugin groups;
- bracket-order strategy plugins;
- pessimistic commission/slippage configuration;
- OANDA calendar and force-close controls;
- existing single-asset tests as compatibility evidence.

## 3. What Must Be Rebuilt

The current bridge is still single-asset:

- one `data0` feed;
- one action slot;
- one scalar position direction;
- one single-asset observation lifecycle;
- no virtual-cell aggregation or instrument netting;
- no heterogeneous multi-timeframe eligibility/staleness contract.

These limitations must not be patched into a custom accounting simulator.
Keep the current Backtrader harness as a parity control while NautilusTrader
and LEAN are evaluated under the canonical-engine bake-off.

## 4. Backtrader Control Runtime Design

The following is the Backtrader control/fallback design. It is no longer an
automatic decision to make Backtrader the canonical portfolio engine. See
`docs/work_plan/14_SIMULATION_ENGINE_SELECTION_2026_07_11.md`.

### Backtrader owns financial truth in Backtrader runs

A single `Cerebro` instance and shared Backtrader broker own:

- accepted/rejected orders;
- fills and fill prices;
- broker-net instrument positions;
- cash, value/equity, margin and commissions;
- bracket/protective-order lifecycle;
- trade closure and realized P&L.

No parallel custom ledger may independently recalculate those values and then
compete with Backtrader as account truth.

### Project code owns orchestration and evidence

New code owns:

- heterogeneous feed completion and causal decision eligibility;
- explicit age, stale and missingness metadata;
- `DecisionContext -> AssetIntent` policy calls;
- virtual-cell target aggregation and same-instrument netting before orders;
- translation to Backtrader order requests;
- broker callback capture into immutable order/trade/account facts;
- per-cell attribution derived from observed broker executions;
- venue/currency/cost capability profiles and reconciliation reports.

### Multi-asset Backtrader shape

```text
many timestamped feeds
        |
        v
MultiAssetCoordinatorStrategy (one Cerebro)
        |
        +--> eligible cell policies
        +--> asset intents
        +--> portfolio weights/risk budgets
        +--> instrument target netting
        |
        v
shared Backtrader broker + per-instrument CommInfo
        |
        +--> notify_order / notify_trade
        +--> broker cash/value/margin snapshots
        |
        v
immutable observed event facts + reconciliation
```

Backtrader's default broker is single-account and not a complete OANDA
multi-currency implementation. Home-currency conversion, instrument-specific
margin, financing, spread and contract multipliers must therefore be expressed
through tested Backtrader `CommInfoBase`/broker extensions and conversion feeds,
not through an unrelated second simulator.

## 5. Migration Strategy

1. Keep the current single-asset Backtrader adapter unchanged as a parity oracle.
2. Add a new isolated multi-asset coordinator and broker profile layer.
3. First reproduce one single-asset replay exactly.
4. Add two assets with one shared broker and hand-calculated costs/margin.
5. Add heterogeneous timeframes and stale/eligibility masks.
6. Add virtual cells and pre-broker netting.
7. Add multi-currency conversion and financing profiles.
8. Expand to the six-cell static portfolio before integrating learned allocation.

The old `dev`, `dev2`, and `feature/tidying` branches are not migration sources
for account logic. They may be consulted only for historical intent or fixtures.

## 6. Worktree Safety

The current `gym-fx` worktree already contains uncommitted Project 3 changes in
`app/env.py`, `app/bt_bridge.py`, `strategy_plugins/direct_atr_sltp.py`, and
associated tests. They were not reverted or overwritten during this audit.

An initial custom Phase 3 ledger drafted during the audit was removed before
acceptance because it duplicated Backtrader's accounting responsibility. No
remaining Phase 3 source files from that draft are present.

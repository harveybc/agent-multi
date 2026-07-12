# 07. Serving, LTS, and OANDA

## 1. Serving Boundary

`prediction_provider` is the runtime model gateway. It resolves promoted DOIN
evidence into locally loadable artifacts and returns validated predictions or
intents. LTS remains the only component allowed to transform those signals into
customer-specific broker orders.

## 2. Model Registry and Release Channels

The provider tracks signed `DeploymentManifest` objects and supports:

- `pinned`: exact release forever until user changes it;
- `stable`: latest approved release;
- `adaptive`: latest weekly release passing automatic gates;
- `experimental`: practice/shadow only;
- `custom`: explicit compatible component bundle.

For each channel it stores current, previous/rollback, preload status, health,
artifact hashes and validity.

## 3. Promotion Controller

Promotion is domain-specific and should be implemented as an `agent-multi`
command/service, not as blockchain consensus logic.

It:

1. queries accepted/Pareto DOIN candidates;
2. freezes training cutoff and component recipe;
3. trains/fine-tunes deployable artifacts;
4. runs compatibility, regression and release validation;
5. evaluates the complete stack;
6. signs/publishes the deployment manifest;
7. requests provider preload;
8. activates only at the declared release/rebalance boundary.

Human approval is required for live-channel major releases. Adaptive practice
channels may use automatic gates.

## 4. Provider Output Modes

### 4.1 Forecast mode

Returns `PredictionBundle` to a heuristic lifecycle policy. Useful for the
legacy long/short entry and early-close strategies.

### 4.2 Asset intent mode

Runs a learned actor-critic or complete cell stack and returns `AssetIntent`.

### 4.3 Portfolio mode

Runs weekly allocator inference and returns `PortfolioIntent` plus component
identities.

### 4.4 Batch consistency

All assets in one portfolio inference use the same declared `as_of`, release,
data-cutoff policy and compatibility set. Partial results are marked and LTS
applies configured fallback.

## 5. Local Versus Decentralized Inference

DOIN already supports inference tasks. Two modes are permitted:

- local provider inference from a hash-verified cached artifact: default for
  latency-sensitive market and early-close decisions;
- decentralized DOIN inference: optional for weekly allocation, redundancy,
  paid inference or non-urgent tasks.

No urgent stop/close decision depends on blockchain block time or remote quorum.
DOIN supplies discovery, evidence, attribution and optional inference; provider
cache supplies deterministic low-latency trading behavior.

## 6. LTS Customer Configuration

Example:

```json
{
  "model_source": {
    "provider": "doin",
    "domain": "trading-stack-v1",
    "channel": "stable",
    "update_policy": "weekly",
    "risk_profile": "balanced",
    "max_model_age_hours": 192,
    "fallback": "previous_verified"
  },
  "portfolio_policy": {
    "max_gross_exposure": 0.50,
    "max_asset_weight": 0.25,
    "max_correlated_group_weight": 0.40,
    "max_drawdown": 0.15
  }
}
```

Client risk settings constrain a generic model recommendation. They do not
mutate the immutable model or require per-user retraining.

## 7. LTS Runtime Flow

1. Select active deployment channel/version.
2. Fetch/validate portfolio and asset signal bundle.
3. Reject stale, incompatible or incomplete required signals.
4. Apply customer risk profile and broker capability snapshot.
5. aggregate virtual cell sleeves by broker instrument;
6. compute idempotent target-position deltas;
7. preflight margin, concentration, precision and market state;
8. send orders;
9. consume broker transaction/order state;
10. reconcile positions and update attribution/audit;
11. apply fallback or kill switch on divergence.

## 8. Heuristic Lifecycle Execution

LTS can load the same `heuristic-strategy` policy package used in backtests.
It receives `PredictionBundle` from the provider, constructs `DecisionContext`,
and obtains `AssetIntent`.

The initial portfolio permits one logical position per cell and many concurrent
assets. Multiple cells for one instrument are virtual sleeves; LTS nets them
before OANDA while retaining cell-level attribution.

## 9. OANDA Capability

OANDA v20 supports multiple instruments, orders, trades/positions, dependent
SL/TP orders, account-level margin state, and multi-instrument pricing. Actual
tradeable instruments depend on the account's regulatory division and account
configuration.

At account onboarding and periodically, LTS must fetch instruments and create a
`BrokerCapabilitySnapshot` with:

- canonical-to-OANDA symbol mapping;
- instrument type;
- price and unit precision;
- minimum trade size;
- maximum order and position units;
- margin rate and account margin state;
- financing/commission;
- stop and guaranteed-stop restrictions;
- hedging/netting and position-fill behavior;
- current market/tradeability status.

Research symbols such as `SOLUSDT` or perpetual futures are not assumed to be
OANDA instruments. Unsupported assets are excluded from that live account or
routed through a future broker/exchange plugin.

## 10. Broker Abstraction

LTS broker plugins consume `OrderIntent` and return `ExecutionReport`.
Required operations:

- capability discovery;
- account snapshot;
- prices/streaming;
- target/delta order placement;
- modify protection;
- partial/full close;
- open orders/trades/positions;
- transaction stream or incremental reconciliation;
- idempotency and request correlation.

The current OANDA plugin is a prototype. Production readiness requires dynamic
precision, capabilities, client IDs, transaction reconciliation, partial fills,
home-currency conversion, account-level risk gates and robust retry semantics.

## 11. Safety and Fallback

- Separate practice and live credentials/configs.
- Default all new releases and customers to practice/shadow.
- Never log or persist tokens in config/OLAP/chain.
- Reject stale signals and unknown schema versions.
- Keep previous verified release loaded for immediate rollback.
- Make order planning idempotent across restart/retry.
- Reconcile before issuing replacement orders.
- Enforce max loss, drawdown, margin, gross/net/correlation and order-count caps.
- Define provider-down policy: hold, reduce or flatten by portfolio profile.
- Define model-divergence and broker-divergence kill switches.
- Require explicit human activation for real capital.

## 12. Usage and Incentives

LTS may optionally report privacy-preserving usage attribution to DOIN:

- deployment/component hashes used;
- inference count and latency class;
- release channel;
- aggregate non-customer-identifying reliability metrics.

Customer identity, balance, exact positions and broker credentials are not
public/on-chain telemetry. Economic rewards for optimization/evaluation/inference
use existing DOIN mechanisms or explicit future extensions, not hidden LTS
side effects.

## 13. Simulation-to-Practice Sequence

1. Direct `agent-multi` deterministic replay.
2. `prediction_provider` historical replay.
3. LTS plus simulation broker.
4. LTS read-only OANDA shadow.
5. minimal-size OANDA practice orders.
6. full practice weekly portfolio operation.
7. frozen release review.
8. conservative live pilot after explicit approval.

Each step compares signals, intended deltas, fills, costs and positions against
the prior layer.

## 14. Acceptance Criteria

- Provider output matches direct model inference for a frozen fixture.
- Channel switch is atomic at release boundary and rollback succeeds.
- LTS never downloads or activates an unverified artifact hash.
- Customer overlays cannot exceed hard portfolio/broker limits.
- Multiple virtual cells net correctly into one OANDA instrument target.
- Practice orders reconcile with no unexplained position divergence.
- Provider/network restart cannot duplicate an order.
- Unsupported instruments fail closed before order creation.

## 15. OANDA References

- https://developer.oanda.com/rest-live-v20/introduction/
- https://developer.oanda.com/rest-live-v20/account-ep/
- https://developer.oanda.com/rest-live-v20/pricing-ep/
- https://developer.oanda.com/rest-live-v20/order-ep/
- https://developer.oanda.com/rest-live-v20/trade-ep/

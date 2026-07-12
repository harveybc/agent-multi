# 04. Models, Policies, and Training

## 1. Component Hierarchy

The system separates statistical questions that require different targets and
validation:

1. market context representation;
2. asset opportunity/rush detection;
3. asset directional/exposure control;
4. trade lifecycle management;
5. risk geometry and sizing hint;
6. portfolio allocation;
7. stack composition.

The first implementation avoids an opaque end-to-end monolith. Restricted joint
refinement is permitted after independent baselines are stable.

## 2. Variable-Length Context Encoder

### 2.1 Inputs

- typed technical/statistical events;
- volatility, trend, liquidity and unsupervised regime events;
- asset identity, timeframe, venue, age and missingness;
- recent execution and policy-outcome context;
- cross-asset relative momentum, correlation, breadth and dispersion;
- scheduled/observed macro or fundamental events available by `as_of`;
- seasonality and market sessions;
- crypto market-wide and on-chain events where causally available.

### 2.2 Architecture ladder

1. no context/local-only control;
2. deterministic engineered context summary;
3. existing attention representation;
4. trainable masked event-token encoder;
5. deeper transformer only after lower-cost variants justify it.

The encoder produces fixed embeddings, masks, confidence and diagnostics.
Vocabulary, normalization and auxiliary fitting occur within each weekly train
cutoff.

## 3. Learned Asset Policy

The initial learned family is Project 3 SAC actor-critic. The plugin contract
also permits PPO, DQN and future models.

Inputs:

- local feature window;
- context embedding/masks;
- current logical position and protection;
- cell/account risk availability;
- current execution conditions;
- rush state;
- closure horizon.

Outputs:

- target directional exposure;
- action confidence;
- local `rel_volume`/exposure factor;
- risk-geometry parameters or references;
- urgency/validity;
- explicit no-trade.

The model returns `AssetIntent`, not broker units.

## 4. Heuristic Trade Lifecycle Policies

`heuristic-strategy` becomes a reusable policy package. The legacy long/short
prediction strategy is preserved as the first refactored plugin:

```text
prediction_entry_exit_v1
  entry: long-horizon prediction exceeds threshold
  direction: choose long/short expected path
  protection: derive SL/TP from forecast/risk geometry
  manage: combine short- and long-horizon predictions
  early close: predicted SL-before-TP or invalidated thesis
  close: TP, SL, weekend, risk gate, or stale model
```

Other plugins can specialize by asset, regime, rush state or strategy family.
They all implement the same pure decision contract.

### 4.1 Prediction inversion

The policy does not call `prediction_provider` itself. A caller obtains a
`PredictionBundle` from CSV, ideal oracle, direct model, API, or DOIN inference
and supplies it in `DecisionContext`. This makes prediction source an adapter
and keeps strategy behavior identical across backtest and live use.

### 4.2 Optimization

The repository's local DEAP optimizer remains a regression/research tool. It is
not nested inside DOIN's Level 2 optimization. DOIN evolves the policy's typed
parameters through the `agent-multi` evaluator.

## 5. Rush Detector

Rush detection predicts opportunity, not action. Per asset/horizon outputs:

- favorable-rush probability;
- expected direction, intensity and duration;
- adverse/hostile regime probability;
- calibrated confidence.

### 5.1 Labels

Labels use future outcomes only as targets and must never enter inputs. Define
rush relative to the asset's rolling training distribution using excess RAP,
return, activity, liquidity and persistence. Store label version and thresholds.

### 5.2 Evaluation

- precision-recall and average precision;
- Brier score and calibration curve;
- lead time and duration error;
- false-positive cost;
- incremental policy/portfolio utility;
- stability across episodes, assets and seeds.

Accuracy alone is insufficient for a rare-event detector.

### 5.3 Initial use

Start with an exposure gate over the base policy. Train a specialized rush
policy only if the gate shows reproducible validation utility.

## 6. Risk Geometry

Keep independently configurable:

- `rel_volume` or exposure factor;
- legacy notional versus risk-at-stop sizing;
- fixed/ATR/margin-aware SL/TP;
- risk penalty lambda;
- maximum adverse excursion and drawdown controls;
- leverage and margin safety cap;
- trailing/modification policy;
- weekend flattening.

The first comparison preserves the Project 3 geometry at `rel_volume=0.05` or
the exact candidate profile. Risk-aware geometry is then optimized as its own
stage so model improvements are not confused with exposure changes.

## 7. Portfolio Allocator

Runs at a configured rebalance boundary and outputs cell weights, cash and risk
budgets.

Inputs:

- causal state and recent history for each cell;
- expected return/RAP/risk and uncertainty;
- rush probabilities;
- downside covariance and correlation regimes;
- liquidity, costs, turnover and margin;
- horizon/asset-class groups;
- prior weights;
- component freshness and availability.

Baselines precede learning: equal weight, inverse volatility, minimum variance,
minimum semivariance and capped RAP rank.

The first eligible portfolio targets at least three short-horizon and three
medium/long-horizon cells. Symbols do not count as diversification when they
share one dominant risk factor.

## 8. Weekly Walk-Forward Protocol

For each validation/test week:

1. freeze cutoff before the target week;
2. construct train data using only prior available information;
3. fit every preprocessor/encoder/calibrator on train only;
4. train or fine-tune asset components;
5. construct portfolio intent using prior state only;
6. simulate target week once;
7. close/roll according to declared rule;
8. persist order, asset-week and portfolio-week facts;
9. advance cutoff.

Validation and test each target a complete chronological year. Coverage gates
are explicit and partial evidence cannot be promoted as annual performance.

## 9. Optimization Levels

### 9.1 L1 candidate training

Gradient optimization and candidate early stopping. Monitors train/validation
only. For deterministic heuristic policies there is no gradient L1; their
predictor dependencies may still have L1 training.

### 9.2 L2 DOIN/DEAP optimization

Evolves typed config patches. L2 patience is independent from L1 callbacks and
uses validation fitness.

### 9.3 L3 meta-optimization

Future use of DOIN OLAP to predict promising parameter regions. L3 proposes
candidates; it does not certify them and never accesses protected test results.

## 10. Objective Functions

### 10.1 L1

Stepwise reward can include:

- realized/unrealized return;
- adverse excursion/equity drawdown;
- spread, commission, financing and turnover;
- invalid action and margin penalties;
- optional behavior-pretraining loss.

Weights are config fields and raw components are logged.

### 10.2 Asset L2

Full validation-year score combines:

- weekly/annual return and RAP;
- drawdown and expected shortfall;
- active weeks and minimum trades;
- costs and turnover;
- seed/subperiod instability;
- no-trade and trivial-strategy improvement.

### 10.3 Portfolio L2

Combines validation portfolio RAP/return with downside tail, drawdown,
concentration, turnover, margin and stability penalties.

Every scalar fitness is accompanied by its complete raw metric vector.

## 11. Pareto Releases

There is no universal best stack. Maintain feasible champions for conservative,
balanced and aggressive profiles across:

- return;
- RAP;
- drawdown/expected shortfall;
- cost/turnover;
- stability;
- concentration;
- rush capture.

LTS maps customer risk profiles to promoted Pareto releases, then applies hard
customer-level constraints.

## 12. Protected Test and Live Evidence

- Test is excluded from selection, optimization, migration, patience and weekly
  allocation.
- Frozen test is opened for major release assessment only.
- Shadow/live outcomes are a separate evidence class.
- Mature live observations can join future training data only after their
  chronological period and embargo are complete.
- Historical scores are never rewritten using later knowledge.

## 13. Acceptance Criteria

- Every component has an independently measurable baseline and contract.
- Weekly fitting is train-only and reproducible.
- Heuristic policy decisions match legacy behavior on a frozen fixture.
- SAC single-cell adapter reproduces a selected Project 3 candidate.
- Rush detector is calibrated and improves downstream validation utility.
- Portfolio allocator beats declared baselines after costs without hidden
  concentration.
- L1 and L2 stopping states cannot overwrite or consume one another.

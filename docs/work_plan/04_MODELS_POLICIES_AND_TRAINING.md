# 04. Models, Policies, and Training

## 1. Component Hierarchy

The system separates statistical questions that require different targets and
validation:

1. market context representation;
2. asset opportunity/rush detection;
3. asset directional/exposure control;
4. execution-state estimation;
5. entry/exit order and trade lifecycle management;
6. risk geometry and sizing hint;
7. portfolio allocation;
8. stack composition.

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

- probability that a favorable rush starts within one day and one week;
- continuation and termination probabilities;
- expected direction, intensity and duration;
- adverse/hostile regime probability;
- volatility, jump and liquidity-stress probabilities;
- calibrated confidence.

### 5.1 Labels

Labels use future outcomes only as targets and must never enter inputs. Define
rush relative to the asset's rolling training distribution using directional
efficiency, realized volatility, maximum favorable/adverse excursion, jump
frequency, liquidity, achievable after-cost return and persistence. Store the
label version, horizon, thresholds and training-distribution cutoff.

The detector is a probabilistic multi-horizon hazard model, not a hard regime
switch. The portfolio allocator and asset policy consume probabilities and
uncertainty. They do not replace an asset model merely because one class has
the largest probability.

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

## 6. Execution State and Adaptive Order Policy

### 6.1 Separation from alpha

The asset policy answers whether and how strongly the system wants exposure.
The execution policy answers how to reach or leave that exposure. We do not
train one independent alpha model for each order type. That would fragment the
sample, confound prediction with fill mechanics and prevent direct comparison
of alternative orders for the same intent.

The deterministic hierarchy in document 19 remains the first baseline. A
learned policy is eligible only after it beats market-only and deterministic
router controls under identical signal streams and replay conditions.

### 6.2 Shared encoder and specialized heads

The execution policy uses one causal state encoder with two heads:

- entry head: `WAIT`, `MARKET`, `LIMIT`, `STOP` or `MARKET_IF_TOUCHED`, plus
  offset, volume, time-in-force, TTL, fallback and initial protection;
- exit head: `HOLD`, `MARKET_CLOSE`, `LIMIT_CLOSE`, `CANCEL_REPLACE`,
  protection modification, trailing stop or forced close.

Both heads observe alpha direction/quantiles/confidence/decay, rush and event
probabilities, volatility, spread, liquidity, order flow, session, position
age, unrealized P&L, remaining quantity and risk budget. Hard risk and broker
capability checks remain deterministic overrides outside the learned policy.

Negative-transfer tests compare the shared encoder against fully separate entry
and exit models. A split is accepted only when comparable walk-forward evidence
justifies the extra artifacts and smaller effective samples.

### 6.3 Auxiliary execution models

Before optimizing the policy, train and calibrate:

- time-to-fill/survival distribution by order type, offset and size;
- post-fill adverse-selection distribution;
- short-horizon price-path quantiles and alpha decay;
- spread, liquidity and jump/event hazard;
- broker/simulator slippage and rejection residuals.

These models expose estimates and uncertainty; they do not submit orders.
Family-level models may pool normalized data across similar FX, crypto or
equity cells. Microstructure-incompatible asset classes are not pooled merely
to increase row count.

### 6.4 Unified action utility

Alternative actions for one intent are compared in common return/risk units:

```text
utility(action | state) =
    P(fill) * (
        expected alpha after fill
      - fees
      - slippage/impact
      - expected adverse selection
    )
  - P(no fill) * missed-opportunity cost
  - tail-risk penalty
```

The stored components remain visible even when DOIN receives one scalar
fitness. Market orders trade fill certainty for cost; passive orders trade
price improvement for non-fill/adverse-selection risk; stop/MIT actions make
entry conditional on a future trigger. Protective stops are risk controls and
are not confused with entry-stop actions.

### 6.5 Training examples and fidelity

At each historical decision point, replay the same immutable asset intent
against a bounded action grid. Persist the realized fill, time-to-fill,
implementation shortfall, adverse post-fill movement, missed opportunity and
tail outcome. This counterfactual action table is simulator-derived evidence,
not a claim that unobserved historical orders actually existed.

Bars can support conservative controls, but learned passive placement requires
timestamped bid/ask data and preferably L1/L2 depth as declared in document 03.
No purchased vendor regime label is required. Purchase decisions, if any,
target raw quote/book data or point-in-time calendar consensus/actual vintages.

### 6.6 Scheduled events and causal evidence

Event inputs are phase-specific:

- pre-release: schedule, event family, importance, time to release, consensus
  dispersion and prior-reaction distribution;
- release: actual value and normalized surprise, only after publication;
- post-release: elapsed time, price/order-flow response and propagation across
  rates or related assets.

Actual results cannot enter a pre-release observation. Event studies, local
projections and heterogeneous treatment-effect estimates may identify robust
event mechanisms and policy priors. They support prediction and execution but
do not replace chronological walk-forward evaluation or establish causality
from ordinary market association alone.

### 6.7 Optimization order

1. Freeze the asset alpha champion and diverse elites.
2. Materialize execution-fidelity data and immutable replay scenarios.
3. Train/calibrate auxiliary execution models with L1.
4. Train the entry/exit policy locally against deterministic controls.
5. Use DOIN L2 for feature masks, order family, offsets, TTL, fallback, sizing
   hints and model hyperparameters under easy, nominal and stress costs.
6. Run bounded joint refinement only for alpha parameters shown to interact
   materially with execution.
7. Freeze a complete cell release before portfolio optimization.

## 7. Risk Geometry

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

## 8. Portfolio Allocator

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

## 9. Weekly Walk-Forward Protocol

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

## 10. Optimization Levels

### 10.1 L1 candidate training

Gradient optimization and candidate early stopping. Monitors train/validation
only. For deterministic heuristic policies there is no gradient L1; their
predictor dependencies may still have L1 training.

For decision-bearing per-asset research, "train/validation" is a nested
chronological contract rather than one repeatedly reused year:

- fit data receives gradient updates;
- a sufficiently long train-monitor interval and a disjoint inner-validation
  interval control L1 patience through `paired_generalization_weekly_v1`;
- an outer-validation interval is invisible to L1 and controls L2 selection;
- the protected test is invisible to both levels until release; and
- each scored interval receives a non-trading causal context prefix so warm-up
  cannot silently shorten the declared period.

The full-fidelity per-asset protocol uses:

- `max_epochs: 2000` as a safety ceiling;
- `l1_patience: 60`;
- a 40-epoch minimum training floor before non-improvement may consume
  patience;
- best-checkpoint restoration;
- a scale-explicit `l1_min_delta` in fractional weekly RAP units;
- an epoch step count derived from the valid training transitions rather than
  an unexplained constant shared by incompatible timeframes; and
- separate improvement and activity-ineligible patience states.

The paired stopping scalar uses the arithmetic mean of the same common-scale
robust weekly utility on train-monitor and inner validation, minus an explicit
generalization-gap penalty. Eligibility remains lexicographic, and all raw
return/risk/activity vectors remain first-class evidence. Opaque lexicographic
rank encodings are never averaged. Validation-only stopping is retained only
for historical configurations and is not permitted in the new ETH decision
domain.

A cheaper run may be labeled as a smoke test or ranking-only fidelity. It
cannot become a promoted per-asset champion until the same decoded candidate
has been evaluated under the full-fidelity protocol. L1 budget controls are
part of the evaluator contract, not genes that may gain fitness by terminating
training early.

### 10.2 L2 DOIN/DEAP optimization

Evolves typed config patches. L2 patience is independent from L1 callbacks and
uses a paired inner/outer chronological objective. Candidate L1 training and
checkpoint selection cannot observe outer validation. L2 receives the frozen
inner and outer evidence packet, applies split-specific eligibility, and stops
generations only after a declared minimum-generation floor.

L2 stages may activate/freeze different gene groups. Decision-bearing staged
domains additionally declare stage-local crossover, mutation, numeric
perturbation, categorical-change, diversity and patience contracts. Current
global-only mutation behavior is historical compatibility, not evidence that a
NEAT-like maturation schedule has been implemented.

Difficulty curricula at L1 and L2 are distinct and composable. They are tested
in isolation and then, only when triggered, in a bounded 2x2 interaction as
defined in document 38. An easy-stage L2 score is invalidated at transition and
cannot compete with or produce a normal-realistic champion.

### 10.3 L3 meta-optimization

Future use of DOIN OLAP to predict promising parameter regions. L3 proposes
candidates; it does not certify them and never accesses protected test results.

## 11. Objective Functions

### 11.1 L1

Stepwise reward can include:

- realized/unrealized return;
- adverse excursion/equity drawdown;
- spread, commission, financing and turnover;
- invalid action and margin penalties;
- optional behavior-pretraining loss.

Weights are config fields and raw components are logged.

### 11.2 Asset L2

Full validation-year score combines:

- weekly/annual return and RAP;
- drawdown and expected shortfall;
- active weeks and minimum trades;
- costs and turnover;
- seed/subperiod instability;
- no-trade and trivial-strategy improvement.

### 11.3 Portfolio L2

Combines validation portfolio RAP/return with downside tail, drawdown,
concentration, turnover, margin and stability penalties.

Every scalar fitness is accompanied by its complete raw metric vector.

## 12. Pareto Releases

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

## 13. Protected Test and Live Evidence

- Test is excluded from selection, optimization, migration, patience and weekly
  allocation.
- Frozen test is opened for major release assessment only.
- Shadow/live outcomes are a separate evidence class.
- Mature live observations can join future training data only after their
  chronological period and embargo are complete.
- Historical scores are never rewritten using later knowledge.

## 14. Acceptance Criteria

- Every component has an independently measurable baseline and contract.
- Weekly fitting is train-only and reproducible.
- Heuristic policy decisions match legacy behavior on a frozen fixture.
- SAC single-cell adapter reproduces a selected Project 3 candidate.
- Rush detector is calibrated and improves downstream validation utility.
- Learned execution beats market-only and deterministic-router controls on the
  same signal stream after costs, including non-fill and adverse-selection
  outcomes.
- Entry/exit decisions are reproducible from point-in-time inputs, and no
  order-policy claim exceeds the declared replay-data fidelity.
- Portfolio allocator beats declared baselines after costs without hidden
  concentration.
- L1 and L2 stopping states cannot overwrite or consume one another.

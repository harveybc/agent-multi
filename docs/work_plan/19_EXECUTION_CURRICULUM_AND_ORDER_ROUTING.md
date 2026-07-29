# 19. Execution Curriculum, State Models, and Adaptive Order Policy

Status: deterministic baseline and curriculum transition verified/deployed;
learned execution-policy contract specified; successor campaign queued behind
the replicated stage barrier
Decision date: 2026-07-28

## 1. Objective

Improve execution realism without discarding the asset-policy learning already
completed. The plan separates three orthogonal capabilities:

1. a visible, versioned cost curriculum for policy fine-tuning; and
2. an account-independent deterministic router that selects market, limit, or
   stop entry as the mandatory control; and
3. a multi-timescale learned execution layer that estimates fill, adverse
   selection, short paths and event hazard before selecting entry/exit order
   actions.

The current `usdcad-4h-full-genome-sac-shared-v1` worker configuration remains
byte-for-byte unchanged. Its useful data/observation and model/training stages
finish first. When every worker enters stage 3 with the same shared generation
and population, the replicated supervisors freeze one agreed champion hash,
archive it with diverse alternatives, verify every local
worker has stopped, and only then start a new domain. No running blockchain,
candidate pool, optimizer genome, seed, or fitness definition is modified in
place.

## 2. Non-Negotiable Boundaries

- Cost scenarios are evaluator conditions, never optimizer genes.
- The policy observes the active cost conditions during training.
- Validation scenarios are immutable and deterministic.
- Validation and test data never select curriculum difficulty.
- The execution router does not size an account or submit broker orders.
- The asset alpha policy does not select an order type. It emits target
  exposure, confidence, urgency, validity and alpha-decay evidence.
- Learned execution decisions cannot claim greater fidelity than their
  bar/quote/L1/L2/live-calibration source permits.
- Actual calendar-event results and surprises cannot enter a pre-release
  observation.
- `gym-fx` owns simulated fills; LTS owns live account sizing and submission.
- `agent-multi` owns policy learning, routing research, robust aggregation, and
  artifacts.
- DOIN coordinates candidates but does not acquire trading-domain logic.

## 3. Cost Curriculum

The fixed three-phase schedule is:

| Phase | Purpose | Cost range |
| --- | --- | --- |
| `easy_nonzero` | learn directional and lifecycle behavior without free fills | low, strictly positive |
| `nominal_randomized` | become invariant to ordinary execution variation | deterministic seeded sample from nominal scenarios |
| `stress` | preserve behavior under adverse but plausible costs | high fixed/randomized stress scenarios |

Every phase specifies its progress interval and allowed scenarios. Scenario
selection is a pure function of:

```text
(curriculum contract hash, training seed, normalized training progress, episode)
```

The applied observation vector is ordered and normalized:

```text
commission_per_side
full_spread
slippage_per_side
financing_enabled
cost_phase_progress
```

Changing the curriculum or observation ordering changes the optimizer resume
contract and requires a new DOIN domain/genesis.

The first follow-up uses an explicit 100-epoch curriculum horizon even though
the hard L1 cap remains 2,000 epochs. Stress begins before the earliest
possible `l1_patience_start_epoch + l1_patience` stop and remains active after
the horizon. This prevents a normally early-stopped candidate from seeing only
easy or nominal costs.

## 4. Immutable Validation Suite

Each checkpoint is evaluated under the same ordered cost scenarios. The suite
must include:

- easy nonzero control;
- nominal low;
- nominal high;
- stress;
- the Project 3 pessimistic profile.

For every scenario, the evaluator stores at least:

- mean weekly return;
- annualized return;
- mean weekly RAP;
- annual RAP;
- mean and maximum weekly drawdown;
- trade count, turnover, cost drag, fill ratio, and expiration count;
- config, scenario, data, code, and model artifact hashes.

All return and RAP values are fractions. Display layers label and convert them
to percentages; storage and fitness do not.

## 5. Robust Fitness

Let `r_i` be validation mean weekly RAP under scenario `i`. The scalar L2
fitness remains in fraction-per-week units:

```text
mean_rap        = mean(r_i)
downside_gap    = max(0, mean_rap - CVaR_alpha(r_i))
dispersion      = population_stddev(r_i)

robust_fitness =
    mean_rap
  - downside_weight * downside_gap
  - dispersion_weight * dispersion
```

The result is accompanied by the complete metric vector:

- `mean_weekly_return`;
- `annualized_return`;
- `mean_weekly_rap`;
- `annual_rap`;
- `worst_scenario_weekly_rap`;
- `lower_tail_cvar_weekly_rap`;
- `scenario_weekly_rap_dispersion`;
- `robust_weekly_rap_fitness`.

Missing or non-finite scenario metrics reject the candidate. There is no silent
zero substitution.

## 6. Deterministic Router Baseline

The asset policy emits signed target exposure, confidence, urgency, risk
geometry, and validity. The router combines those values with causal execution
context:

- reference bid/ask or mid price;
- current spread;
- ATR/volatility;
- breakout score;
- market session and venue capability.

Initial deterministic hierarchy:

1. hold when exposure is inside the deadband;
2. market when urgency is high and spread is acceptable;
3. stop when breakout evidence and decision strength exceed the threshold;
4. limit otherwise for passive price improvement.

Limit and stop prices use the maximum of a small tick guard, spread multiple,
and ATR multiple. Pending entries have an explicit bar TTL and either cancel or
market fallback. Long and short paths are symmetric.

Optimizable router parameters:

- exposure deadband;
- market urgency and maximum-spread thresholds;
- breakout threshold;
- limit/stop spread and ATR offsets;
- pending-order TTL;
- unfilled fallback.

Order type is a router result. Transaction costs and validation scenarios are
not genes.

This router must remain runnable and versioned after the learned policy exists.
It is the principal ablation for proving that model complexity adds execution
utility rather than merely changing trade frequency.

## 7. Multi-Timescale State Contract

Execution combines three clocks without conflating them:

| Clock | Typical horizon | Role |
| --- | --- | --- |
| portfolio/rush | one day to several weeks | opportunity onset, continuation, allocation context |
| alpha/lifecycle | native asset bars to days | desired direction, exposure, confidence and decay |
| execution | ticks to minutes | spread, liquidity, fill, short path, trigger and cancellation decisions |

The execution observation includes:

- alpha direction, quantiles, confidence, urgency and expected decay;
- rush onset/continuation/termination and hostile-regime probabilities;
- volatility, trend/mean-reversion, spread, liquidity and jump state;
- bid/ask or order-book context, order flow and feature age/missingness;
- session, market closure and broker capability;
- position side, age, P&L, remaining quantity, SL/TP distance and risk budget;
- scheduled-event phase, importance, time to release and point-in-time surprise
  only when published.

A weekly regime is context, not an order. The router/policy recomputes from the
latest causally complete execution state whenever an entry, replacement,
protection update or exit is eligible.

## 8. Auxiliary Execution Models

The learned policy is preceded by independently measurable auxiliaries:

1. time-to-fill survival distribution by type, offset, size and TTL;
2. post-fill adverse-selection distribution;
3. short-horizon path quantiles and alpha-decay estimate;
4. spread, liquidity and jump/event hazard;
5. broker-versus-simulator fill, slippage, rejection and latency residuals.

Every output contains uncertainty, cutoff, fidelity, model/artifact identity
and calibration metrics. Similar assets may share a normalized family model
when their execution microstructure is compatible. FX, centralized crypto and
equity order books are not pooled by default.

## 9. Shared Entry/Exit Policy

One causal encoder feeds two specialized heads:

- entry: `WAIT`, `MARKET`, `LIMIT`, `STOP` or `MARKET_IF_TOUCHED`, then offset,
  size hint, time-in-force, TTL, fallback and initial protection;
- exit: `HOLD`, `MARKET_CLOSE`, `LIMIT_CLOSE`, `CANCEL_REPLACE`, protection
  modification, trailing stop or force-close request.

Risk, account and broker capability constraints remain deterministic overrides.
Separate full models are introduced only if a controlled negative-transfer
test beats the shared encoder across seeds/subperiods after accounting for
sample fragmentation and artifact complexity.

Alternative actions use a common objective:

```text
execution_utility =
    P(fill) * (
        expected alpha after fill
      - fees
      - slippage/impact
      - expected adverse selection
    )
  - P(no fill) * missed-opportunity cost
  - tail-risk penalty
```

The raw components are persisted. A market order typically buys certainty at a
cost; a passive order accepts non-fill and adverse-selection risk; stop/MIT
makes entry conditional on a future trigger. Entry stops and protective
stop-loss orders remain distinct actions and metrics.

## 10. Execution Training Data and Event Causality

At each immutable asset-intent timestamp, the simulator evaluates a bounded
action grid under the same causal state. It records fill, time-to-fill,
implementation shortfall, price improvement, adverse movement, expiration,
missed opportunity and tail outcome. These are simulated counterfactual action
outcomes, not fabricated historical orders.

Required data follow the fidelity ladder in document 03:

- OHLCV for conservative market/protection controls;
- timestamped bid/ask for spread-aware stop/MIT and coarse limit fills;
- L1/L2 where queue/size-sensitive passive execution is claimed;
- OANDA practice/live reports for broker-specific calibration.

We construct rush/regime targets from future paths inside the training
pipeline. We do not buy opaque regime labels. New purchases are limited to a
demonstrated gap in raw point-in-time quote/book data or calendar
expectation/actual vintages.

Scheduled events use three availability phases:

```text
pre-release: schedule + family + importance + consensus dispersion
release:     actual + normalized surprise, after publication only
post-release: elapsed time + market/order-flow propagation
```

Event studies, local projections and heterogeneous effect models may create
features and policy priors. Association is not relabeled as intervention or
counterfactual evidence, and causal estimates do not replace chronological
walk-forward utility.

## 11. Optimization Sequence

The sequence minimizes repeated model training:

1. finish `data_observation` and `model_training` in the active zero-cost
   campaign, stopping at the synchronized transition into stage 3;
2. load its exact champion weights and resolved genome;
3. fine-tune the asset policy under the cost curriculum;
4. select with immutable robust validation fitness;
5. freeze the robust asset-policy artifact;
6. pass the execution-fidelity gate and train/calibrate the auxiliary models;
7. optimize the deterministic execution router against frozen policy actions;
8. train the shared entry/exit policy locally and compare it with market-only
   and deterministic-router controls;
9. optimize the learned execution policy with DOIN over allowed feature masks,
   order family, offsets, TTL, fallback, size hints and model parameters;
10. run one bounded joint refinement over only the asset-policy parameters that
   materially interact with routing;
11. package model, router/policy, auxiliaries, preprocessing, feature, cost and
   metric contracts as
   one cell release;
12. use frozen cell releases in the portfolio optimizer.

The joint stage is skipped when the router-only stage provides no validation
improvement or when it increases tail instability.

## 12. Artifact Contract

Each accepted cell release contains:

- trained SB3 `.zip` policy and SHA-256;
- encoded and decoded winning genome;
- router config and hash;
- execution encoder, entry/exit heads and exact observation/action manifest;
- fill-time, adverse-selection, path and event-hazard auxiliary artifacts;
- execution data-fidelity and point-in-time event manifests;
- cost curriculum and validation-suite hashes;
- resolved canonical experiment config;
- ordered feature/preprocessing contract and hashes;
- complete per-scenario metric vector and robust aggregate;
- source DOIN domain, genesis, block, peer, and candidate identity;
- inference smoke evidence.

The release is usable without DOIN. DOIN distributes and improves it; it is not
required for local inference.

## 13. Campaign Transition Safety

The successor is present in an append-only lifecycle plan but cannot start
until all of these conditions hold:

1. every worker reports stage 3, the same current generation and pool
   fingerprint, bootstrap lineage, component versions, and champion artifact
   hash;
2. that agreement remains stable for the configured barrier interval;
3. every supervisor archives exactly the frozen hash, plus up to five
   deterministic strong/diverse alternatives;
4. model bytes, declared bytes and SHA-256 all match;
5. every local worker process and API is verified stopped;
6. every supervisor acknowledges the same boundary evidence and champion;
7. the launchable curriculum config is generated independently on every host
   from its local handoff artifacts;
8. startup preflight proves identical config semantics, dataset, seed,
   population, code versions and worker order before creating the new chain.

Terminal chain tips may differ at the boundary because DOIN can retain
equal-generation local forks. That is accepted only when the frozen champion,
generation pool, initial lineage and complete boundary evidence hash are
identical. Normal campaign completion still requires one tip or one verified
finalized anchor.

The disabled template is:

```text
examples/config/phase_1_asset_policy/optimization/
phase_1_asset_policy_usdcad_4h_execution_curriculum_template_v1.json
```

The launchable config must be generated with:

```text
examples/scripts/materialize_execution_curriculum_followup.py
```

Without both the archived champion policy and decoded optimization-parameter
files, the materializer fails closed. It hashes both files, seeds the bounded
follow-up genome from decoded categorical/numeric values, enables
optimization, and records source lineage. The template itself has
`optimization.enabled=false`.

The replicated transition plan and profiles are under:

```text
examples/campaigns/phase_1_full_genome_to_curriculum_fleet_v1/
```

The idempotent campaign materializer is:

```text
examples/scripts/materialize_execution_curriculum_campaign.py
```

The router profile and bounded parameter ranges are versioned at:

```text
examples/config/execution_router/project3_adaptive_order_router_v1.json
```

It remains blocked until the robust asset-policy artifact is frozen.

## 14. Acceptance

- Existing market-only configs produce unchanged behavior.
- Curriculum selection is deterministic across machines.
- Active costs appear in every policy observation when curriculum is enabled.
- Replay buffers never mix hidden cost regimes.
- Market, limit, and stop entries pass long/short fill fixtures.
- MIT, cancel/replace, passive exit, protection update and forced-close
  fixtures pass where the venue profile supports them.
- Untouched and expired pending orders do not create fills.
- Fill-time and event-hazard outputs are calibrated on validation.
- The learned policy beats market-only and deterministic-router controls under
  identical frozen alpha streams and declared replay fidelity.
- Pre-release mutation of an unpublished actual value cannot change an
  observation or action.
- Bar-only evidence cannot promote a queue-sensitive limit-order claim.
- Conservative OHLC ambiguity and pessimistic costs remain active.
- Robust fitness has weekly return units and fails closed.
- Resume hashes change for any curriculum, validation-suite, or router change.
- The complete focused and repository test suites pass before publication.

## 15. Implemented Components

| Repository | Component |
| --- | --- |
| `trading-contracts` 0.2.0 | optional, backward-compatible asset-intent urgency |
| `gym-fx` 0.3.0 | visible cost context; market, limit and stop entry requests; GTD expiry; cancel/market fallback; native Nautilus replay |
| `agent-multi` 0.4.0 | deterministic curriculum wrapper, robust weekly metrics/fitness, SAC observation expansion, adaptive router, fail-closed follow-up materializer |

Weekly trace rows remain in return-trace artifacts for OLAP ingestion. Candidate
and blockchain telemetry retain compact per-scenario metrics, hashes and the
robust aggregate rather than embedding the full trace in every candidate.

The deterministic router is implemented. Auxiliary execution models and the
shared learned entry/exit policy are planned components blocked on the
execution-fidelity gate and frozen alpha handoff; they are not represented as
already completed code.

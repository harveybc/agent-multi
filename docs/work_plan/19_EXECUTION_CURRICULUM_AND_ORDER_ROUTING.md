# 19. Execution Curriculum and Adaptive Order Routing

Status: implementation verified locally; active USDCAD campaign unchanged
Decision date: 2026-07-27

## 1. Objective

Improve execution realism without discarding the asset-policy learning already
completed. The implementation adds two orthogonal capabilities:

1. a visible, versioned cost curriculum for policy fine-tuning; and
2. an account-independent router that selects market, limit, or stop entry.

The current `usdcad-4h-full-genome-sac-shared-v1` campaign remains byte-for-byte
unchanged. Its champion is the market-entry pretraining source for a new domain.
No running blockchain, candidate pool, resume contract, or fitness definition
is modified in place.

## 2. Non-Negotiable Boundaries

- Cost scenarios are evaluator conditions, never optimizer genes.
- The policy observes the active cost conditions during training.
- Validation scenarios are immutable and deterministic.
- Validation and test data never select curriculum difficulty.
- The execution router does not size an account or submit broker orders.
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

## 6. Hierarchical Entry Router

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

## 7. Optimization Sequence

The sequence minimizes repeated model training:

1. finish and archive the active market-bracket campaign;
2. load its exact champion weights and resolved genome;
3. fine-tune the asset policy under the cost curriculum;
4. select with immutable robust validation fitness;
5. freeze the robust asset-policy artifact;
6. optimize the deterministic execution router against frozen policy actions;
7. run one bounded joint refinement over only the asset-policy parameters that
   materially interact with routing;
8. package model, router, preprocessing, feature, cost, and metric contracts as
   one cell release;
9. use frozen cell releases in the portfolio optimizer.

The joint stage is skipped when the router-only stage provides no validation
improvement or when it increases tail instability.

## 8. Artifact Contract

Each accepted cell release contains:

- trained SB3 `.zip` policy and SHA-256;
- encoded and decoded winning genome;
- router config and hash;
- cost curriculum and validation-suite hashes;
- resolved canonical experiment config;
- ordered feature/preprocessing contract and hashes;
- complete per-scenario metric vector and robust aggregate;
- source DOIN domain, genesis, block, peer, and candidate identity;
- inference smoke evidence.

The release is usable without DOIN. DOIN distributes and improves it; it is not
required for local inference.

## 9. Campaign Transition Safety

The supervisor may enqueue the new campaign only after:

1. the active swarm reaches its normal stop barrier;
2. all participants agree on the final block and champion artifact hash;
3. the champion archive passes independent load/inference verification;
4. the new config resolves all profile paths and hashes;
5. all node revisions and plugin versions match;
6. the new domain/genesis is unique and no parallel chain exists.

The future job may be materialized before then, but it remains disabled and
must not be inserted into the active campaign plan automatically.

The disabled template is:

```text
examples/config/phase_1_asset_policy/optimization/
phase_1_asset_policy_usdcad_4h_execution_curriculum_template_v1.json
```

The launchable config must be generated with:

```text
examples/scripts/materialize_execution_curriculum_followup.py
```

Without both the archived champion policy and optimization-parameter files,
the materializer fails closed. It hashes both files, seeds the bounded
follow-up genome from the archived parameters, enables optimization, and
records source lineage. The template itself has `optimization.enabled=false`.

The router profile and bounded parameter ranges are versioned at:

```text
examples/config/execution_router/project3_adaptive_order_router_v1.json
```

It remains blocked until the robust asset-policy artifact is frozen.

## 10. Acceptance

- Existing market-only configs produce unchanged behavior.
- Curriculum selection is deterministic across machines.
- Active costs appear in every policy observation when curriculum is enabled.
- Replay buffers never mix hidden cost regimes.
- Market, limit, and stop entries pass long/short fill fixtures.
- Untouched and expired pending orders do not create fills.
- Conservative OHLC ambiguity and pessimistic costs remain active.
- Robust fitness has weekly return units and fails closed.
- Resume hashes change for any curriculum, validation-suite, or router change.
- The complete focused and repository test suites pass before publication.

## 11. Implemented Components

| Repository | Component |
| --- | --- |
| `trading-contracts` 0.2.0 | optional, backward-compatible asset-intent urgency |
| `gym-fx` 0.3.0 | visible cost context; market, limit and stop entry requests; GTD expiry; cancel/market fallback; native Nautilus replay |
| `agent-multi` 0.4.0 | deterministic curriculum wrapper, robust weekly metrics/fitness, SAC observation expansion, adaptive router, fail-closed follow-up materializer |

Weekly trace rows remain in return-trace artifacts for OLAP ingestion. Candidate
and blockchain telemetry retain compact per-scenario metrics, hashes and the
robust aggregate rather than embedding the full trace in every candidate.

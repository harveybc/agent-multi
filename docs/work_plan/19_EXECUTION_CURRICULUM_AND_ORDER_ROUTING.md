# 19. Execution Curriculum, State Models, and Adaptive Order Policy

Status: corrected ETH easy-normal smoke verified and four-worker `full-v2`
running; paired normal/easy/easy-normal decision packet is the next compute
priority before curriculum generalization
Decision date: 2026-07-29

## 1. Objective

Improve execution realism without discarding the asset-policy learning already
completed. The plan separates three orthogonal capabilities:

1. a visible, versioned cost curriculum for policy fine-tuning; and
2. an account-independent deterministic router that selects market, limit, or
   stop entry as the mandatory control; and
3. a multi-timescale learned execution layer that estimates fill, adverse
   selection, short paths and event hazard before selecting entry/exit order
   actions.

The `usdcad-4h-full-genome-sac-shared-v1` domain was stopped on 2026-07-29
after its blockchain proved that one annual validation trade could win by
remaining almost flat. Its chain, 60 completed candidates and artifacts are
preserved as incident evidence; they are never resumed or used as an eligible
warm start.

The successor is `phase-1-protected-execution-fleet-v2`. Job 0 starts a fresh
domain/genesis under `easy_floor` positive costs, optimizes the complete mixed
genome and requires both annual activity and protected entries. Job 1 is a
separate warm-started domain that advances from easy through nominal and stress
costs with immutable robust weekly-RAP validation. No existing blockchain,
candidate pool, seed or fitness is mutated in place.

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
- Every risk-increasing market, limit or stop parent is submitted as a bracket
  with both SL and TP children. A plugin exception fails closed.
- Risk-reducing closes may cancel an existing bracket and flatten exposure;
  they cannot increase or reverse exposure in the same bar.
- Annual validation requires at least 12 completed trades. Losing active
  policies remain eligible; inactivity does not.

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

### 3.1 Evidence-gated solvency curriculum

Solvency/termination is a proposed second curriculum axis, independent of
execution costs. It originates in the owner's historical NEAT observation:
realistic margin-call termination repeatedly killed candidates before they
learned, while train-only continuation produced policies that later survived
realistic margin, fees and rates.

It is not active in the paused/resumable USDCAD job 0 or job 1. On 2026-08-05
the owner activated it for the new ETH campaign and superseded the earlier
instrumentation-blocked priority. The active implementation must expose
termination causes before producing ETH curriculum evidence; that dependency is
part of the authorized work rather than a reason to postpone it.

The ETH implementation requires this sequence:

1. distinguish `data_end`, `min_equity`, external stop and safety termination
   in `gym-fx` and propagate split/mode aggregates to candidate OLAP evidence;
2. keep separate train-only and evaluation solvency contracts;
3. implement forced liquidation, full loss accounting, explicit
   recapitalization debt and chronological continuation so training remains
   active after would-be ruin without manufacturing equity;
4. select every arm under the unchanged realistic train-tail/validation suite;
5. run matched `normal-only` and stage-integrated `easy -> normal` DOIN domains
   with identical data, seeds, initial population and budget, each with its own
   domain/genesis and no cross-arm champion migration.

Naively allowing negative cash is not the implementation: current
relative-volume sizing falls to zero when cash is negative, creating inert
states rather than continued trading. The implementation therefore separates
economic equity/debt from deterministically recapitalized operational training
capital. Live/demo solvency is never relaxed.
The canonical design and collision test are in
`docs/audits/AUDIT_SOLVENCY_RELAXATION_CURRICULUM_2026_08_02.md`.

### 3.2 Current ETH decision state (2026-08-05)

The corrected anchored smoke completed four distinct claims on one chain and
preserved a loadable champion artifact. The active
`trading-asset-policy-eth-4h-anchored-full-v2` campaign is an easy-to-normal
arm and currently has one shared population/tip across four GPUs.

The committed one-seed mechanism fixture is not sufficient to generalize the
curriculum. Under realistic 2024 validation, normal-only returned +0.02754%
mean weekly and +1.41384% total with 3.62981% maximum drawdown. Easy-only and
easy-to-normal both returned +0.01111% mean weekly and +0.55662% total with
2.67410% maximum drawdown; their selected policy payloads were identical and
no easy episode crossed a would-margin-call boundary.

Therefore the next decision packet uses four fresh paired seeds and equal
training compute:

```text
N14:     14 normal epochs
EN4_10:   4 easy epochs -> 10 normal epochs
E4:       4 easy epochs -> normal-condition inference diagnostic
```

All arms keep the same ETH/SAC anchor, genome, data, causal observation,
execution policy, validation and seed, with 20,000 timesteps per training
epoch. Each GPU owns one seed and runs every arm sequentially. Selection uses
the complete validation lexicographic tuple; reports retain raw weekly/annual/
total return, drawdown, activity, action, order, termination, entropy, compute,
trace and artifact evidence. The disclosed 2025 period remains disabled.

The primary decision is N14 versus EN4_10. E4 diagnoses whether learned easy
behavior survives realistic conditions. If no margin event occurs, the packet
must state that solvency relaxation itself remains untested and may run a
separate non-promotional stress probe. Any easy effect is then ablated across
solvency, costs and action deadband before broad rollout.

The active DOIN campaign stays productive while the experiment and reversible
pause/resume tooling are prepared. It is paused only after profile-drift,
GPU-verification and same-chain-resume findings 119/121 pass independent
preflight. Results authorize at most ETH/SAC; a second SAC asset and a separate
model family require reduced transfer checks before system-wide adoption.

The 2026-08-06 preflight accepted this experimental design but did not
authorize execution. Findings 122-126 require a real operator boundary,
post-rejoin chain proof, blocking profile guard, fail-closed GPU verification,
best-plus-terminal evaluation, explicit margin/termination telemetry, a pinned
base contract, strict four-seed aggregation and a verified two-host artifact
manifest. The running `full-v2` chain is not paused until those corrections
pass independent reproduction.

Direct logs also opened finding 127: after warm-up, trade-gate failures do not
consume patience, allowing collapsed policies to run all 2,000 epochs. The
correction uses a separate bounded activity patience and is a semantic change.
Consequently `full-v2` is preserved at pause, while the post-decision full
campaign starts on a fresh domain with the selected curriculum and corrected
patience behavior; no mixed-semantics continuation is permitted.

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

1. optimize the complete mixed genome under `easy_floor` positive costs with
   protected market/limit/stop/adaptive entries;
2. reject any candidate that misses train-tail or annual-validation activity,
   collapses its actions, lacks its exact model artifact, or emits an
   unprotected entry;
3. archive the agreed eligible champion weights, resolved genome, metrics and
   five diverse elites;
4. warm-start a new domain and fine-tune under the visible cost curriculum;
5. select with immutable robust validation fitness;
6. freeze the robust asset-policy artifact;
7. pass the execution-fidelity gate and train/calibrate the auxiliary models;
8. optimize the deterministic execution router against frozen policy actions;
9. train the shared entry/exit policy locally and compare it with market-only
   and deterministic-router controls;
10. optimize the learned execution policy with DOIN over allowed feature masks,
   order family, offsets, TTL, fallback, size hints and model parameters;
11. run one bounded joint refinement over only the asset-policy parameters that
   materially interact with routing;
12. package model, router/policy, auxiliaries, preprocessing, feature, cost and
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

## 15.1 ETH Reference Stack Before Multi-Asset Campaigns (2026-08-06)

ETH is the reference laboratory for the complete per-asset stack. The first
easy/normal comparison calibrates whether solvency relaxation enters later
search and bounds its genes; it does not freeze the final schedule. Dedicated
DOIN domains then optimize causal/decomposition inputs, point-in-time event
context, representations/autoencoders, SAC topology and learning dynamics,
auxiliary heads and bounded synthetic pretraining. Weak or unavailable lines are
rejected or deferred under a predeclared rule rather than left open indefinitely.

No decision-bearing package default may freeze. Each material value is a hard
invariant, evidence-fixed, a typed gene with justified bounds, or experimentally
excluded. Component optimizers remain locally runnable and expose only thin DOIN
adapters. They select by downstream realistic-normal trading utility; internal
reconstruction, forecast or generator metrics are eligibility/diagnostic facts.

After adopted variants are jointly reconfirmed, the following interface freezes
as one reusable per-asset template:

- source and point-in-time availability contract;
- feature order, preprocessing and observation shape;
- model architecture and learning dynamics;
- fixed-total-compute curriculum gene schema and activity patience;
- action, order, protection and cost semantics;
- selection, raw metric and artifact contract; and
- live inference/parity evidence schema.

Each component domain preserves a champion and diverse elites. A restricted
joint ETH domain then co-optimizes only confirmed genes/ranges and fixed-budget
curriculum allocation. Its winner is confirmed against matched normal-only and
neighboring easy-normal schedules before release.

The transferable artifact is the global causality/interface/safety contract,
component families, gene schema and justified ranges. ETH-specific masks,
topology, hyperparameters, curriculum, risk geometry and weights remain the ETH
solution and are not copied blindly. A representative second asset validates the
search contract, then per-asset DOIN campaigns optimize their own values. The
portfolio starts after the selected library contains loadable champion weights,
resolved genomes, complete metrics, traces and lineage for every included cell.

Portfolio simulator code may be unit-tested earlier with deterministic fixtures.
That is infrastructure verification, not portfolio optimization and not a reason
to divert campaign GPUs from the ETH or per-asset sequence.

## 16. ETH Solvency-Curriculum Audit State (2026-08-05)

The train-only `easy_chronological_continuation` environment and the
`post_easy -> normal_realistic` weight continuation pass focused and complete
local suites. The first distributed ETH-EN campaign does not pass acceptance.
It was stopped and disabled after independent audit found:

- internal checkpointing still used the former risk-adjusted composite;
- the configured outer lexicographic metric was unimplemented;
- the transport scalar could reverse the declared ordered tuple;
- evaluation failures remained champion-eligible and one `-1e9` failure was
  accepted on-chain;
- optimizer artifacts retained USDCAD identities and paths; and
- four workers remained on three equal-height tips while fork repair raised an
  exception.

Canonical audit and correction order:

```text
docs/audits/AUDIT_SATOSHI_III_ETH_CHAMPION_STAGE_CURRICULUM_2026_08_05.md
docs/handoffs/MUSASHI_TO_SATOSHI_III_ETH_CURRICULUM_CORRECTION_ORDER_2026_08_05.md
```

The archived chain is diagnostic evidence, not a resumable scientific run.
The next executable step is a new-domain local candidate followed by a
four-worker, one-generation smoke. Full ETH-EN/ETH-N optimization and SAC Demo
authority remain blocked until that smoke independently proves objective,
rejection, artifact, convergence, pause and raw-feature parity contracts.

### 16.1 Corrected anchored execution (2026-08-05)

The corrected current-stack ETH smoke is complete. All four workers used one
seed, genesis and shared population; each claimed one of four distinct
candidates. The terminal equal-height branches converged to tip
`13fbfbe5369f06d1d8562db1e62bec3f96ac894c66a6028c412f8b51e09d74d3`
after block transfer was given a dedicated 120-second timeout. The winning
candidate produced 17 normal-validation trades, +0.9695% total return and
5.0750% maximum drawdown. Its loadable SB3 artifact is 33,705,132 bytes with
SHA-256 `4da5de5eaa2e7455130ea36b1a7d14f65007d9ea3b4d5ad8d556a0879a6e4230`.

The first full anchored domain was then rejected after two of twenty
candidates exposed an SB3 fixed-to-automatic entropy restoration defect
(`policy.optimizer`). That chain remains immutable evidence and is not a
resume source. The correction builds SAC from each candidate's decoded genes
and transfers the champion policy state without stale optimizer moments.

The executable campaign is now the fresh single-job domain:

```text
examples/campaigns/phase_2_eth_anchored_full_fleet_v2/
examples/config/phase_2_eth_anchored/optimization/phase_2_eth_anchored_full_v2.json
```

Its four workers must continuously match plan hash, semantic domain hash,
dataset hash, seed, genesis, population fingerprint and component revisions.
The `full-v2` artifact root is separate from both the accepted smoke and the
rejected `full-v1` run.

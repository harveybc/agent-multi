# 33. ETH Decision, Research, and Multi-Asset Roadmap

Status: active
Version: 1.0.0
Date: 2026-08-06
Owner priority: decide the ETH easy/normal curriculum, mature the complete ETH
stack, transfer the frozen winner to all selected assets, and only then optimize
the portfolio

## 1. Purpose

This document integrates the ETH solvency-curriculum decision, live-versus-sim
calibration, feature and agent research, per-asset optimization, and portfolio
construction into one dependency-aware program. It replaces a single long
sequence with two coupled tracks:

1. a critical delivery track that matures ETH, freezes the winning stack,
   produces the selected multi-asset library and then enters portfolio work;
2. a parallel research track whose cheapest pilots determine which improvements
   deserve full ETH confirmation before that stack is frozen.

The source proposal remains preserved at:

```text
docs/handoffs/SATOSHI_III_RESEARCH_ROADMAP_PROPOSAL_2026_08_06.md
```

The executable assignment and precedence rules are in:

```text
docs/handoffs/MUSASHI_TO_SATOSHI_III_CONSOLIDATED_ETH_DECISION_AND_RESEARCH_ORDER_2026_08_06.md
```

## 2. Corrected Current State

- `phase-2-eth-anchored-full-fleet-v2` is operationally coordinated on one
  domain, chain and shared pool. That does not make its candidate behavior
  scientifically healthy: four observed candidates remained activity-ineligible
  after warm-up and the current patience logic can run them to 2,000 epochs.
- The `N14`, `EN4_10`, and diagnostic `E4` design is accepted. Execution is not
  accepted until findings 113 and 122-127 pass independent verification.
- The currently documented pause/resume path does not yet prove an authenticated
  exact-chain rejoin. A prose binding or pre-launch equality check is not a
  substitute for post-rejoin lineage evidence.
- The fixed selection contract is `lexicographic_weekly_v1`. It is a
  weekly-return-first ordered tuple with risk and total return tie-breakers. It
  must not be described as a numerical return/risk ratio. Reports preserve raw
  return, drawdown, activity and cost values in their stated units.
- The ETH pipeline is not yet called end-to-end proven. That claim requires the
  selected SAC artifact, exact feature contract and hashes to drive at least one
  protected Demo lifecycle and reconcile against direct venue facts.

## 3. Non-Negotiable Experimental Contract

### 3.1 Decision-stage evidence

A result may change the frozen model, data or curriculum contract only after:

- the cheapest feasibility pilot passes;
- the comparison uses paired anchors and equal compute;
- four paired seeds are complete for a campaign-level decision;
- every declared seed and arm is present exactly once;
- best-checkpoint and terminal weights are both preserved and evaluated;
- selection and all transforms exclude the protected test period;
- raw metrics and paired differences are published without an opaque composite;
- artifacts, traces, configs, data and code revisions have content hashes; and
- a second verified artifact copy exists outside the producing host.

Mechanical tests, CPU diagnostics and one-seed feasibility pilots do not require
four seeds. They cannot promote a contract or make a general performance claim.

### 3.2 Metrics

Every decision table reports, at minimum:

- mean weekly net simple return;
- annualized return derived from the ordered weekly series;
- total net return;
- maximum drawdown as fraction and percent;
- trades, wins, losses, long/short/hold actions and order-family counts;
- SL, TP, close and expiration outcomes;
- explicit commission, spread, slippage and financing drag when available;
- termination, would-margin-call and recapitalization facts;
- wall time, GPU time, peak memory and thermal samples.

The current lexicographic selector remains frozen during the curriculum A/B so
the experiment changes one mechanism. The report additionally includes the
Pareto frontier over weekly return and maximum drawdown and a risk-bounded raw
table. Those diagnostics do not silently change the active objective.

### 3.3 Leakage and calibration

- Every online feature must be invariant at time `t` to appending rows after
  `t`, including fitted preprocessing state.
- Calendar actuals become available only at their recorded release timestamp;
  revisions require point-in-time vintages.
- A live execution profile is estimated on a frozen calibration window and
  assessed on a later holdout window. The same fills cannot both tune and prove
  the profile.
- Live downtime and missing decisions are explicit coverage failures, not rows
  removed because they are inconvenient.

## 4. Critical Delivery Track

### D0. Correct and verify the decision harness

Close or independently verify findings 113 and 122-127 without mutating the
active domain. Required outcomes include exact repair-rule validation,
fail-closed GPU/pause evidence, blocked profile drift, authenticated loopback
mutation, actual post-rejoin chain proof, a pinned A/B contract, complete best
and terminal evidence, idempotent four-GPU orchestration, two verified copies,
strict aggregation and bounded activity-ineligible patience.

### D1. Execute the ETH curriculum decision

Run exactly four paired seeds with these arms:

- `N14`: 14 normal epochs;
- `EN4_10`: 4 easy epochs followed by 10 normal epochs;
- `E4`: diagnostic easy-only arm, never a deployable winner by itself.

Use 20,000 timesteps per epoch, shared per-seed anchors, realistic-normal
validation and no early stopping. The decision is based on paired raw outcomes,
not a pooled average that can hide seed disagreement.

### D2. Freeze the curriculum winner as the research baseline

After the owner chooses the curriculum from the complete packet, freeze its
best-checkpoint and terminal evidence as the research baseline. Do not start a
multi-day full DOIN campaign yet. Use the fixed-genome/paired harness to resolve
the remaining input, preprocessing and learning-stack research first.

Do not hot-patch or resume `full-v2` under changed evaluation semantics.
Preserve it as diagnostic lineage. Its GPUs move directly into the first bounded
research job after the decision packet, avoiding an idle gap without paying for
an incomplete full optimization.

### D3. Prove exact-model Demo parity

In parallel with D0-D2, inventory the active Paper/Demo seats and prove which
artifact actually drives each decision. The ETH seat must record the exact SAC
artifact/config/feature/input/decision hashes. A linear or heuristic controller
may remain a labeled shadow but cannot be reported as the ETH champion.

Build a due-bar versus synchronized-simulation join with:

- action agreement and disagreement reason;
- trade and hold rates;
- entry family and mandatory SL/TP geometry;
- modeled versus realized spread, slippage, latency, rejection and fill facts;
- protection and reconciliation outcomes; and
- expected versus observed decision coverage.

Estimate execution costs on a calibration window, freeze a versioned profile,
then assess it on a later holdout. Daily reports are descriptive; weekly reports
remain descriptive until sample requirements are met.

### D4. Freeze the mature ETH reference stack

The curriculum winner is the starting point, not automatically the final
per-asset template. Complete the bounded ETH research sequence in section 5.
Each line begins with the cheapest decisive pilot and only credible signals earn
four-seed confirmation. Adopted improvements are jointly reconfirmed against the
unmodified curriculum winner before the ETH contract freezes.

Freeze only after every admitted ETH line is one of:

- accepted on complete paired evidence;
- rejected by its preregistered kill condition;
- explicitly deferred because its required data is unavailable; or
- retained as research that does not alter the per-asset input/model contract.

The frozen reference stack includes data, feature/preprocessing, architecture,
training curriculum, action/order, cost, metric and artifact contracts. This is
the reusable template for the expensive per-asset campaigns.

### D5. Run the final full ETH optimization once

Materialize a fresh semantic domain with the D4 contract and corrected activity
patience. Run all workers on one genesis, shared pool and chain. The accepted ETH
release contains the loadable policy artifact, winning genome, preprocessing and
feature contract, data/config/code hashes, complete metric vector, deterministic
action trace and inference smoke. No earlier diagnostic chain is a resume source.

### D6. Transfer to one representative second asset

Before claiming SAC-wide behavior, run a reduced decision packet on one second
asset chosen from the evidence-backed universe using:

- complete causally available data;
- venue/inference availability;
- usable validation coverage;
- distinct market behavior from ETH; and
- expected portfolio diversification value.

The reduced packet tests transfer of the complete frozen ETH stack. It does not
repeat every rejected research option and does not require the second asset to
beat ETH in standalone return. A material transfer failure returns to D4 once
with a bounded cross-asset correction; it does not start asset-by-asset redesign.

### D7. Optimize and freeze the selected per-asset library

Apply the D4 contract to each selected asset using one coordinated DOIN campaign
at a time and all available workers on the same chain. Preserve champion weights,
genome, metrics, traces and complete lineage for every cell. A cell may be kept
as a diversification control even when standalone return is weaker, but it must
pass activity, safety, data and artifact gates.

Do not enter portfolio optimization until the owner-selected library is complete
or the owner explicitly narrows the universe based on the per-asset evidence.

### D8. Optimize the portfolio

Portfolio work begins with the frozen D7 library. Use frozen cells and the
existing Nautilus multi-asset replay path to implement:

1. a two-asset synchronized observation and target-exposure contract;
2. account-level cash, equity, margin, concentration and turnover accounting;
3. mandatory per-order SL/TP and venue capability checks;
4. static equal-risk and inverse-volatility baselines;
5. a simple constrained allocator using frozen action traces; and
6. deterministic replay plus Paper/Demo shadow evidence.

Later model improvements challenge frozen cells through champion succession.
They do not invalidate the portfolio contract or force an immediate full-library
reoptimization unless their input/action interface changes.

Portfolio software may be unit-tested earlier with deterministic fixtures. Such
tests consume no per-asset campaign budget and are not called the portfolio
optimization phase.

## 5. Parallel Research Track

### R0. Live/sim and training diagnostics - start now, CPU first

- Complete D3's direct artifact and due-bar parity inventory.
- Add post-hoc spectral-band and phase diagnostics only as supplementary
  channels. Pre-register estimator, window, normalization and minimum sample;
  do not let them block or replace direct action/cost evidence.
- Diagnose observed `ent_coef` and entropy logging before asserting entropy
  collapse. Separate fixed coefficient, automatic tuning, missing telemetry,
  reward scale and actual policy-entropy behavior.

### R1. Causal input expansion - separate families

Do not combine calendar and decomposition features into one first campaign.
Build and screen these contracts independently:

1. simple structured calendar surprise/importance/decay features;
2. each causal decomposition family, with future-append invariance tests;
3. a combined contract only if one or more independent families show stable
   marginal value.

Feature selection reports stability across seeds and resamples. One GA genome
dropping or keeping a group is not sufficient evidence by itself.

### R2. SAC learning dynamics - before expensive encoders

After R0 identifies the failure mode, compare the smallest relevant changes:

1. reward-scale and automatic entropy targets/ranges;
2. replay recency or actor-aware prioritization only if uniform replay remains a
   measured bottleneck;
3. a published actor-critic replay baseline before a custom implementation.

Vanilla TD-error PER is not presumed beneficial for SAC. Actor-critic studies
report mixed or adverse behavior, so this line needs its own bounded evidence.

### R3. Time-series foundation model event context - conditional

First establish that the cheap point-in-time calendar baseline in R1 has
headroom. Then benchmark frozen candidate encoders on the exact dataset and
latency contract. Chronos-2, Moirai-family and other candidates are options, not
preselected winners. Pin model bytes and licenses, measure causal availability,
embedding dimension, latency, memory and incremental value over the simple
baseline.

### R4. Auxiliary and self-supervised representations - conditional

Test latent self-prediction, causal-component auxiliary heads and frozen SSL
embeddings as separate arms. Each starts with one cheap feasibility seed. Only a
credible signal earns four paired seeds. Record representation rank/variance,
gradient interference and the additional optimization dimensions.

### R5. Synthetic regimes - independent high-risk line

Before freezing D4, give synthetic regimes a bounded feasibility decision.
Compare simple transparent generators such as moving-block bootstrap,
regime-conditioned resampling and fitted volatility/state baselines before a
diffusion model. Any neural generator must pass memorization, diversity,
stylized-fact and real-only downstream utility tests. Synthetic samples may
train or pretrain; they never select or validate a model. A failed or ambiguous
pilot closes/defer this line and cannot hold the entire program indefinitely.

## 6. Corrected Ordering

```text
critical: D0 -> D1 -> D2 -> bounded ETH R1-R5 -> D4 -> D5 -> D6 -> D7 -> D8
                    \-> D3 runs continuously and calibrates later simulations

parallel CPU: R0 -> R1 feasibility
ordered ETH GPU after D1: R2 -> conditional R3/R4/R5
```

The fleet has one GPU owner at a time. During D1 and D2, optional GPU research
waits. CPU-only research and live evidence collection continue.

## 7. Stop and Promotion Rules

- Stop a pilot when its data contract, action path or metric evidence is invalid;
  do not spend remaining compute to make the packet look complete.
- Stop an option after a negative feasibility result unless a specific measured
  defect explains the result and one bounded correction is preregistered.
- Promote an option only on complete paired evidence and report its incremental
  GPU-hour cost.
- Do not launch D5 while an admitted interface-changing ETH line
  remains untested and unclassified.
- Do not let a negative or ambiguous optional line hold D4 indefinitely; apply
  its preregistered kill/defer rule.
- Do not publish a cross-asset claim from ETH alone.
- Do not claim live parity until the exact selected artifact is the controller,
  all due inputs are causal and direct venue facts reconcile.

## 8. Primary Reference Disposition

The roadmap's literature is hypothesis support, not expected performance:

- Chronos-2 officially supports multivariate and covariate-informed zero-shot
  forecasting, but that does not establish utility for this SAC policy:
  https://arxiv.org/abs/2510.15821
- Salesforce describes Moirai 2.0 as a decoder-only forecasting model; the exact
  event-covariate interface and our use case still require a local benchmark:
  https://www.salesforce.com/blog/moirai-2-0/
- Actor-prioritized replay reports why ordinary TD-error prioritization can harm
  actor learning; replay is therefore an experiment, not a default upgrade:
  https://arxiv.org/abs/2209.00532
- ROER reports SAC gains on 6 of 11 tasks and parity or small differences on the
  rest, reinforcing the need for a domain-specific test:
  https://arxiv.org/abs/2407.03995
- CoFinDiff reports stylized-fact fidelity and downstream deep-hedging utility;
  it does not prove improved trading-policy control:
  https://www.ijcai.org/proceedings/2025/1040

Incomplete DOI fragments and unverified 2026 claims in the proposal cannot
support implementation priority until the academic track resolves them to
primary sources.

# 33. ETH Decision, Research, and Multi-Asset Roadmap

Status: active
Version: 1.3.0
Date: 2026-08-08
Owner priority: calibrate ETH easy/normal training, mature and jointly optimize
the complete ETH stack, transfer the validated search contract to all selected
assets, and only then optimize the portfolio
Technical lead: General Satoshi III (owner promotion, 2026-08-06)
Independent auditor: General Musashi

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
docs/handoffs/MUSASHI_TO_GENERAL_SATOSHI_III_J0_J4_RETRAINING_CORRECTION_ORDER_2026_08_06.md
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

### 3.2 No unoptimized decision-bearing defaults

Every parameter that can materially change observations, representation,
learning, actions, risk or selection must have exactly one recorded status:

1. fixed by a hard business/safety invariant;
2. fixed by prior project evidence with an immutable citation;
3. exposed as a typed optimization gene with evidence-based bounds; or
4. excluded after a bounded ablation or infeasibility result.

Library defaults may initialize a pilot, but they never justify a frozen value.
Every component domain publishes a parameter registry containing type, units,
bounds/choices, bound provenance, conditional activation, repair rules and final
disposition. Unknown or inactive parameters fail closed rather than silently
falling back to a package default.

Component optimization is hierarchical, not one enormous first genome:

```text
contract/causality proof
  -> cheap local range/feasibility screen
  -> dedicated DOIN component domain
  -> downstream realistic-normal confirmation
  -> diverse elite set and sensitivity-supported bounds
  -> restricted joint integration domain
```

Every optimizer must remain runnable locally without DOIN. Its plugin extends
that local optimizer with migration, lineage and distributed evaluation; it does
not place component logic inside `doin-node`.

### 3.3 Metrics

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

### 3.4 Leakage and calibration

- Every online feature must be invariant at time `t` to appending rows after
  `t`, including fitted preprocessing state.
- Calendar actuals become available only at their recorded release timestamp;
  revisions require point-in-time vintages.
- A live execution profile is estimated on a frozen calibration window and
  assessed on a later holdout window. The same fills cannot both tune and prove
  the profile.
- Live downtime and missing decisions are explicit coverage failures, not rows
  removed because they are inconvenient.

### 3.5 Exact current ETH data and observation contract

The current 4-hour ETH decision runner consumes the immutable dataset SHA-256
`1b447c66e68495e826c53e2ab2b08ecd3922c8fdc735747628f8d0435ebe440f`.
It contains 18,085 rows: 13,699 training bars from 2017-09-28 04:00 through
2023-12-31, 2,196 validation bars in 2024, and 2,190 disclosed 2025 test bars.
The N14/EN4_10/E4 packet keeps the 2025 period disabled.

The corrected current observation has 83 engineered features over 32 bars plus
four live-observable agent-state values: signed position, equity relative to
session start, unrealized PnL from the true entry price and capped holding
duration. It contains exactly 2,660 flattened values. Raw 32-price and
32-return windows are prohibited: their absolute scale killed or saturated the
first actor layer. The 32-bar context spans 128 hours; rolling scaling over 256
bars spans 1,024 hours. Live provisioning retains at least 800 closed H4 bars
so the longest causal feature warm-up and scaling history are finite; missing
warm-up refuses inference rather than replacing missing values with zero.
The explicit date split overrides the dormant base-config `train_years=4` with
approximately 6.25 years of unique training history. Materializers must remove
that contradiction or declare the override in their resolved contract.

These values are **currently used**, not thereby evidence-selected. Window,
scaling and lookback parameters remain subject to the no-default registry until
comparative evidence freezes or excludes them.

### 3.6 Adaptation and structural-optimization clocks

The business has two distinct clocks:

1. fixed-contract weight/replay adaptation using newly available causal bars;
2. slower DOIN structural optimization of features, preprocessing, topology,
   hyperparameters, curriculum or succession candidates.

Do not claim that the complete distributed structural search fits inside a
6- or 12-hour deployment interval until measured candidate and campaign
latencies prove it. Fast adaptation may fit; RT0 measures it. Cadence is encoded
in closed bars. For the current 4-hour source, screen 2/3/6/18/42 bars
(8/12/24/72/168 hours), with one bar/4 hours as a feasibility stress case.
Six hours is not bar-aligned and is excluded unless a causal one-hour input
contract is later selected.

Selection is rolling-origin and prequential: at origin `t`, score the next
interval before its rows can enter adaptation. Ordered next-interval and weekly
series govern selection; a complete-year static score remains a baseline, not
the final deployment objective.

## 4. Critical Delivery Track

### D0. Correct and verify the decision harness

Close or independently verify findings 113 and 122-134 without mutating the
active domain. Required outcomes include exact repair-rule validation,
fail-closed GPU/pause evidence, blocked profile drift, authenticated loopback
mutation, actual post-rejoin chain proof, a pinned A/B contract, complete best
and terminal evidence, idempotent four-GPU orchestration, two verified copies,
strict aggregation, bounded activity-ineligible patience, host-aware exact
controller provenance and a versioned adaptation contract.

### D1. Execute the ETH curriculum calibration

The historical bounded calibration used exactly four paired seeds with these
arms:

- `N14`: 14 normal epochs;
- `EN4_10`: 4 easy epochs followed by 10 normal epochs;
- `E4`: diagnostic easy-only arm, never a deployable winner by itself.

Use 20,000 timesteps per epoch, shared per-seed anchors, realistic-normal
validation and no early stopping. Those artifacts remain mechanism and collapse
diagnostics. They do **not** establish whether solvency relaxation deserves
entry into the mature optimization genome because a fixed 14-epoch allocation
does not test the owner's train-plus-validation stopping mechanism and the
historical easy handoff was affected by findings 159-161.

The decision-bearing replacement is document 38's nested chronological program:

1. `fit_train` through 2022, train-monitor 2022, inner validation 2023, outer
   validation 2024 and sealed test 2025;
2. high L1 epoch safety ceilings with paired train-monitor/inner-validation
   early stopping;
3. a matched four-cell normal/easy x normal-LR factorial over four seeds;
4. a frozen-L1 L2 normal-only versus staged-easy-normal comparison; and
5. a bounded 2x2 interaction only if an isolated axis survives.

No smoke or fixed-epoch diagnostic may be aggregated into that decision. The
full replacement launch is standing owner-authorized once its executable tests
and mechanics smoke pass; it does not wait for another owner phrase.

### D2. Materialize the provisional research baseline

Preserve N14, EN4_10, E4 and M0 evidence. Select a provisional schedule for
component search only from document 38's complete L1 packet, retaining
normal-only as the mandatory control. The provisional value is experimentally
chosen, not a library default. Do not start a broad full-stack DOIN campaign
before the isolated L1 and feature-selection mechanisms have decision evidence.

Every dedicated component domain keeps total training compute fixed. Where the
component plausibly interacts with curriculum, expose `easy_epochs` (including
zero) and derive `normal_epochs = total_epochs - easy_epochs`; never let a
candidate buy fitness by consuming more total epochs. Easy-only remains
diagnostic because promotion always requires normal-realistic training or
fine-tuning and normal-realistic validation.

Do not hot-patch or resume `full-v2` under changed evaluation semantics.
Preserve it as diagnostic lineage. Its GPUs move directly into the first bounded
research job after the decision packet, avoiding an idle gap without paying for
an incomplete full optimization.

### D3. Prove exact-model Demo parity

In parallel with D0-D2, inventory the active Paper/Demo seats and prove which
artifact actually drives each decision. The ETH seat must record the exact SAC
artifact/config/feature/input/decision hashes. It continues inference while a
position is open and can issue an explicit model close without removing the
mandatory native SL/TP protection. A linear or heuristic controller may remain
a labeled shadow but cannot be reported as the ETH champion.

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

### D4. Execute hierarchical ETH component domains

Complete the component sequence in section 5. A component with tunable topology,
feature selection, preprocessing or learning parameters receives a dedicated
DOIN domain after its cheap contract screen. Its final fitness is always
downstream realistic-normal trading validation; reconstruction, forecast or
generator scores are eligibility/diagnostic facts, never substitutes for control
utility.

Each domain archives the champion plus up to five strong, behaviorally diverse
elites. The next domain consumes those artifacts and evidence-supported ranges,
not one winner treated as unquestionable truth. OLAP records every genome,
component metric, downstream metric, resource fact and lineage.

### D5. Run restricted joint ETH integration and final curriculum optimization

After the component domains, create one integration genome containing only:

- component families that passed downstream confirmation;
- feature/component masks and parameters with measured sensitivity;
- compatible encoder/policy topology genes and evidence-supported ranges;
- SAC learning/replay genes retained by their dedicated domain; and
- fixed-budget curriculum genes, including normal-only as `easy_epochs=0`.

Warm-start the integration population from the diverse component elites. Do not
reopen every rejected gene or take the Cartesian product of all historical
ranges. The integration DOIN campaign co-optimizes interactions that isolated
domains cannot identify while keeping the search space bounded.

The winning curriculum is therefore selected near the end, jointly with the
mature stack, rather than assumed from N14/EN4_10. Confirm the integrated winner
against matched normal-only and nearest easy-normal controls over four seeds
before release.

### D6. Freeze the ETH reference release and transferable search contract

The accepted ETH release contains loadable component and policy artifacts,
winning/elite genomes, complete parameter registry, preprocessing and feature
contracts, data/config/code hashes, raw metric vectors, deterministic action
trace and inference smoke.

Separate two things explicitly:

- **frozen global contract:** causality, interfaces, safety, available component
  families, gene schema and evidence-supported bounds;
- **ETH-specific solution:** selected masks, topology, hyperparameters,
  curriculum allocation, risk geometry and trained weights.

The ETH-specific values are not copied blindly to every asset.

### D7. Transfer to one representative second asset

Before claiming SAC-wide behavior, run a reduced decision packet on one second
asset chosen from the evidence-backed universe using:

- complete causally available data;
- venue/inference availability;
- usable validation coverage;
- distinct market behavior from ETH; and
- expected portfolio diversification value.

The reduced packet tests transfer of the frozen search contract and initializes
from ETH elites. DOIN optimizes per-asset masks, windows, topology,
hyperparameters, curriculum allocation and risk geometry within the justified
ranges. It does not repeat rejected component research and does not require the
second asset to beat ETH in standalone return. A material transfer failure
returns to D5 once with a bounded shared-contract correction; it does not start
silent asset-by-asset architecture drift.

### D8. Optimize and freeze the selected per-asset library

Apply the D6 search contract to each selected asset using one coordinated DOIN
campaign at a time and all available workers on the same chain. Preserve
champion weights, genome, metrics, traces and complete lineage for every cell. A
cell may be kept as a diversification control even when standalone return is
weaker, but it must pass activity, safety, data and artifact gates.

Do not enter portfolio optimization until the owner-selected library is complete
or the owner explicitly narrows the universe based on the per-asset evidence.

### D9. Optimize the portfolio

Portfolio work begins with the frozen D8 library. Use frozen cells and the
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

### R1. Causal inputs and decomposition DOIN domains

Do not combine calendar and decomposition features into one first campaign.
Build and screen these contracts independently, then optimize each admitted
family through its own domain:

1. simple structured calendar surprise/importance/decay features;
2. wavelet family, basis, causal window, depth/bands and component mask;
3. multitaper window, bandwidth, band definitions and component mask;
4. causal Hilbert source, FIR order/window, phase/amplitude outputs and lags;
5. fractional-differencing order, threshold, window and source mask;
6. EMD family and component mask only after strict boundary-stability tests; and
7. a combined contract only if one or more independent families show stable
   marginal value.

Feature selection reports stability across seeds and resamples. One GA genome
dropping or keeping a group is not sufficient evidence by itself. Every family
optimizes source-feature selection, transform parameters, component selection,
normalization and warm-up behavior; no transform is applied to a price/feature
source merely because it is conventional.

### R2. Representation and autoencoder DOIN domains

Representation candidates are locally runnable plugins and expose typed genes
for encoder family, causal context, depth, width, latent dimension, masking,
loss family/weights, optimizer, learning rate, regularization and frozen versus
fine-tuned use. Use a two-level evaluation:

1. train-only representation eligibility: anti-collapse, reconstruction or
   self-prediction facts and resource limits;
2. downstream SAC utility under the same realistic-normal validation contract.

An encoder with excellent reconstruction and weak control utility is rejected.
Optimize autoencoder/SSL parameters with DOIN; do not compare one hand-configured
encoder against an optimized no-encoder baseline.

### R3. SAC topology and learning-dynamics DOIN domain

After R0 identifies the failure mode, compare the smallest relevant changes:

1. actor/critic depth and width, activation and feature-extractor coupling;
2. learning rates, batch/buffer size, gamma, tau, train frequency and gradient
   steps under bounded resource constraints;
3. reward scale and automatic entropy mode/target/range;
4. replay recency or actor-aware prioritization only if uniform replay remains a
   measured bottleneck; and
5. a published actor-critic replay baseline before a custom implementation.

Vanilla TD-error PER is not presumed beneficial for SAC. Actor-critic studies
report mixed or adverse behavior, so this line needs its own bounded evidence.
Inactive conditional genes must not leak package defaults into candidate builds.

The scalar actor uses the target-exposure contract from document 39: strong
positive/negative actions target long/short, a near-zero action explicitly
targets flat, and the hysteresis band preserves current exposure. Opposite
targets close first and cannot reverse in the same bar. Entry-order family is
held at market while this policy contract is calibrated; order routing receives
its own downstream experiment so fill mechanics cannot masquerade as a better
learning curriculum.

### R4. Event-context and TSFM adapter DOIN domain - conditional

First establish that the cheap point-in-time calendar baseline in R1 has
headroom. Then benchmark frozen candidate encoders on the exact dataset and
latency contract. Chronos-2, Moirai-family and other candidates are options, not
preselected winners. Pin model bytes and licenses, measure causal availability,
embedding dimension, latency, memory and incremental value over the simple
baseline. Optimize event families, surprise normalization, decay, context,
encoder/forecast output selection, adapter dimension, fusion and freeze/fine-tune
policy. Pretrained weights are fixed only by an immutable artifact hash; adapter
and usage parameters are not left at defaults.

### R5. Auxiliary-control heads DOIN domain - conditional

Test latent self-prediction and causal-component auxiliary heads as separate
families. Optimize target family, horizon, head topology, loss, loss weight and
gradient-sharing strategy. Each starts with one cheap feasibility seed. Only a
credible downstream signal earns a dedicated DOIN domain and four-seed
confirmation. Record representation rank/variance, gradient interference and
the additional optimization dimensions.

### R6. Synthetic-regime generator and pretraining DOIN domains

Before D5 integration, give synthetic regimes a bounded feasibility decision.
Compare simple transparent generators such as moving-block bootstrap,
regime-conditioned resampling and fitted volatility/state baselines before a
diffusion model. Any neural generator must pass memorization, diversity,
stylized-fact and real-only downstream utility tests. Synthetic samples may
train or pretrain; they never select or validate a model.

Generator parameters may be optimized in a fidelity domain, but promotion uses
a second downstream domain that optimizes synthetic/real ratio, curriculum
placement, pretraining duration and real-data fine-tuning. A failed or ambiguous
pilot closes/defers this line and cannot hold the entire program indefinitely.

### R7. Retraining/fine-tuning cadence and handover DOIN domain

Start RT0/RT1 after R3 fixes the SAC topology/learning contract. Finalize RT2
after every admitted interface-changing R4/R5/R6 line and before D5 joint
integration, so cadence is not optimized against a representation that is then
replaced.

1. **RT0 runtime feasibility:** one frozen 28-day 2024 block, one seed, fixed
   SAC/config, 8/12/24/72/168-hour cadences, one- and two-year lookbacks. A
   4-hour arm is admitted only when the measured update path can finish safely.
   Record p50/p95 latency, deadline misses, GPU/RAM/VRAM/temperature, new bars,
   model age, rollback and handover facts. This stage cannot promote by return.
2. **RT1 performance screen:** four preregistered non-overlapping 28-day 2024
   blocks, two paired seeds, 1y/2y/4y or expanding lookbacks, and strict
   test-then-train ordering. It estimates robustness across calendar regimes
   without spending a complete DOIN campaign on every combination.
3. **RT2 dedicated DOIN domain:** optimize bar-aligned cadence, rolling versus
   expanding lookback, warm/reset/bounded-refit mode, update budget per new bar,
   replay retention/recency, encoder freeze/fine-tune and bounded activation
   policy. Use one-block/one-seed, four-block/two-seed and full-2024/four-seed
   successive fidelity under a fixed compute/deadline contract.
4. **RT3 frozen confirmation:** freeze schedule and adaptation parameters, then
   evaluate prequentially. The repeatedly inspected 2025 period is a disclosed
   secondary benchmark; prospective 2026 Paper/live-shadow evidence is the
   clean final confirmation source.

A 28-day pilot proves orchestration, leakage controls and deadline feasibility;
it cannot alone choose profit/risk. The proposed operational deadline budget is
p95 update latency no greater than two thirds of cadence with zero unreconciled
handover, pending owner ratification.

Every model handover preserves account continuity: stop new risk, close and
reconcile protected exposure, record exact post-close balance, activate the
hash-bound successor, and resume. Mandatory SL/TP and Paper/Demo-only authority
remain unchanged.

## 6. Corrected Ordering

```text
critical: D0 -> D1 -> D2 -> D4/R1-R7 -> D5 -> D6 -> D7 -> D8 -> D9
                    \-> D3 runs continuously and calibrates later simulations

parallel CPU: R0 -> R1 feasibility
sequential DOIN: R1 -> R2 -> R3 -> R7 RT0/RT1 + conditional R4/R5/R6
                 -> R7 RT2 -> D5 integration
```

The fleet has one GPU campaign owner at a time. Each accepted component domain
uses all workers on one chain before the next component domain starts. CPU-only
research and live evidence collection continue in parallel.

## 7. Stop and Promotion Rules

- Stop a pilot when its data contract, action path or metric evidence is invalid;
  do not spend remaining compute to make the packet look complete.
- Stop an option after a negative feasibility result unless a specific measured
  defect explains the result and one bounded correction is preregistered.
- Promote an option only on complete paired evidence and report its incremental
  GPU-hour cost.
- Do not launch D5 integration while an admitted interface-changing ETH line
  remains untested and unclassified.
- Do not let a negative or ambiguous optional line hold D4 indefinitely; apply
  its preregistered kill/defer rule.
- Do not freeze a decision-bearing package default; optimize it, justify it as an
  invariant/evidence value, or exclude the component.
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

## 9. RT Implementation Audit and Sequential Screen (2026-08-06)

The first RT runner is not decision-bearing. Independent reproduction showed
that it scores its 256-bar warm-up as part of the next interval, resets equity
to 10,000 at each origin, and can reuse a run identity after initialization,
device, base-contract or code changes. Its current OLAP cannot answer the
business question and must not launch RT1.

After corrections 140-142 pass, the bounded screen is:

- RT1-A: `{2,3,6,42}` 4-hour bars x `{1y, expanding}` x four fixed
  non-overlapping 28-day blocks x two paired seeds, plus one frozen/no-update
  control in every block;
- RT1-B: add 18 bars and 2y/4y rolling lookbacks only for the best two cadence
  regions or when a boundary winner, non-monotone curve or interaction appears;
- warm-up supplies context but contributes no score;
- each block begins from a paired account state and preserves account balance,
  effects and handover costs across its origins; and
- update compute per new bar and the end-to-end deadline contract remain fixed.

The p95 two-thirds deadline is proposed only for end-to-end readiness from data
cutoff through durable artifact, validation, replica and activation-ready
state. Numeric RT2 bounds remain unratified until this evidence exists.

Canonical audit:
`../audits/AUDIT_SATOSHI_III_128_134_CORRECTIONS_AND_RT1_RULING_2026_08_06.md`.

## 10. RT v2 Acceptance Result (2026-08-06)

The second RT delivery is still not decision-bearing. Independent execution
against the real ETH/gym-fx path showed that removing 256 equity samples from
the metric does not make warm-up contextual: the policy trades during warm-up,
changes account equity and enters the scored interval with an unrecorded open
position. A three-bar cadence emits four scored facts, and the next origin
inherits only a cash number while the position and closing cost disappear.

The restart claim also fails at the exact SQLite-commit/JSON-pointer crash
boundary. RT identity has no starting policy artifact and origin zero builds a
fresh SAC, so the materialized grid would test random initialization rather
than cadence adaptation of a mature ETH champion. The deadline rule names
handover reconciliation without measuring it.

Consequently:

- RT1-A remains `MATERIALIZED_NOT_EXECUTED`;
- RT mechanics continue under findings 151-158 after the 143/146 corrections;
- performance execution remains after R3, using its load-proven hash-bound
  champion/config/observation contract;
- every boundary must score exactly h bars and execute an explicit flat,
  costed, reconciled model handover using the post-close balance; and
- authoritative restart state moves into the same SQLite transaction as the
  interval row, while JSON becomes a derived export.

Canonical audit:
`../audits/AUDIT_SATOSHI_III_135_142_ACCEPTANCE_2026_08_06.md`.

## 11. RT v3 Correction Audit (2026-08-06)

The third delivery fixes two important foundations: acceptance probes now have
typed outcomes, and SQLite commits each interval with its authoritative restart
state. Forced-hold warm-up and exact-h sample cardinality also work.

Performance execution remains blocked because independent CPU execution found:

- uninterrupted origin 1 did not inherit origin 0's adapted weights;
- the flat handover uses directional position as units, omits spread/slippage
  and asserts reconciliation without a simulator close;
- a bare compatible fresh SAC can be called a mature anchor;
- restarted p95 excludes latencies from earlier process sessions;
- every RT1-A cadence omits its final complete interval, including one of four
  weeks for the weekly arm;
- replica authority and untracked-source identity remain self-asserted; and
- same-second evidence with an unchanged PID generation proves rejoin.

Corrections 151-158 therefore precede any RT1-A execution. The next mechanics
fixture must contain at least three uninterrupted origins plus a restart and
prove exact model-hash succession, simulator-executed costed flat handovers,
all-row latency statistics, mature champion provenance and complete block
coverage.

Canonical audit:
`../audits/AUDIT_SATOSHI_III_143_150_CORRECTIONS_2026_08_06.md`.

## 12. D1 Four-Seed Result and D2 Disposition (2026-08-07)

The bounded D1 fleet run completed all 12 arm executions: seeds
`101/202/303/404`, each with `N14`, `EN4_10` and diagnostic `E4`. All 20
published model artifacts were independently re-hashed on their declared
replica authority. The complete fail-closed aggregate is:

```text
~/.local/share/agent-multi/eth_curriculum_decision_20260807_v2/
  fleet_manifest.json
  decision_summary.json
```

The selected-checkpoint result is a tie, not a curriculum gain. For seeds 101
and 404, every arm reports +0.0513556% mean weekly return, +2.71308% total
return, +2.70870% annualized return, 2.66335% maximum drawdown and 136 trades on
the same 2024 validation contract. For seeds 202 and 303, every arm reports
-0.0393064% mean weekly return, -2.11683% total return, -2.11349% annualized
return, 3.77031% maximum drawdown and 130 trades. Consequently every paired
`EN4_10 - N14` raw difference is exactly zero.

That equality is explained by artifact behavior, not by equivalent successful
learning. The selected `N14` and `EN4_10` checkpoint remained the unchanged
warm-start anchor in every seed. Every one of the 14 post-anchor normal epochs
in `N14` and every one of the 10 normal epochs in `EN4_10` failed the activity
gate; all eight terminal normal-trained artifacts produced zero validation
trades and zero return. `E4` retained 130-136 validation trades after its easy
phase but did not improve the anchor's raw metrics. Margin/recapitalization
telemetry remained unavailable, so this packet does not show that relaxed
solvency, specifically, caused the retained behavior.

D2 therefore does not admit a claimed easy-to-normal advantage. Normal-only
remains the mandatory control and `easy_epochs` remains an experimental gene,
including zero, rather than a frozen schedule. Before R1/R2 feature or
representation domains can spend downstream-control compute, R0/R3 must isolate
the measured SAC normal-update activity collapse using existing traces and a
bounded learning-dynamics screen. A new broad full-stack campaign or a resume of
`full-v2` is not evidence-compatible with this result.

One collection defect was found without invalidating the executions: Git's
host-specific abbreviation length recorded the identical `agent-multi` commit
as `46ce057` and `46ce057b`. `agent-multi@38643550` now records full SHAs and
allows historical abbreviations to canonicalize only against the complete SHA
captured by the ready fleet preflight. Raw packets and artifacts were not
modified.

## 13. D1 Follow-up: SAC Inner-Curriculum Mechanism Screen

The binding execution order for the D1 follow-up is:

`../handoffs/MUSASHI_TO_GENERAL_SATOSHI_III_SAC_INNER_CURRICULUM_ORDER_2026_08_07.md`.

This is a bounded lower-level SAC experiment, not another broad DOIN campaign.
Easy dynamics already run inside SAC; the immediate question is whether the
easy-to-normal handoff loses the learned behavior because normal continuation
reconstructs the learner with a fresh replay buffer, fresh optimizer state and
an abrupt dynamics change. The order first instruments that boundary and then
compares four equal-compute, four-seed arms: normal-only, unchanged-rate
easy-to-normal, and two reduced normal fine-tuning rates.

No positive-return screen is permitted. The first decision is mechanism
survival: a terminal policy must remain active under normal-realistic
validation, differ from its anchor, contain applied normal updates, remain
loadable and preserve protected execution. Only a mechanism that survives in
at least three of four paired seeds may enter the longer M1 confirmation. A
failed screen branches immediately into the pre-materialized R0/R3 collapse
diagnostic so the fleet does not wait for a new planning cycle.

This order does not authorize edits to `doin-node`, `doin-core` or
`doin-plugins`, does not authorize test-period evaluation and does not resume
the superseded `full-v2` chain. Any later DOIN domain exposes only controls
whose behavior this local screen has measured.

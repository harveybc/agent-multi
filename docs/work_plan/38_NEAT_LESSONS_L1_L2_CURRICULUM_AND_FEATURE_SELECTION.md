# 38. NEAT Lessons Applied to L1/L2 Curriculum and Feature Selection

Status: active execution contract, v1.0.0, 2026-08-08
Owner: project owner
Technical lead/independent verifier: General Musashi
Implementer: General Satoshi III
Scope: ETH reference stack first; transferable mechanisms only after falsification

## 1. Decision

The owner's historical NEAT results are primary motivating evidence, not a
claim that SAC or DEAP must behave identically. They establish four hypotheses
that are cheap enough and important enough to test explicitly:

1. relaxed training dynamics can prevent absorbing-state starvation;
2. sparse inherited access to inputs can improve search and consistency;
3. ordered representation/growth/tuning/maturation phases can improve search;
4. train-plus-validation stopping can control overfit better than a manual
   epoch cap or validation-only patience.

The project will test these mechanisms at the level where they actually act.
It will not infer an L2 benefit from an L1 result, or vice versa.

L1 and L2 curricula can coexist:

- **L1** changes gradient training of one candidate and therefore changes
  policy weights, replay experience and optimizer state.
- **L2** changes evolution of config genomes: initialization, active genes,
  mutation/crossover, promotion and generation stopping.

Because the two axes can interact, the sequence is L1 isolation, L2 isolation,
then a bounded interaction confirmation. The full integration genome receives
only mechanisms that survive their isolated comparison.

## 2. Evidence Classes

| Class | Purpose | Can decide? |
| --- | --- | --- |
| mechanics smoke | imports, schemas, handoff, artifacts, one short run | no |
| mechanism screen | detect collapse/activity and reject impossible contracts | no profit/risk promotion |
| decision run | paired seeds, sufficient chronology, stopping, raw metrics | yes, for its declared hypothesis |
| outer confirmation | nested outer year under frozen candidate/config | yes |
| sealed release test | one frozen release on 2025 | release evidence only |
| Paper/Demo evidence | execution and business-reality divergence | future research input, not historical rewrite |

N14/EN4_10/E4 and M0 remain preserved in the first two classes. They supplied
useful collapse and learning-rate information. They do not answer the new
decision question because they disabled early stopping and fixed phase length.

## 3. Chronological Data Contract

### 3.1 ETH decision partition

Pinned source:

`predictor/examples/data/project3/ethusdt_4h_tech_stat_full_model_ready.csv`

SHA-256:

`1b447c66e68495e826c53e2ab2b08ecd3922c8fdc735747628f8d0435ebe440f`

| Partition | Range | Rows | Authority |
| --- | --- | ---: | --- |
| `fit_train` | 2017-09-28 04:00 through 2022-12-31 | 11,509 | gradient updates only |
| `train_monitor` | calendar 2022, subset of fit | materializer computes exact | L1 paired stopping, in-sample side |
| `inner_validation` | calendar 2023 | 2,190 | L1 stopping, no fitting |
| `outer_validation` | calendar 2024 | 2,196 | L2 selection/confirmation, no fitting |
| `sealed_test` | calendar 2025 | 2,190 | disabled until release freeze |

`train_monitor` is the last complete fit year, not seven days. The materializer
must emit exact start/end/row count/hash instead of trusting this prose.

### 3.2 Causal prefix without score leakage

Every evaluation partition receives a context prefix of:

```text
max(observation_window, scaling_window, maximum_feature_lookback)
```

closed bars immediately preceding its score start. Prefix bars may initialize
causal transformations and observations. They cannot:

- emit actions or orders;
- mutate account cash/equity/positions;
- enter return, risk, trade or activity metrics;
- enter adaptation or replay; or
- change the declared scored dates.

This preserves a complete calendar-year score while preventing the 256-bar
scaling warm-up from silently removing six weeks.

### 3.3 Future rolling-origin contract

This static nested contract answers architecture/curriculum questions. The
business cadence remains a later rolling-origin decision per document 33. A
winning static contract is not relabeled as weekly or 12-hour retraining.

After L1/L2/feature configuration is frozen, the release refit advances the
origin once without changing the selected contract: fit through 2023, use 2023
as the in-fit monitor and 2024 as validation, then open 2025 exactly once for
release evidence. Hyperparameters, masks, curriculum and stopping semantics are
already frozen before that refit; 2025 cannot send work back into selection.

## 4. Common Economic Evidence

### 4.1 Raw metrics

Every checkpoint, candidate, generation and arm emits at least:

- total and mean weekly return;
- total and mean weekly RAP;
- annual return and annual RAP only for complete annual series;
- maximum drawdown, expected shortfall and downside deviation;
- trades, active weeks, turnover and all modeled costs;
- long/short counts, hold/non-hold actions and protection rejections;
- train-monitor, inner-validation and outer-validation weekly vectors;
- epochs, gradient updates, environment interactions, candidates, wall time and
  GPU time; and
- model/config/data/feature/code hashes and seed.

Metrics use fractional units in storage and explicitly labeled percentages in
display. The composed stopping/fitness scalar never replaces these fields.

### 4.2 Paired generalization comparator

Do not average opaque lexicographic rank encodings. Define
`paired_generalization_weekly_v1`:

1. fail closed unless safety and split-specific activity eligibility hold on
   both members of the pair;
2. compute the same common-scale robust weekly utility on each split;
3. use

```text
paired_score = 0.5 * (score_a + score_b)
               - beta * abs(score_a - score_b)
```

4. preserve deterministic tie breaks using minimum split score, mean weekly
   return, drawdown, cost and stable candidate identity; and
5. emit `score_a`, `score_b`, mean, gap, penalty, eligibility and all raw facts.

For L1, the pair is `train_monitor` and `inner_validation`. For L2, the pair is
`inner_validation` and `outer_validation`. `beta`, minimum trade/activity
requirements and robust-week aggregation are typed contract fields. Initial
values come from the current evidence contract; bounds are calibrated, never
silently defaulted.

## 5. L1 Stopping and Curriculum Program

### 5.1 Stopping contract

- `max_epochs=2000` is a safety ceiling, not the intended stopping point.
- `l1_patience=60` and a 40-checkpoint minimum floor are initial evidence-backed
  values from document 04; they remain visible and later tunable.
- One checkpoint interval is one derived pass-equivalent over valid fit
  transitions, not an unexplained 20,000-step constant.
- Improvement uses `paired_generalization_weekly_v1` and a scale-explicit
  `l1_min_delta`.
- Activity-ineligible patience is independent and cannot consume improvement
  patience.
- Best-checkpoint restoration is mandatory.
- Sealed test is structurally inaccessible.

For a two-phase arm, 2,000 is the global maximum number of pass-equivalent
checkpoints across both phases, not 2,000 per phase. `phase1_max_fraction` and a
reserved normal-phase minimum are explicit. The historical 4/14 allocation is
the initial range anchor, not a selected optimum. A bounded one-seed cap probe
may verify that stopping occurs before the cap; it cannot choose the winner or
enter performance aggregation. If the cap binds, widen it before the four-seed
decision rather than interpreting truncation as convergence.

### 5.2 Easy-phase stopping

Easy mode relaxes only training dynamics. Losses still enter reward/metrics;
normal-realistic evaluation never relaxes solvency or costs.

An easy checkpoint is selected using:

- the easy `train_monitor` utility; and
- normal-realistic `inner_validation` utility.

It must be a trained epoch (`epoch > 0`) with demonstrated policy-parameter
change. Epoch-zero warm starts may be logged as baselines but cannot become an
easy treatment handoff. Phase-1 selection never consults phase-2 outcomes.

### 5.3 L1 comparison

Use four paired seeds and one shared anchor per seed. The primary factorial is:

| Factor | Levels |
| --- | --- |
| phase-1 dynamics | normal, easy |
| normal-phase LR multiplier | 1.0, 0.3 (M0-supported range point) |

All cells use the same:

- source, nested dates, observation and fixed 83-feature control;
- maximum total interaction cap;
- checkpoint cadence and paired stopping rule;
- phase-boundary replay/optimizer treatment;
- phase-2 normal-realistic dynamics; and
- final outer-validation procedure.

The normal/easy arms must share the same phase boundary. If replay/optimizer is
reset, it is reset in both; if preserved, it is preserved in both. The current
repair plan's matched-boundary requirement remains binding.

The decision packet reports both effectiveness and compute efficiency. A
matched-realized-compute sensitivity may be run after the primary result, but
may not rewrite the primary allocation.

### 5.4 L1 decision

The easy curriculum enters the default L1 recipe only if:

- at least three of four paired seeds complete valid artifacts;
- the paired outer-validation effect favors it without a material safety,
  activity or tail-risk regression;
- the effect is not explained solely by extra interactions or phase reset; and
- artifact/action replay proves the selected trained handoff was used.

Positive profit is not an eligibility gate. A null or negative curriculum
effect is a valid result and freezes normal-only for the next domain.

## 6. L2 Curriculum and Stopping Program

### 6.1 Freeze L1 before L2 attribution

Use the selected L1 recipe from section 5 for every L2 arm. Do not activate an
L1 curriculum gene while asking whether the L2 search curriculum works.

### 6.2 Two L2 arms

**L2-N:** all generations evaluate under normal-realistic training/evidence.

**L2-EN:** early generations use easy training/evidence to evolve and triage
genomes; the surviving population is re-evaluated under normal-realistic
training/evidence before normal generations continue.

Rules:

- L2-N and L2-EN receive the same total candidate-evaluation budget, initial
  genome seeds and population size.
- Easy and normal scores never share one comparable leaderboard or chain
  objective. Stage transition invalidates old easy fitness and re-evaluates.
- Only normal-realistic candidates can become champion, migrate, archive or
  seed a release.
- L2 inherits typed genes. It does not claim inherited SAC weights/topology.
- L2 patience uses paired inner/outer evidence after a minimum generation floor.
- Population diversity, unique-genome count and rejected/ineligible share are
  logged so apparent convergence is not confused with collapse.

### 6.3 Conditional 2x2 interaction

Run only if L1 or L2 shows a useful effect:

```text
L1 curriculum {off,on} x L2 curriculum {off,on}
```

Use the nearest winning settings, equal caps and four paired seeds. This is a
bounded interaction confirmation, not a new unbounded search.

## 7. Sparse Feature-Selection Program

The first curriculum comparison keeps the existing 83 features fixed to avoid
confounding. Feature selection follows immediately afterward and before broad
SAC topology optimization.

### 7.1 FS0 fixed control

Current fixed 83-feature observation, with exact list/hash and no hidden drop.

### 7.2 FS1 inherited L2 mask

Add a hierarchical genome:

- source-family gate;
- feature-family gate;
- optional within-family feature bit;
- mandatory agent/risk/protection state; and
- deterministic repair requiring at least one eligible market feature.

Initialize the population sparsely in the spirit of `fs_neat_nohidden`: each
genome begins with mandatory non-market state plus one randomly seeded eligible
market feature and its ancestor gates, using deterministic campaign seeds.
Mutation may add/drop families or features. Observation shape remains stable:
inactive inputs are causally zero-masked after scaling, and the mask/hash
accompanies every artifact.

Do not reward sparsity at the expense of trading utility. Use active feature
count as a tie-breaker or bounded penalty after eligibility; report both.

### 7.3 FS2 L1 learnable sparse gate

Implement a separate SAC agent plugin with a feature-wise gate shared across
the lookback dimension. A hard-concrete/L0-style or equivalently bounded,
reparameterizable gate is acceptable after deterministic inference and export
tests. Requirements:

- sparse initialization;
- mandatory unmasked state/risk/protection channels;
- explicit gate regularization coefficient and schedule;
- deterministic threshold/frozen mask at inference;
- artifact round trip reproduces gates and actions; and
- no feature may use future information because of the gate.

### 7.4 FS comparison

Compare FS0, FS1 and FS2 under the frozen winning L1 curriculum and equal
compute. If both FS1 and FS2 improve outer validation, run one bounded
FS1+FS2 interaction. The final feature domain later expands from the current 83
columns to every causally materialized source that passes real-time parity;
current technical inputs are not the final universe.

## 8. Ordered L2 Search Stages

The NEAT phase analogy is used as a hypothesis, not as a false equivalence:

| Stage | Active genes | Search behavior |
| --- | --- | --- |
| representation emergence | source/feature masks, context, preprocessing | sparse start, high add/change, low deletion initially |
| capacity/connectivity | encoder family/depth/width/latent/policy topology | higher structural/category change |
| learning dynamics | LR, replay, entropy, batch/update schedule, L1 curriculum | moderate mutation, topology mostly frozen |
| control/execution/risk | action threshold, order family, SL/TP, sizing, cost curriculum | bounded domain mutation |
| maturation | confirmed numeric genes only | no structural/category mutation; smaller numeric perturbations |

Each stage declares local crossover probability, mutation probability, numeric
scale, categorical-change probability, minimum/maximum generations, patience,
diversity floor and active/frozen genes. A stage advance re-evaluates carried
elites under the new objective when difficulty or evidence semantics changes.

Lamarckian weight inheritance between DEAP genomes is deferred. It may be
tested later as an independent line only when observation/topology compatibility
and exact optimizer/replay-state semantics are proven.

## 9. Orchestration and Non-Idle Authority

This document and the owner's approving message are standing authorization to:

- implement and test all contracts above;
- run mechanics smokes;
- execute L1, L2, FS and conditional interaction jobs in the declared order;
- start the next pre-approved job after objective acceptance evidence; and
- use all available workers on one coordinated DOIN chain where the job is a
  distributed campaign.

No repeated owner phrase is required. Independent review verifies evidence; it
does not suspend the entire fleet. While a blocking correction is CPU-side or
awaits review, the orchestrator pulls a compatible pre-approved job from:

1. valid ETH component implementation/CPU tests;
2. feature-gate/mask implementation and local fixtures;
3. data-prefix/materializer verification;
4. Paper/Demo evidence ingestion and parity analysis; or
5. another owner-approved research queue item that cannot contaminate this
   domain.

Owner action is still required for real capital, secrets, paid/legal
commitments, destructive history changes, protected-test opening and changes to
mission or mandatory risk/protection invariants.

## 10. Execution Order

1. Correct findings 166-169 and preserve the invalid-successor quarantine.
2. Implement nested splits, causal prefixes and paired L1/L2 evidence.
3. Replace validation-only lexicographic stopping in the new domain.
4. Correct the easy handoff and matched phase boundary (159-161).
5. Run mechanics smoke; exclude it from all scientific aggregation.
6. Execute the full four-seed L1 factorial on the declared ETH chronology.
7. Freeze the L1 recipe and execute L2-N versus L2-EN.
8. Run the conditional curriculum 2x2 if triggered.
9. Implement and compare FS0/FS1/FS2, then the conditional FS interaction.
10. Add stage-local mutation/maturation to the dedicated component domains.
11. Continue document 33's component roadmap and restricted integration.
12. Open 2025 once for the frozen ETH release, never during development.

## 11. Acceptance

The program is complete when every executed cell has loadable model weights,
resolved hyperparameters, feature/gate artifacts, raw metrics, chronological
weekly vectors, code/data/config hashes, resource facts and DOIN lineage; the
scheduled comparisons have an unambiguous typed outcome; and the winning ETH
contract can be reproduced by inference and Paper/Demo shadow parity.

## 12. L1 Decision Result and Immediate Mechanism Work (2026-08-10)

Decision identity `2de49ea9225e2baf` completed 16/16 and is independently
accepted as `INCONCLUSIVE`: all 16 cells are valid, all 16 are inactive and no
paired effect is estimable. The sealed/replica digest is
`f3bb41516f8f3bb9b458c345aae3c1f261cc9688bece697cadc898d60401d374`.

This result inserts one diagnostic step before execution-order item 7. Run the
bounded M0-to-L1 one-change-at-a-time ladder specified in
`../handoffs/MUSASHI_TO_GENERAL_SATOSHI_III_L1_RESULT_WPB_AND_MECHANISM_ORDER_2026_08_10.md`.
Its purpose is to identify the first protocol transition that removes activity;
it neither selects a champion nor opens a broad hyperparameter sweep.

The ladder is covered by the standing non-idle authority in section 9. Its
materialization and dispatch are not an owner gate. L2-N/L2-EN remains queued
behind a reproducible activity-bearing L1 recipe.

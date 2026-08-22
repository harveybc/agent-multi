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

## 12. 2026-08-11 Mechanism-Ladder Result and Phase-1 LR Amendment

The seed-101 M0-to-L1 ladder is preserved under diagnostic identity
`97c0bb29e82dfea3`. It does not select a curriculum winner. It establishes the
following narrower facts:

1. The v3 M0 semantics selected and transferred the epoch-zero anchor and the
   resulting normal policy remained active.
2. The v4 semantics transferred a genuinely easy-trained epoch-one policy.
3. All easy-trained epoch-one policies already emitted near-constant actions
   with magnitude below `0.015`; their immediate normal probes traded zero
   times under threshold `0.1`.
4. Normal training did not restore activity within the diagnostic budget.
5. Replay and optimizer state were reset in every arm and therefore are not
   factors in the next experiment.

The phrase "boundary handoff" is insufficiently precise because the v3/v4
configuration field also changes checkpoint eligibility, selection objective,
probe gating and fallback. The accepted mechanism label is
**phase-1 checkpoint-selection/handoff bundle with pre-existing action-amplitude
collapse exposed by the normal deadband**.

Before another decision-bearing L1 run:

- replay one identical post-easy artifact under action thresholds `0.0` and
  `0.1` without learning;
- record raw action variance/range/quantiles, threshold-crossing fraction and
  deterministic observation sensitivity;
- classify checkpoint handoff viability explicitly; and
- require custody/load proof for every terminal arm, including inactive arms.

The prior L1 factorial varied phase-2 normal LR while phase-1 LR remained
`1e-4`. The next bounded factorial therefore crosses:

```text
phase-1 dynamics {normal,easy} x phase-1 LR {1e-4,3e-5}
```

Phase-2 normal LR is fixed at `3e-5`; all other boundary, data, model, cost,
protection and stopping facts remain fixed. Seeds 101/202/303/404 are pinned
one per physical GPU, and every seed executes all four cells on that same GPU.
This estimates phase-1 LR, difficulty and their interaction without mixing in
hardware identity.

A one-pass mechanics screen precedes full spending. If every combination is a
typed constant/below-threshold collapse, return
`PHASE1_LR_REGION_COLLAPSED`. If a viable region exists, execute the full
document-38 L1 stopping contract (`max_epochs=2000`, patience 60, floor 40,
paired train-monitor/inner-validation selection, best restoration and outer
validation). Only a viable easy result can activate the later bounded
`LR_easy x LR_normal` response surface.

This amendment and the owner's standing approval authorize implementation,
screening and the conditional full run without another phrase. Audit waiting
does not suspend unrelated live evidence or compatible approved compute.

## 13. 2026-08-11 Electrical-Outage Recovery Amendment

The long electrical outage preserved the completed ladder collection: Omega
and the independent Dragon replica recomputed the same tree digest
`cdb6ef9947887992fc0a133a8c66adb76d64a4484cccb5cfc9f63fbea1c2ed8e`.
The interruption exposed three operational gaps that are now part of the
execution contract:

1. a running supervisor is not evidence of GPU availability or useful work;
2. a running Linux bridge/VM is not evidence that a broker terminal is logged
   in and producing fresh direct facts; and
3. a kernel update must not become the active boot kernel on a GPU worker
   without a matching NVIDIA module and post-boot UUID/framework verification.

The detailed recovery and prevention order is
`docs/handoffs/MUSASHI_TO_GENERAL_SATOSHI_III_POST_OUTAGE_RECOVERY_ORDER_2026_08_11.md`.
Its runtime facts are preserved in
`docs/audits/evidence/MUSASHI_POST_OUTAGE_RUNTIME_FACTS_2026_08_11.json`.

GPU dispatch is fail-closed per host and never falls back silently to CPU. A
failed host does not stop compatible work on healthy workers. IBKR and MT5
require direct post-login reconciliation after reboot; stale last-known facts
are never treated as current protection evidence. Once the four expected GPU
UUIDs return, the same-artifact replay and phase-1 difficulty x phase-1 LR
mechanics screen proceed under section 12 without another owner phrase.

Gamma also enters the next long run through a storage-budget gate: its root
volume was observed at 89% usage after reboot, dominated by a 219 GB historical
pre-trading-stack backup. That tree is preserved until a content inventory,
independent replica and owner-reviewed disposition prove what is unique versus
reproducible. No result, model or OLAP database is deleted to manufacture free
space.

## 14. 2026-08-11 Post-Outage Execution-Identity Correction

Independent pre-launch reproduction found that the P1LR materializer recorded
the nested split SHA in metadata without passing the nested contract to the
executing pipeline. The resulting config used the legacy split path and
validation-only metric. No P1LR cell had launched, so no output needs to be
discarded; the experiment identity is replaced before first compute.

The executable P1LR contract now requires these exact roles:

| Role | Scored rows | Use |
| --- | ---: | --- |
| fit_train | 11,509 | gradient updates |
| train_monitor | 2,190 | paired in-sample stopping member |
| inner_validation | 2,190 | paired held-out stopping member |
| outer_validation | 2,196 | one final truth evaluation |
| sealed_test | inaccessible | release only |

The screen verdict also requires an external 16-of-16 replica load proof as a
real boolean gate. A string saying that a collector is required is not proof.
The conditional long decision runner must exist before WP4 is considered
complete, although it runs only after a viable corrected screen.

The exact audit and executable order are:

- `docs/audits/AUDIT_SATOSHI_III_POST_OUTAGE_WP2_WP5_2026_08_11.md`
- `docs/handoffs/MUSASHI_TO_GENERAL_SATOSHI_III_POST_OUTAGE_224_230_CORRECTION_AND_DISPATCH_ORDER_2026_08_11.md`

Standing authority remains unchanged: once deterministic split, custody and
GPU preflights pass, the four-worker mechanics screen starts without another
owner phrase. Review proceeds in parallel; it does not create idle time.

## 15. 2026-08-12 Decision-Run Execution Correction

The corrected mechanics screen is accepted and establishes a viable phase-1
LR region at `3e-5` under both easy and normal dynamics across four seeds. The
active long run is retained as diagnostic compute while three execution defects
are corrected:

1. causal-prefix rows are materialized but not excluded by the internal
   train-monitor/inner-validation selector;
2. typed inactive cells cannot publish a decision record because the runner
   requires a best checkpoint; and
3. status/recovery observe the screen root while decision mode runs under a
   different root through non-durable `nohup` processes.

No output under decision identity `1434685bfdf52911` is promotion-eligible.
The corrected run uses the same original per-seed anchors under a new identity;
it never warm-starts from the diagnostic terminals. Replacement is one worker
at a time after a corrected four-worker smoke so the standing anti-idle
directive remains satisfied.

The exact audit and execution order are:

- `docs/audits/AUDIT_SATOSHI_III_RETURN_224_230_2026_08_12.md`
- `docs/handoffs/MUSASHI_TO_GENERAL_SATOSHI_III_224_233_CORRECTION_AND_NONIDLE_ORDER_2026_08_12.md`

The final decision report separates raw effect magnitude, sign consistency and
predeclared practical materiality. An inactive arm is a non-promotable measured
outcome, never a harness crash and never an invented zero-performance score.

## 16. 2026-08-12 Exact-Horizon Replacement Run

The 231-233 return audit found two additional execution defects before the
conditional long run: an active policy finishing on its best checkpoint was
rejected when terminal and best bytes matched, and Backtrader emitted one
synthetic terminal action beyond the input rows. Both are corrected before
further performance evidence is accepted. Nested evaluations now enforce
exact equality between observed scored steps and the verified role manifest;
role-file rewrites also invalidate the manifest-verification cache.

The replacement runtime is pinned uniformly across Omega, Dragon and Gamma:

- `agent-multi@374dd2eb`
- `gym-fx@634c3fd`
- screen identity `886b776e022d0d7c`
- conditional decision identity `2f5054dc59785e2a`

All four seed workers run as rootless `p1lr-screen@` units with GPU-readiness
and mode-bound idle-guard timers. After 16/16 records, the collector must seal
and independently load all terminal artifacts, emit the typed screen verdict,
replicate that verdict to every worker, switch the guard to decision mode and
start only the decision units. Old screen/decision identities remain
diagnostic and are never mixed or used as warm starts.

Before the decision run starts, replay-buffer capacity is bounded by
`p1lr_decision_execution_profile_v1.json`: 40,000 transitions, two complete
20,000-transition pass-equivalents, identical across every seed and host. The
profile hash participates in decision identity and each decision process is
also cgroup-bounded at 5G high/6G maximum/1G swap. This is an operationally
feasible starting profile, not a claim that 40,000 is optimal. Replay capacity
remains an explicit future optimization parameter; neither silent
host-dependent capacity nor an unmeasured fallback is allowed.

The warm-start source is a weight donor, not a second trainer. It is loaded
with replay capacity 1 and transfers no replay entries or optimizer moments;
the target is built separately with the profile's exact 40,000-transition
capacity. This distinction is asserted in transfer evidence and tests so an
archived 200,000-transition champion cannot create a transient 21.80 GiB
allocation before the bounded target starts.

The mechanics screen stays on its already-pinned revision and identity while
it finishes. Its sealed viable verdict is then the parent gate for a newly
materialized decision child identity containing the execution-profile hash.
This declared parent/child boundary avoids recomputing a valid screen while
also preventing the old unbounded decision identity from launching.

Canonical audit:
`docs/audits/AUDIT_SATOSHI_RETURN_231_233_AND_RUNTIME_CORRECTIONS_2026_08_12.md`.

The screen subsequently sealed 16/16 with `SCREEN_VIABLE_REGION`: LR `3e-5`
preserved activity under both phase-1 dynamics while LR `1e-4` collapsed under
both. The long decision child is identity `8cc6ca5e45e4f993`, running on all
four GPUs from original anchors under the bounded replay contract. This result
isolates a viable LR region; easy-versus-normal performance and interaction
remain questions for the decision records, not conclusions from the screen.

Unattended decision restarts execute from a detached, clean worktree pinned to
the experiment's source revision. The canonical repository may advance for
audits and documentation without changing the identity a restarted seed
derives. `pin_p1lr_decision_runtime.sh` materializes and verifies this boundary
on every host; changing the runtime revision is an experiment transition, not
a routine repository update.

## 17. 2026-08-13 Outer-Validation Adapter Correction

The first long decision identity reached its final outer-validation load on
Dragon and Gamma seed 303 and exposed a runtime interface defect: the causal
prefix adapter implemented the Gymnasium reset/step protocol but did not
inherit from `gymnasium.Env`. Stable-Baselines therefore accepted the complete
training path and rejected the selected artifact only when it was reloaded for
the final outer replay. The affected seed attempts are diagnostic failure
evidence: they contribute no cell records and are never retried into the same
scientific collection indefinitely.

`ContextPrefixWrapper` now has an explicit Gymnasium environment identity, and
the regression suite invokes Stable-Baselines' own environment acceptance
boundary. The correction is a source change, so the replacement decision run
uses a new content-derived identity from the same original anchors, contract,
screen verdict and four seeds. No failed terminal, replay buffer or partially
trained policy is a warm start. Omega, Dragon, Gamma 5070 Ti and Gamma 5090
must all report the replacement identity before its records can be aggregated.

## 18. 2026-08-15 P1LR Terminal Decision Result

The phase-1 difficulty x phase-1 learning-rate factorial reached a terminal,
sealed and independently replicated result under decision identity
`c0e53cf18b7d60dd` with 16/16 cell records. This section is the published
evidence required by the 2026-08-15 post-restart order §6.

Machine-readable companion:
`docs/audits/evidence/P1LR_DECISION_FINAL_EVIDENCE_c0e53cf18b7d60dd_2026_08_15.json`.

### 18.1 Identity and superseded output

The terminal identity is `c0e53cf18b7d60dd` under output root
`~/.local/share/agent-multi/p1_difficulty_lr_factorial_20260811_v1_decision/`.

The earlier decision identities `1434685bfdf52911` (§15),
`8cc6ca5e45e4f993` (§16/§17) and `7b55ef7eac30ae6a` are SUPERSEDED and
diagnostic. They are never aggregated and never mixed with the terminal
identity. Directly observed on 2026-08-15: each of the three holds ZERO
`cell_record.json` files on omega, dragon and gamma, so no superseded record
can enter any aggregation even by accident. Their surviving directories are
attempt scaffolding and lock files only.

### 18.2 Formal aggregate outcome, preserved as recorded

The outcome below is emitted by `tools/p1_difficulty_lr_factorial.py
--decision-verdict` and is reproduced verbatim. It is NOT rewritten:

```text
schema                  agent_multi.p1_difficulty_lr_decision_verdict.v1
outcome                 INCONCLUSIVE
activity_classification PARTIAL_ACTIVITY_SURVIVAL
active_cells            8 of 16
process exit code       4  (EXIT_CLASS["INCONCLUSIVE"])
gates                   records_16_16=true, identity_coherent=true,
                        replica_terminal_loads=true
effect_basis            outer_validation mean_weekly_rap (fraction per week,
                        2024 scored window); paired within seed
materiality_rule        an effect is material iff all four per-seed paired
                        effects share one strict sign
paired_effects_available    []
paired_effects_unavailable  ["101", "202", "303", "404"]
imputation_policy       NONE — a cell without comparable active performance is
                        excluded from the paired utility set and its per-seed
                        effect is typed unavailable with the exact reason; zero
                        and sentinel values are never imputed (finding 232)
outcome_rationale       partial activity survival: 8 of 16 cells are inactive
                        [...], so the paired effect(s) for seed(s) ['101',
                        '202', '303', '404'] do not exist. No zero and no
                        sentinel is imputed for a missing active pair; the
                        surviving per-seed effects are reported as computed and
                        the document-38 decision is withheld (finding 232)
```

The full 2x2 paired effects (difficulty, phase-1 LR, interaction) DO NOT EXIST
for any seed, because every seed lost both of its `1e-4` cells. That is the
correct typed answer and it stands.

### 18.3 Raw economic evidence, viable `3e-5` stratum

Role: `outer_validation`. Window: 2024-01-01 .. 2024-12-31 20:00, 2,196 scored
rows, 53 evaluation weeks, 256 context rows per role forced-hold and excluded
from every metric. Storage units are fractions (§4.1); percentages are display
only. `Annual return (x52)` and `Annual RAP (x52)` use the runner's declared
`weekly_arithmetic_mean_x_52` convention; `Annualized compounded` is
`(1+total_return)^(365.25/days)-1` over the same window and is shown so the two
annualizations are never confused. RAP is `weekly return - lambda x weekly max
drawdown`. Max drawdown is a fraction of peak equity over the whole window.

| Seed | Arm | Host | Mean weekly return | Annual return (x52) | Annualized compounded | Mean weekly RAP | Annual RAP (x52) | Max drawdown | Trades |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 101 | easy 3e-5 | omega | +0.000145 (+0.0145%) | +0.007524 (+0.7524%) | +0.007409 (+0.7409%) | -0.001570 (-0.1570%) | -0.081638 (-8.1638%) | 0.014826 (1.4826%) | 42 |
| 101 | normal 3e-5 | omega | +0.000290 (+0.0290%) | +0.015065 (+1.5065%) | +0.015179 (+1.5179%) | -0.002111 (-0.2111%) | -0.109771 (-10.9771%) | 0.019386 (1.9386%) | 48 |
| 202 | easy 3e-5 | dragon | -0.000181 (-0.0181%) | -0.009435 (-0.9435%) | -0.010051 (-1.0051%) | -0.004288 (-0.4288%) | -0.222953 (-22.2953%) | 0.032122 (3.2122%) | 125 |
| 202 | normal 3e-5 | dragon | -0.000181 (-0.0181%) | -0.009435 (-0.9435%) | -0.010051 (-1.0051%) | -0.004288 (-0.4288%) | -0.222953 (-22.2953%) | 0.032122 (3.2122%) | 125 |
| 303 | easy 3e-5 | gamma | -0.000253 (-0.0253%) | -0.013141 (-1.3141%) | -0.013787 (-1.3787%) | -0.004367 (-0.4367%) | -0.227061 (-22.7061%) | 0.033250 (3.3250%) | 90 |
| 303 | normal 3e-5 | gamma | -0.000181 (-0.0181%) | -0.009435 (-0.9435%) | -0.010051 (-1.0051%) | -0.004288 (-0.4288%) | -0.222953 (-22.2953%) | 0.032122 (3.2122%) | 125 |
| 404 | easy 3e-5 | gamma | +0.000561 (+0.0561%) | +0.029188 (+2.9188%) | +0.029571 (+2.9571%) | -0.003254 (-0.3254%) | -0.169197 (-16.9197%) | 0.028380 (2.8380%) | 124 |
| 404 | normal 3e-5 | gamma | +0.000561 (+0.0561%) | +0.029188 (+2.9188%) | +0.029571 (+2.9571%) | -0.003254 (-0.3254%) | -0.169197 (-16.9197%) | 0.028380 (2.8380%) | 124 |

Every active arm has NEGATIVE risk-adjusted performance on 2024 outer
validation. Nothing in this table is promotable and nothing here is a champion.

### 18.4 Typed inactivity of both `1e-4` arms

All eight `1e-4` cells are typed inactive under BOTH phase-1 difficulties. Per
finding 232 an inactive cell is a MEASURED OUTCOME with full custody, not a
harness crash; it receives exactly one final outer evaluation OF ITS TERMINAL
artifact as diagnostic truth, and it contributes NO utility. Zero is never
imputed as performance, and the zero trade count below is an activity fact, not
a return.

| Seed | Arm | Host | Activity status | Typed cause | Trades on diagnostic outer | Artifact evaluated | Performance value | Terminal sha256 (12) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 101 | easy 1e-4 | omega | `inactive` | `no_activity_eligible_checkpoint` | 0 | `terminal` (diagnostic only) | none — not imputed | `2ba440c20be9` |
| 101 | normal 1e-4 | omega | `inactive` | `no_activity_eligible_checkpoint` | 0 | `terminal` (diagnostic only) | none — not imputed | `d3bd25cfdab9` |
| 202 | easy 1e-4 | dragon | `inactive` | `no_activity_eligible_checkpoint` | 0 | `terminal` (diagnostic only) | none — not imputed | `4cf965407d3c` |
| 202 | normal 1e-4 | dragon | `inactive` | `no_activity_eligible_checkpoint` | 0 | `terminal` (diagnostic only) | none — not imputed | `01a7da0c6713` |
| 303 | easy 1e-4 | gamma | `inactive` | `no_activity_eligible_checkpoint` | 0 | `terminal` (diagnostic only) | none — not imputed | `bd933575780b` |
| 303 | normal 1e-4 | gamma | `inactive` | `no_activity_eligible_checkpoint` | 0 | `terminal` (diagnostic only) | none — not imputed | `12951af075a0` |
| 404 | easy 1e-4 | gamma | `inactive` | `no_activity_eligible_checkpoint` | 0 | `terminal` (diagnostic only) | none — not imputed | `1a62bb69bcb8` |
| 404 | normal 1e-4 | gamma | `inactive` | `no_activity_eligible_checkpoint` | 0 | `terminal` (diagnostic only) | none — not imputed | `f762dfda6e72` |

The recorded termination cause is uniform: training ended before an
activity-eligible L1 checkpoint became available; the train-tail and validation
trade gates must both pass; activity-ineligible for 40 consecutive epochs after
epoch 40; the trade gate never passed, so no eligible checkpoint can exist.
Every trained phase-1 checkpoint at `1e-4` is typed `CONSTANT_POLICY` with
`any_action_crosses_phase2_threshold=false` and `probe_trades_total=0`, under
both difficulties. This is the same amplitude-collapse mechanism named in §12,
now measured at decision budget rather than screen budget.

### 18.5 Conditional-stratum report

This report is SEPARATE from §18.2 and does not modify the `INCONCLUSIVE`
enum. It answers exactly two questions.

(a) Was `1e-4` inactive under both difficulties? **YES** — 8 of 8 cells, as
tabulated in §18.4. This is an activity/viability statement, not a performance
statement; no return number is attributed to a `1e-4` arm.

(b) Inside the viable `3e-5` stratum, did easy have a sign-consistent advantage
over normal? **NO.** Paired within seed, easy minus normal, on the same utility
the runner uses (outer `mean_weekly_rap`, fraction per week, 2024):

| Seed | easy mean weekly RAP | normal mean weekly RAP | delta (easy - normal) | delta annual RAP (x52) | Sign | easy/normal trades |
| --- | --- | --- | --- | --- | --- | --- |
| 101 | -0.001570 | -0.002111 | +0.000541015 | +0.028132756 | positive | 42/48 |
| 202 | -0.004288 | -0.004288 | +0.000000000 | +0.000000000 | exactly zero | 125/125 |
| 303 | -0.004367 | -0.004288 | -0.000078993 | -0.004107625 | negative | 90/125 |
| 404 | -0.003254 | -0.003254 | +0.000000000 | +0.000000000 | exactly zero | 124/124 |

Directional consistency and practical materiality are reported separately:

- **Directional consistency**: one positive seed, one negative seed, two seeds
  exactly zero. The declared rule — all four per-seed paired deltas share one
  strict sign — is NOT satisfied. Easy has no sign-consistent advantage.
- **Magnitude**, reported without any consistency claim: median delta 0.0,
  mean +0.000115505, range -0.000078993 .. +0.000541015 fraction per week.
- **Practical materiality**: NOT ASSESSABLE. No practical-materiality
  threshold was predeclared for this delta. A merely nonzero sign-consistent
  delta would not be "material" without such a threshold, and this delta is not
  even sign-consistent. No materiality claim is made in either direction.

Conclusion: freezing L1 to `normal_realistic` at `3e-5` for the L2 program is
justified by VIABILITY and SIMPLICITY — `1e-4` produces no usable policy at all,
and easy buys no demonstrated advantage — never by a demonstrated performance
advantage of normal over easy.

### 18.6 Custody, replication and digests

Two INDEPENDENT collections were fetched from the contract-assigned hosts,
sealed into separate roots and replicated to dragon, where all 16 terminal
artifacts were rehashed and really loaded (`stable_baselines3 SAC.load`):

| Collection | Sealed (UTC) | Tree digest | Replica host | Load proofs | Verdict |
| --- | --- | --- | --- | --- | --- |
| A `p1lr_collection_c0e53cf18b7d60dd_20260814` | 2026-08-14T18:16:23Z | `6877945cfbc924f940ce48cd5b46ad56d37339d55590501c488f1cc2077a01cf` | dragon | 16/16 `loads=true` | INCONCLUSIVE |
| B `p1lr_collection_c0e53cf18b7d60dd_20260815_evidence` | 2026-08-15T16:25:49Z | `e21c00c28adc29929c11680ba179ea6310ec1fea5b07f092893d11155e6e8c63` | dragon | 16/16 `loads=true` | INCONCLUSIVE |

The two aggregations agree exactly on `outcome`, `per_cell_metrics`,
`inactive_cells` and `per_seed_paired_effects`; the ONLY differing field is the
collection tree digest. Both sealed trees carry the same 424 files with
identical content except four `seed101/*/heartbeat.json` liveness files that the
worker rewrote between fetches. All 16 `cell_record.json` files, all model
artifacts and all custody bindings are bit-identical. Heartbeats are mutable
telemetry inside the fetched subtree; excluding them from the sealed tree digest
would make a re-collection byte-reproducible and is recommended.

Exact digests:

```text
source        agent-multi d3d9b99034d6d2af9157fa92cabd4ceb7822cd29 (clean,
              detached runtime worktree)
              gym-fx      634c3fd3c344cae3c4048b334158185c8bf4e1ef (clean)
              the four collector/verdict tools are byte-identical between
              d3d9b99 and the aggregation revision, so aggregating from the
              newer checkout is faithful
contract      examples/config/phase_3_eth_sac_dynamics/
              p1_difficulty_lr_factorial_v1.json
              4a4e0f16b7da0783b3a0f3d1336474e8a286ec62acc88963599e762eedd00bd6
profile       p1lr_decision_execution_profile_v1.json
              7606ac12dc6b0f8808e52594ed1d6a090d0dbc7a6a2edd957cdfaec3d7c80831
data          examples/data/project3/ethusdt_4h_tech_stat_full_model_ready.csv
              1b447c66e68495e826c53e2ab2b08ecd3922c8fdc735747628f8d0435ebe440f
split         examples/config/phase_3_eth_sac_dynamics/splits/
              eth_nested_split_contract_v1.json
              2b31b7770f815b75b14d8234961d848787ae7c7fde9c03dbc494480fcb4130c6
              mode l1, metric paired_generalization_weekly_v1, beta 0.25
roles         fit_train        11,509 rows  b9a35d6cef2936979168d28e0b4113d2c6a1a309fd3a60c9ee943ea55a791568
              train_monitor     2,190 rows  f9a0e25a5ee7009fa76bac583766455b43359ea616e68d79cb1ce81bf8cb0c06
              inner_validation  2,190 rows  e36ec652aa2935d7bc0f74e0c0f452688e8eb34d47de5a5dd379ec56f1419bd6
              outer_validation  2,196 rows  2244dfc00efa6f681653425d364e720bb66eb2c25ccc2112b80235519e367a30
              sealed_test       SEALED — no path, no sha, no rows, no dates
anchors       101 cb27375c663819333aa4442d5817c657b19ad0e8b61395b04c5e6212217fbc62
              202 82f0a8b66f8f7e9c35dbd550e143a8aa2e13ce2baef599a03d4b137fa91624ee
              303 2340e4566bdc322ef0cfba2040b5f013960cbd04377d2e879bdc091c26eb7003
              404 b8a00b9a300232547333bd1713bafc71d77d1a263f7c971230f8d9267cbb2076
```

The per-cell `cell_identity`, `resolved_config_sha256`, per-attempt split
manifest sha, best-checkpoint sha and terminal sha for all 16 cells are in the
evidence JSON. The split MANIFEST sha is per-cell by construction (it is
re-materialized per attempt directory); the split CONTRACT sha and every
per-role CSV sha are identical across all 16 records.

### 18.7 The 2025 sealed test remains UNOPENED

Every number in §18.3, §18.4 and §18.5 is 2024 OUTER-VALIDATION evidence under
the L1 nested contract. None of it is 2025 release performance and none of it
may be described as such. All 16 records bind
`nested_role_facts.sealed_test = {status: SEALED, csv_sha256: null,
scored_rows: null, score_start: null, score_end: null, context_rows: null}`, and
a scan of all 16 records for any 2025 date materialization returns zero hits.

### 18.8 Residual doubts

These are stated plainly and are not closed here.

1. **The promoted best checkpoint of every active cell is the phase-1 handoff
   policy; phase-2 normal training contributed nothing.** For all 8 active cells
   the policy-tensor sha256 of the selected best checkpoint is bit-identical to
   `boundary_transfer_evidence.source_policy_tensor_hash`. In every
   `epoch_history`, phase-2 epoch 0 (`warm_start_normal_baseline`) is the only
   epoch whose `early_stop_trade_gate_passed` is true; epochs 1..79 all report
   `composite=-1000000.0`, `composite_raw=0.0` and zero trades, and
   `best_composite` never moves off its epoch-0 value. Consequence: the 2024
   numbers for the `3e-5` arms measure the PHASE-1 policy evaluated under
   normal-realistic conditions. The phase-1 difficulty x LR contrast is still
   the thing that varies, so §18.5 holds, but no claim may be made about
   phase-2 normal training.
2. **Every cell stopped at epoch 80 of a declared 2,000 pass-equivalent
   ceiling.** All 16 cells report `stop_reason=
   activity_stop_no_eligible_checkpoint` at exactly 80 epochs and 1,579,000
   gradient updates. The binding stop was the activity patience
   (`l1_activity_patience=40`, start epoch 40), not the paired-generalization
   patience (60, floor 40, max 2,000). About 4% of the declared per-cell budget
   was spent. This is a valid measurement of the AS-RUN stopping contract; it is
   not evidence about the full §6 2,000-checkpoint contract.
3. **The four seeds are not four independent outcomes.** Only 5 distinct outer
   weekly return vectors exist among the 8 active cells. `seed202/P1E_LR3E5`,
   `seed202/P1N_LR3E5` and `seed303/P1N_LR3E5` share one bit-identical 2024
   weekly P&L vector (125 trades) despite distinct model artifacts on different
   hosts; `seed404/P1E_LR3E5` and `seed404/P1N_LR3E5` share another (124
   trades). Distinct weights are collapsing onto the same discretized action
   sequence, which is why two of the four paired deltas are exactly 0.0.
   Effective replication is weaker than n=4 and the sign-consistency test in
   this stratum is underpowered. This weakens any future POSITIVE claim more
   than it weakens the present negative one.
4. **Telemetry naming wart.** Phase-1 per-checkpoint telemetry labels every
   phase-1 epoch `checkpoint_source='easy_training_epoch'` even in
   `normal_realistic` cells. The factor itself is correctly applied and
   separable: `phase1_mode`, `resolved_config_sha256` and
   `phase1_artifact_sha256` all differ per cell.
5. **Collection tree digests are not byte-reproducible** across re-collection
   because mutable heartbeat telemetry lives inside the sealed subtree (see
   §18.6).

### 18.9 What this authorizes

L1 is frozen to `normal_realistic` with phase-1 learning rate `3e-5` for the L2
program of §6, on viability grounds. The formal P1LR outcome remains
`INCONCLUSIVE`; the 2x2 paired effects do not exist and are not claimed. Before
the L2 comparison consumes this frozen L1, residual doubts 1 and 2 must be
resolved or explicitly accepted in writing, because an L1 whose phase-2 stage
contributes nothing is a weak foundation for attributing L2 effects.

## 19. 2026-08-15 Stopping-Contract Correction and Dead-Actor Finding

The terminal P1LR decision identity `c0e53cf18b7d60dd` sealed 16/16 records
with `PARTIAL_ACTIVITY_SURVIVAL`. An independent evidence review raised two
HIGH doubts. Both are resolved here. This section replaces the §5.1 stopping
sentence and amends §12's mechanism label.

### 19.1 The stopping contract §5.1 should have said (corrected text)

Three rules can end L1 training, and the contract declares all three or it is
incomplete:

1. `l1_early_stop` — improvement patience `l1_patience=60` starting at floor
   `l1_patience_start_epoch=40`, measured on `paired_generalization_weekly_v1`.
   It advances ONLY on activity-eligible epochs. A policy that never passes the
   trade gate can never consume improvement patience, so this rule cannot
   terminate an inactive arm at all. Earliest possible stop: epoch 100.
2. `activity_stop` — activity-ineligible patience `l1_activity_patience=40`
   starting at `l1_activity_patience_start_epoch=40`. Independent of, and never
   charged to, improvement patience. Earliest possible stop: **epoch 80**.
3. `max_epochs_budget` — the 2,000 pass-equivalent global ceiling, a safety
   ceiling and never the intended stopping point.

The earliest effective stop is therefore **epoch 80 — 4% of the declared
ceiling — and that is the contract**, not a truncation. The prior text named
only the paired rule and left the reader to derive epoch 80 from two knobs
buried in `stopping_knobs`. Every future decision contract states
`effective_stopping_rules.terminators` and
`effective_stopping_rules.earliest_stop_epoch` explicitly, and
`app/stopping_contract.py` REFUSES to start a decision seed whose effective
rules differ from its declared ones.

### 19.2 Doubt B verdict — correct fail-fast, wrong contract text

The activity stop did not pre-empt the paired comparator; the paired comparator
was structurally inert. `_update_l1_checkpoint_state` returns `no_improve`
unchanged for any epoch that fails the trade gate, and every terminal record
shows `l1_patience_used = 0` and `l1_patience_eligible = false` at epoch 79.
With the activity stop disabled, every cell would have run to the ceiling and
selected the SAME checkpoint at ~25x the compute. The behaviour is correct; the
declaration was not.

One real (already corrected) defect is visible in the sealed records: all
sixteen cells carry `stop_reason = "activity_stop_no_eligible_checkpoint"`,
including the eight ACTIVE cells that did hold a best checkpoint.
`_activity_stop_disposition` (agent-multi@2f531780, 2026-08-14) now emits
`activity_stop_after_best_checkpoint` for that case. The pinned decision
runtime predates it, so the label in the sealed records is stale; the run is
not affected.

### 19.3 Doubt A verdict — confirmed, with the mechanism named

Phase 2 contributed NOTHING to any selected artifact. Recomputed read-only from
the sealed zips, the seed-101 active cells' `model.zip` policy-tensor digests
are bit-identical to `boundary_transfer_evidence.source_policy_tensor_hash`
(`d9298ab7…` for P1E_LR3E5, `33edd255…` for P1N_LR3E5). The reported 2024
outer-validation numbers measure the PHASE-1 handoff evaluated under normal
conditions after 1,579,000 phase-2 gradient updates changed nothing.

The cause is not the deadband, the learning rate or the epoch budget. It is a
**dead first hidden layer in the SAC actor**. For every scored step of every
split, `val_action_raw_min == val_action_raw_max` and `val_action_raw_std` is
exactly `0.0`: the actor is a CONSTANT FUNCTION of the observation. That
constant is reproducible from the weights alone as
`tanh(W_mu · ReLU(b_latent2) + b_mu)` — i.e. with the first layer's ReLU output
identically zero — and it matches the recorded per-epoch action to float32
precision in all four seed-101 cells (for example P1E_LR1E4 predicts
`-0.00099396764` against the recorded `-0.000993967056`).

Replaying the real `inner_validation` observations against the sealed
artifacts: the phase-1 handoff has **21/256** live first-layer units and emits
actions in `[0.034, 0.101]`; the phase-2 terminal has **0/256** and emits the
single constant `-0.001271`. Mean first-layer pre-activation is `-63.8` at the
handoff and `-72.8` at the terminal. The driver is the observation contract:
`include_price_window: true` injects 64 UNNORMALIZED dimensions (raw ETH prices
around 1,742 and raw price diffs) into an otherwise rolling-z-scored,
±10-clipped 2,724-dimension observation. The unnormalized block dominates the
first layer and pushes it into the dead-ReLU regime.

A dead ReLU layer has exactly zero gradient through the observation path. The
collapse is therefore IRRECOVERABLE by construction, which falsifies every
remaining hypothesis: LR `1e-4` and `3e-5` behave identically; the action
amplitude shows no trend across 79 epochs (first-20 mean `0.00162`, last-20
mean `0.00168`, all-time max `0.0094` against a `0.1` threshold); and no
checkpoint bookkeeping bug exists — the warm-start baseline floor correctly
protected the handoff from being replaced by a collapsed epoch-1 policy.

§12's label "pre-existing action-amplitude collapse exposed by the normal
deadband" is refined to **dead-ReLU actor collapse driven by an unnormalized
price block in the observation contract, observed as a constant policy far
below the deadband**.

### 19.4 Consequence for L2

The L1 recipe measured by identity `c0e53cf18b7d60dd` is a PHASE-1 measurement
only. Its phase-2 normal-realistic fine-tuning stage is not merely
uninformative — it destroys the policy in its first epoch and is provably
unable to recover. Freezing that recipe for L2 freezes a stage that cannot
contribute, so the L2 arms would differ only in phase-1 treatment while burning
four GPUs on phase-2 compute that is guaranteed dead. The observation contract
is corrected before L2 attribution, or L2 declares in writing that its phase-2
stage is a known no-op.

## 20. 2026-08-15 Disposition Qualifier and Withdrawal (Order §4)

APPEND-ONLY. Nothing above this section is rewritten. The formal §18.2
aggregate outcome of decision identity `c0e53cf18b7d60dd` REMAINS exactly as
recorded: `INCONCLUSIVE`, `PARTIAL_ACTIVITY_SURVIVAL`, 8 of 16 active cells.
The historical enum is not edited.

### 20.1 Qualifier added to the v1 P1LR result

Per the auditor's scientific disposition (order 2026-08-15 §4; audit
`AUD-P1LR-20260815-235`), the preserved collection `c0e53cf18b7d60dd` and the
whole §18 result carry the additional qualifier:

```text
INVALID_FOR_L1_RECIPE_SELECTION_OBSERVATION_CONTRACT_235
```

Reason, independently reproduced by same-weight counterfactual replay on the
sealed artifacts (audit §2): every v1 terminal policy trained under the
defective 2,724-input observation whose 64 unnormalized raw-price dimensions
(32 raw ETH closes + 32 raw diffs) drove the actor's first layer into the
dead-ReLU regime. Both learning-rate arms converged to the same constant
terminal policy (`-0.001271069`, 0/256 live first-layer units) under that
representation, so the collection cannot choose the learning rate or the
difficulty curriculum. The numbers in §18.3-§18.5 remain preserved evidence
about DEFECTIVE-representation policies; they select nothing.

### 20.2 The §18.9 freeze is WITHDRAWN

The §18.9 conclusion "L1 is frozen to `normal_realistic` with phase-1
learning rate `3e-5` for the L2 program" is **withdrawn** (order §4). No L2
run may be dispatched against that frozen recipe or its 2,724-input anchors
(`anchor_seed*.zip` are preserved as diagnostic evidence only, order §3).

The next scientific job is the corrected-observation L1 factorial:
`examples/config/phase_3_eth_sac_dynamics/p1_difficulty_lr_factorial_v2.json`
(2,660-input accepted observation contract; per-seed zero-update genesis
artifacts; new experiment identity, output root and replica root; typed
actor-liveness, action-variation and selected-versus-genesis facts per cell).
L2 remains implemented and parked until corrected L1 produces a defensible
recipe.

Machine-readable companion of this disposition:
`docs/audits/evidence/P1LR_V1_DISPOSITION_QUALIFIER_2026_08_15.json`.

## 21. 2026-08-16 Causal Early-Stopping Correction

APPEND-ONLY. The stopped decision identities that used four phase-1 epochs are
preserved as diagnostic evidence, but they are not evidence for or against an
easy pre-training effect. Four epochs with `easy_patience=10000` ended phase 1
by its hard cap; early stopping could not fire.

### 21.1 Exact question

For each seed and LR stratum, compare:

```text
control:   normal_realistic -> normal_realistic
treatment: easy_chronological_continuation -> normal_realistic
```

The pair shares the same zero-update genesis tensor, chronology, observation,
SAC architecture, LR, timesteps per epoch, phase boundary, replay/optimizer
reset, phase budgets, stopping rule, cost/protection contract, selection roles
and final outer evaluation. Within either LR stratum the only materialized
configuration difference is `phase1_mode`.

The LR strata remain `1e-4` and `3e-5`, but a cell now uses its stratum LR
unchanged in both phases. This permits a separate LR main effect and
difficulty-by-LR interaction without introducing an undeclared LR change at
the phase boundary. A future phase-specific LR schedule is a separate
experiment/DOIN domain; it is not folded into this contrast.

### 21.2 Real stopping and compute

Each phase receives:

- maximum 1,000 epochs;
- 20,000 SAC timesteps per epoch;
- patience 60;
- patience floor 40;
- `min_delta=1e-4` on the declared paired validation utility.

The combined ceiling is 2,000 epochs (40 million timesteps) per cell. The
phase-2 activity-ineligible terminator is disabled for this experiment. A
policy that never becomes eligible runs to the phase ceiling and is recorded
as such; it is not silently discarded around epoch 80. Both phase summaries
persist maximum, epochs run, best epoch, stopped epoch, stop reason, patience,
floor and minimum improvement. Phase 1 additionally records the first and
count of positive monitor-return epochs.

### 21.3 Explicit held-fixed SAC contract

The resolved config pins `MlpPolicy`, `[256,256]`, batch 256, learning starts
1,000, train frequency 1, one gradient step, gamma 0.99, tau 0.005, fixed
entropy coefficient 0.2, target update interval 1, target entropy `auto`, no
SDE, and the decision replay capacity of 40,000. These values are controlled
for this comparison, not claimed optimal; later topology/training domains may
optimize them.

The materializer now returns the already-applied 2,660-input observation
contract (`include_price_window=false`, rolling z-score window 256, clip 10)
rather than returning legacy base values and relying on a later pipeline repair.
Train-monitor 2022 and inner-validation 2023 control stopping; outer-validation
2024 is evaluated once after selection; sealed 2025 is inaccessible.

### 21.4 Interpretation boundary

Easy-positive-return checkpoints are recorded, but positivity is not inserted
as a treatment-only selection gate in this primary contrast. Doing so would
change both the treatment and its censoring rule. After the paired result, a
secondary analysis may test the historical NEAT rule (only hand off a positive
easy checkpoint) as its own declared factor.

## 22. 2026-08-18 Explicit Close and Live-Stationary Observation Amendment

APPEND-ONLY. The 2,660-input correction removed raw prices but still inherited
two semantic defects: a weak action meant hold rather than target flat, and the
legacy agent state included episode steps remaining while unrealized PnL could
not be reconstructed without the removed price window. Those semantics could
not drive an honest continuous Paper/Demo controller.

The replacement experiment is
`p1_difficulty_lr_factorial_20260818_v4_live_state_and_explicit_close`.
Both phases now use `target_exposure_hysteresis_v2`: normal entry threshold
0.10, normal exit threshold 0.02, easy thresholds 0/0, and
`opposite_signal_semantics=close_then_wait`. Existing exposure remains in the
inference loop. Near-zero actions explicitly close/cancel and opposite targets
close before any later reversal. Native SL/TP remains mandatory.

The four state values are now signed position, session-relative equity, true
unrealized PnL from broker/simulator entry price and holding duration capped at
42 H4 bars. Episode steps remaining is prohibited. The resulting dimension is
still 2,660, but same width is not treated as same meaning: the observation
contract hash changed and fresh per-seed zero-update genesis artifacts are
required.

The causal P1LR comparison fixes every parent entry to market and pending TTL
zero. The pre-existing adaptive market/limit/stop router was previously being
selected through plugin defaults rather than the materialized experiment
contract; allowing it here would confound difficulty with fill behavior.
Document 39 owns the downstream order-family and separate entry/exit-model
comparisons. All earlier P1LR artifacts remain diagnostic and cannot be
promoted under this amended action/observation contract.

## 23. 2026-08-22 Plateau scheduler diagnostic disposition

The bounded fixed-versus-plateau screen remains officially `INCONCLUSIVE` and
cannot promote a checkpoint. The exploratory post-intervention signs are all
negative, but audit reproduced an identity bypass in that diagnostic,
prefix-based completion-schema acceptance and incomplete fsync of the canonical
launch artifact. The early-intervention plateau screen is the conditionally
accepted next GPU experiment. Dispatch follows correction and independent
reproduction of PLR-08, REC-05 and REC-04 plus a scheduler timing preflight over
the existing monitor histories.

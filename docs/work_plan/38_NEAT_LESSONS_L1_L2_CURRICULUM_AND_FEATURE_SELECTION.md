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

## 18. 2026-08-15 Stopping-Contract Correction and Dead-Actor Finding

The terminal P1LR decision identity `c0e53cf18b7d60dd` sealed 16/16 records
with `PARTIAL_ACTIVITY_SURVIVAL`. An independent evidence review raised two
HIGH doubts. Both are resolved here. This section replaces the §5.1 stopping
sentence and amends §12's mechanism label.

### 18.1 The stopping contract §5.1 should have said (corrected text)

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

### 18.2 Doubt B verdict — correct fail-fast, wrong contract text

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

### 18.3 Doubt A verdict — confirmed, with the mechanism named

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

### 18.4 Consequence for L2

The L1 recipe measured by identity `c0e53cf18b7d60dd` is a PHASE-1 measurement
only. Its phase-2 normal-realistic fine-tuning stage is not merely
uninformative — it destroys the policy in its first epoch and is provably
unable to recover. Freezing that recipe for L2 freezes a stage that cannot
contribute, so the L2 arms would differ only in phase-1 treatment while burning
four GPUs on phase-2 compute that is guaranteed dead. The observation contract
is corrected before L2 attribution, or L2 declares in writing that its phase-2
stage is a known no-op.

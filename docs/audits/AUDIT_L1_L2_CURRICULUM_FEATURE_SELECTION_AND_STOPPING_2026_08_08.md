# Audit: L1/L2 Curriculum, Feature Selection and Stopping Contract

Date: 2026-08-08 America/Bogota
Auditor: General Musashi, independent verifier and ML architecture lead
Audited head: `agent-multi@0c095284a0d631d7a65ba66bb61b957dfd86ef00`
Scope: executable training/optimizer contracts and work-plan claims
Runtime mutation by this audit: none
Protected-test access by this audit: none

## 1. Verdict

The existing ETH history contains useful mechanism evidence, but the current
`N14`/`EN4_10` decision contract is not an adequate final answer to the owner's
question. It uses a fixed 14-epoch budget with early stopping disabled, while
the executable lexicographic L1 path monitors validation alone and the L2
optimizer scores one selected split. The current M1 program therefore does not
test train-plus-validation stopping at either optimization level, does not
separate L1 curriculum from L2 curriculum, and has no online learned feature
selection analogous to the owner's successful FS-NEAT experiment.

The historical `N14`, `EN4_10`, `E4` and M0 files remain evidence. They must not
be deleted or relabeled. Their correct role is implementation/mechanism
diagnostics and prior-range evidence, not the final decision-bearing comparison.

The correct next program is staged:

1. repair nested chronological evidence and paired L1 stopping;
2. compare normal-only against easy-to-normal inside L1 with meaningful data,
   high safety ceilings and stopping rather than manually selected epochs;
3. freeze the winning L1 recipe and compare normal-only against staged
   easy-to-normal allocation at L2;
4. run one bounded 2x2 confirmation only if either curriculum axis survives;
5. compare fixed features, inherited L2 sparse masks and an L1 learnable sparse
   gate before topology/risk integration; and
6. retain normal-realistic validation and the sealed test as immutable truth.

L1 and L2 curricula are not mutually exclusive. They act on different state:
L1 changes one candidate's gradient-training dynamics and weights; L2 changes
which typed genomes are generated, inherited, promoted and evaluated. They must
first be estimated separately because enabling both at once destroys
attribution.

## 2. Reproduced Code and Data Facts

### 2.1 Data currently available

The pinned ETH file has 18,085 rows at 4-hour frequency and SHA-256
`1b447c66e68495e826c53e2ab2b08ecd3922c8fdc735747628f8d0435ebe440f`.
A nested year split is possible without buying or inventing data:

| Role | Dates | Rows |
| --- | --- | ---: |
| L1 fit | 2017-09-28 04:00 through 2022-12-31 | 11,509 |
| L1 inner stop/selection | calendar 2023 | 2,190 |
| L2 outer validation | calendar 2024 | 2,196 |
| sealed release test | calendar 2025 | 2,190 |

This is sufficient for the bounded ETH curriculum and feature-selection
program. The defect is not a lack of ETH rows. The defect is how those rows and
budgets are currently used.

### 2.2 Current L1 behavior

`pipeline_plugins/rl_pipeline_with_validation.py:292-333` computes a
train-tail/validation mean and gap penalty for ordinary scalar metrics, but
explicitly replaces that pair with validation alone for
`lexicographic_weekly_v1`. `:336-386` then uses that result for checkpoint
patience. The default training tail is seven days (`:451-482`, `:600-609`), only
42 four-hour bars before fallback logic, while the declared scaling context is
256 bars.

`pipeline_plugins/rl_pipeline_with_solvency_curriculum.py:195-398` has a
separate easy-phase loop. Its checkpoint score is easy-probe economic equity;
normal train-tail and validation are activity gates, not the paired performance
criterion. Historical M0 additionally demonstrated that epoch-zero fallback
could become the easy handoff; findings 159-165 already cover that defect class.

### 2.3 Current L2 behavior

`optimizer_plugins/default_optimizer.py:1206-1369` accepts
`ga_fitness_split in {train,val}`. It trains a candidate and replaces the
summary with one frozen validation inference when `val` is selected. It does
not compute a typed inner/outer train-plus-validation candidate objective.

`optimizer_plugins/default_optimizer.py:831-863` supports ordered active gene
groups, generation counts and patience. `:380-498` applies global `ga_cxpb` and
`ga_mutpb`; it does not support stage-local mutation rates, mutation magnitude,
categorical-change rates or diversity floors. It inherits config genes, not
SAC policy topology or learned weights.

### 2.4 Current feature selection

Documents 17 and 18 preserve useful train-only pre-screening and a future
feature-mask genome, but the active ETH M1 observation is a fixed 83-feature
tensor. There is no L1 learnable sparse input gate, and M1 does not evolve an
inherited feature mask. The current statement "membership optimizable later"
in document 34 is accurate; a claim that online selection has already happened
would not be.

## 3. Findings

### AUD-F1-20260808-170 - S2 - Fixed 14 epochs bypass the required L1 stopping experiment

Documents 33 and 34 define `N14` and `EN4_10` with 14 manually allocated
epochs, 20,000 steps each and no early stopping. That can expose gross collapse
or handoff defects, but it cannot determine where training should stop or test
the owner's train-plus-validation stopping result. It also makes a hand-picked
epoch count part of the treatment.

Required correction: retain those runs as historical diagnostics, then replace
the decision run with high safety ceilings, a minimum observation floor and
paired L1 early stopping. Epoch length must derive from valid fit transitions.
Report actual epochs, gradient updates, environment interactions and wall/GPU
time; never let a candidate gain fitness by silently receiving more compute.

### AUD-F1-20260808-171 - S2 - Lexicographic L1 stopping monitors validation alone

The explicit exception at `rl_pipeline_with_validation.py:310-322` sets the
train-plus-validation score equal to the validation order key. This recreates
the validation-overfitting failure the owner already observed in NEAT.

Required correction: introduce a typed paired-generalization comparator on a
common economic scale. Eligibility/safety remains lexicographic, but the
stopping scalar is based on both train-monitor and inner-validation robust
weekly utility, with an explicit generalization-gap penalty. Preserve and
report each raw component; do not average opaque encoded rank keys.

### AUD-F1-20260808-172 - S2 - L2 fitness and patience use one split

The optimizer can score train or validation but cannot consume a nested
train/inner/outer evidence packet. L2 patience therefore stops generations on
one split and can overfit the repeatedly queried outer year.

Required correction: candidate training uses L1 fit plus inner stopping only.
L2 selection receives a typed pair of inner and outer chronological evidence,
with raw weekly vectors, activity and risk facts. Generation patience begins
only after a minimum generation floor and uses the paired objective. Test never
enters L1, L2, migration or patience.

### AUD-F1-20260808-173 - S2 - Warm-up and train-monitor windows are not sufficient for the declared claim

The seven-day default monitor is shorter than the 256-bar scaling context. The
current split implementation also writes each validation slice without its
prior causal context, while document 34 deducts 256 bars from every stream.
That turns a nominal annual validation into roughly 323 scored days.

Required correction: every inner/outer/test split receives a read-only causal
prefix of at least `max(observation_window, scaling_window, feature_lookback)`.
Prefix rows may initialize features and recurrent/stateful transforms, but may
not trade, mutate account state or contribute metrics. Score the complete
declared calendar interval. Use at least the last complete fit year as the
train-monitor series for this 4-hour decision run.

### AUD-F1-20260808-174 - S3 - Online feature selection is absent from the ETH policy path

Offline proxy selection and a fixed 83-feature SAC observation do not reproduce
the mechanism that made FS-NEAT successful: sparse inherited feature access
during controller learning.

Required correction: implement and compare two independent mechanisms with a
fixed-feature control: (a) hierarchical sparse feature masks inherited in the
L2 genome, with stable observation dimensions and repair rules; and (b) a
separate L1 learnable sparse gate shared across each feature's lookback. State,
risk and protection fields are mandatory and never masked. Export the final
mask/gates with the model and enforce numerical parity in inference.

### AUD-F1-20260808-175 - S3 - L2 stages cannot express the proposed maturation schedule

Current stages freeze gene groups but use global crossover and mutation. They
cannot encode the relevant analogue of neurogenesis, synaptogenesis, tuning
and maturation. Calling the existing schedule "incremental complexification"
would overstate it.

Required correction: add stage-local crossover probability, mutation
probability, numeric perturbation scale, categorical-change probability,
minimum generations and diversity telemetry. Use stages for sparse
representation emergence, capacity/connectivity choices, training dynamics,
execution/risk and low-amplitude maturation. Do not claim topology or weight
inheritance unless separately implemented and tested.

### AUD-F1-20260808-176 - S2 - The experiment matrix cannot attribute curriculum benefit to L1 or L2

The work plan currently tests a fixed L1 easy-normal schedule and later admits a
curriculum gene, but has no matched L2 normal-only versus L2 staged comparison
and no interaction confirmation. A positive result cannot answer whether the
benefit came from weight learning, genome search or their combination.

Required correction: execute the sequential L1, L2 and conditional 2x2 program
in document 38. Easy-stage scores cannot compete with normal-stage scores; all
champions must win under normal-realistic evidence.

### AUD-GEN-20260808-177 - S3 - Approved work is still expressed as repeated owner gates

Several handoffs treat already-approved implementation steps as requiring a new
owner phrase. This creates avoidable idle time and obscures the real distinction
between safety authority and experiment execution.

Required correction: this plan revision is the owner's standing authorization
for code, tests, local/DOIN Paper research and the scheduled experiment queue.
Independent verification remains required for evidence claims, but waiting for
review must dispatch pre-approved fallback work. New owner action is reserved
for real capital, credentials, legal/cost commitments, destructive history
changes, protected-test opening or mission/risk-contract changes.

## 4. Acceptance Conditions

Findings 170-177 are accepted only when:

1. exact nested rows, hashes, causal prefixes and scored intervals are emitted;
2. L1 paired stopping is unit-, mutation- and integration-tested for scalar and
   lexicographic objectives;
3. L2 consumes typed inner/outer evidence and cannot fall back to one split in
   the new domain;
4. fixed-epoch smokes are labeled mechanics-only and excluded from decisions;
5. L1-N versus L1-EN completes on four paired seeds with equal caps;
6. L2-N versus L2-EN completes under a frozen L1 recipe and equal total
   candidate-evaluation budgets;
7. any 2x2 interaction run is triggered only by the declared rule;
8. feature-mask and learnable-gate artifacts round-trip and reproduce actions;
9. every result reports same-scale weekly and annual metrics with raw vectors;
10. the 2025 test remains sealed until one release candidate is frozen; and
11. no machine is idled waiting for redundant owner approval or document-only
    ceremony.

The detailed implementation and execution order is:

`docs/handoffs/MUSASHI_TO_GENERAL_SATOSHI_III_L1_L2_CURRICULUM_FEATURE_SELECTION_EXECUTION_ORDER_2026_08_08.md`

The governing research contract is:

`docs/work_plan/38_NEAT_LESSONS_L1_L2_CURRICULUM_AND_FEATURE_SELECTION.md`

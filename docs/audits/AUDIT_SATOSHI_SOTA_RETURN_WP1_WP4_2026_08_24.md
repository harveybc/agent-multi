# Audit: Satoshi SOTA Return WP1-WP4

Date: 2026-08-24
Audited commits: `b20fc3f9..8347de51`
Auditor: General Musashi
Runtime mutation: none

## Verdict

`REVISE_BEFORE_DISPATCH`.

The roadmap direction is accepted: baselines, action contract, adaptation,
capacity-matched architecture, then DOIN. No post-P1 screen is accepted for
execution yet. The defects below must be corrected while P1 continues.

## Findings

### S2 — SOTA-R01: frozen P1 policies introduce temporal lookahead in Screen B

Screen B evaluates frozen P1 champions on three rolling origins including
fit-through-2021/score-2022 and fit-through-2022/score-2023. The P1 policies
were trained using data through 2022 and selected with later development facts.
Inference-only evaluation on earlier origins does not remove that information.

Correction: either train a causally eligible SAC baseline independently at each
origin using only that origin's fit data, or restrict existing frozen P1 policy
evaluation to a clearly labeled 2024 diagnostic. Gate G1 across three origins
requires the former.

### S2 — SOTA-R02: sealed-2025 cannot choose among surviving configurations

The work plan says surviving configurations, plural, are evaluated once on
2025. Comparing them on 2025 would make the sealed test a selection set.

Correction: freeze exactly one finalist and its complete ensemble/seed rule
before release. Evaluate that single frozen system once. If multiple systems
must be reported, their ordering may not change deployment or research choices;
otherwise a new untouched future test period is required.

### S2 — SOTA-R03: Screen A is not a single-factor comparison as written

The four arms change different objects: action mapping, sizing, reward turnover
term, head dimensionality and state-dependent close semantics. "Identical SAC
trunk" does not make their effects identifiable. Deadband calibration also
creates undeclared trials.

Correction: stage the screen:

1. A0 maps the same scalar actor output to sign, continuous target and
   predeclared ternary mappings under one common economic reward/sizing engine.
2. A1 optimizes the surviving mapping's deadband/hysteresis on development only,
   counting every candidate as a trial.
3. A2 compares explicit close/hold or multi-head architecture only after A0/A1,
   capacity-matched and as a separate mechanism.

### S2 — SOTA-R04: retraining cadence is confounded with update method

Screen R allows fresh, warm and shrink-perturb variants "if budget allows".
That is neither fixed nor causally interpretable. Trailing-window length,
updates per refresh, replay handling and optimizer continuity are unspecified.

Correction: first compare cadence with one frozen update contract and equal
compute accounting. Then compare update methods at the winning cadence in a
separate screen. Materialize exact trailing windows, timesteps, replay and
optimizer rules before dispatch.

### S3 — SOTA-R05: source validation is section-level, not claim-level

The validator passes when a section contains one registered source line. It
does not bind each numeric claim to a source/locator, validate locator syntax,
or preserve source version/hash. Therefore it does not satisfy the order's
"every quantitative claim" acceptance statement.

Correction: introduce claim IDs or inline source markers for quantitative
tables and assertions; validate source ID plus non-generic locator. Registry
entries need retrieval date/version and content hash when a local PDF is used.

### S3 — SOTA-R06: warmup re-probe is not independently reproducible

`WARMUP_REPROBE_NESTED_2026_08_24.json` has no command, script, code identity,
config hash, dataset hash or per-feature evidence. It reports
`fit_context_rows: 0` while calling itself the nested context-prefix path. The
conclusion may be correct, but the artifact cannot establish it independently.

Correction: commit the probe, bind all identities, distinguish source-data
zeros from scaler output, report the denominator/features, and test fit plus
each evaluation role with their actual context-prefix lengths.

### S3 — SOTA-R07: statistical procedures are named but underspecified

SPA/DSR and bootstrap intervals require explicit return frequency, benchmark
loss differential, block/bootstrap method, dependence handling and trial-count
scope. Historical OLAP coverage is not proven complete, so "true trial count"
is currently an assertion.

Correction: publish a statistics contract. Treat reconstructed historical trial
count as a documented lower bound until completeness is proved. Predeclare block
length selection and how overlapping folds/seeds are aggregated.

### S3 — SOTA-R08: baseline and architecture contracts retain free choices

The volatility baseline lacks target volatility, estimator, lag and leverage
cap. C2 says TCN "or" GRU; approximate parameter matching has no tolerance.
These choices can be made after seeing results.

Correction: bind every baseline formula before execution. Select C2 before
results and require a declared parameter-budget tolerance, for example +/-5%,
plus matched training-update budget.

## Accepted Work

- Program-level relabel of 2024 as development data.
- Single-asset-first ordering.
- Baselines before architectural expansion.
- Action semantics before retraining and architecture.
- Capacity, FLOPs, latency, turnover and per-bar returns as required evidence.
- No use of sealed 2025 during development.
- Four seeds as mechanism screening rather than champion selection.
- Corrected language for Demo/Paper execution.

## Required Return

Satoshi must return corrected documents, tests for the source validator and
warmup probe, and materialization-negative tests proving future-trained policies
and sealed-2025 cannot enter development screens. No GPU dispatch is authorized
by this audit.


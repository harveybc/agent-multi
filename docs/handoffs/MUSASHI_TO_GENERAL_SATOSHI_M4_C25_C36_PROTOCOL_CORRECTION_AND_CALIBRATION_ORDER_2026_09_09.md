# Musashi to General Satoshi: M4 C25-C36 protocol correction and calibration

Date: 2026-09-09

Priority: P0 CPU work. Start after committing the final C17-C24 packet and
recording its exact final-suite count.

Authority basis:

- owner instruction to keep the data-centric and model-capacity work moving;
- code tip `f0e25dec0f8de217c179282d0d72b9c665e6f7c9`;
- v4 design identity
  `30a86f0924fd7d31abcacf6691f084ea3b68821b1fb9735a82458879cac37e52`;
- independent focal reproduction: 13/13; and
- Musashi audit
  `MUSASHI_AUDIT_M4_C17_C24_2026_09_09.md`.

Recorded disposition:

`M4_C17_C24_REVISE_BEFORE_CALIBRATION`

This order authorizes a v5 successor, corrected DEVELOPMENT rerun and bounded
CALIBRATION execution on CPU. It does not authorize CONFIRMATION, GPU, DOIN
genes, production gates, financial data, venue access or changes to B4/T2/M3.
Preserve v1-v4 and all existing run evidence byte for byte.

## 1. M4-C25: one paired intervention tape and one genesis

Freeze a canonical `association_tape_id` derived from design, generator,
width and model seed. Arm and checkpoint names must not enter this identity.
For one experimental unit, treatment, initialization control and every
checkpoint must consume byte-identical association inputs, labels, ordering
and update-sampling tape.

Generate association inputs from a disjoint draw of the same family-compatible
input process, not standard normal for every family. Generate targets by an
independent permutation or draw from that task family's training target
distribution:

- Boolean tasks retain Boolean support and binary labels;
- temporal tasks retain the declared window process and continuous target
  scale; and
- input-target pairing is independent and blinded while each marginal remains
  compatible with the task.

Persist and re-derive both tape digests. A one-byte difference between arms or
checkpoints refuses before optimization.

Derive one genesis parameter object per `(generator, width, model_seed)`.
`initialization` is that exact object before task learning. Every later
checkpoint descends from it. Report task-training compute separately;
intervention-phase compute and association tapes must be matched across the
paired checkpoint arms. Do not claim equal total history between learned and
untrained checkpoints.

## 2. M4-C26: truthful model and baseline semantics

Choose and materialize one exact task contract before outcomes:

- Boolean families: sigmoid output, binary cross-entropy optimization and
  held-out accuracy; or explicitly amend the design to a linear/MSE surrogate
  and justify it. Do not declare one and run the other.
- Temporal families: linear output and mean-squared-error optimization.

The Boolean baseline is selected from TRAIN but evaluated on EVALUATION: choose
the training-majority class, predict that fixed class on every evaluation row
and compute its evaluation accuracy. The temporal persistence baseline remains
the last input value evaluated on the same EVALUATION rows.

Freeze regressions for every previous Boolean baseline mismatch. Re-adjudicate
the learnability table from corrected raw observations; do not edit aggregate
outcomes.

## 3. M4-C27: causal train/stop/evaluation split and real checkpoints

Supersede the two-way `train/held` bank with three byte-disjoint roles per
generator:

1. TRAIN: optimizer updates only;
2. STOP: early-stopping decisions only; and
3. EVALUATION: learnability, retention and scientific endpoints only.

No EVALUATION value may select a checkpoint, threshold, width, model seed or
budget. Bind all arrays and boundaries in the generator manifest.

Materialize the four declared checkpoints from the same genesis:

- `initialization`;
- `pre_stop`, under an exact predeclared rule tied to the selected stopping
  point;
- `calibration_stop`, selected only from the STOP trajectory under frozen
  cadence, patience and minimum delta; and
- `post_stop_bounded`, after an exact bounded number of additional TRAIN-only
  updates.

Persist parameter, optimizer-state, update-count, STOP-trajectory and parent
checkpoint digests. The verifier must replay each lineage. If the four-way
question is no longer intended, supersede it explicitly before outcomes; never
retain a taxonomy that code does not execute.

## 4. M4-C28: honest v5 chronology and one generator contract

Classify v4 and its run as
`EXPLORATORY_DEVELOPMENT_EVIDENCE_WITH_PROTOCOL_DEFECTS`. Build a v5 successor
that maps every changed field and leaves no accepted v3 mechanic silently
altered.

Commit and push the final v5 design, generator contract, analysis contract,
unit ledger template and executable verifier before computing any v5
DEVELOPMENT or CALIBRATION score. The commit boundary is part of the return.

Reconcile all inherited contradictions:

- one family spelling (`discontinuity` or another single canonical name);
- one generator-id rule;
- one exact index namespace per DEVELOPMENT/CALIBRATION/CONFIRMATION role;
- one optimizer/loss/head contract per task type; and
- one executable wall limit. Use 48 hours only if it is the actual hard limit;
  otherwise amend the affordability statement to the enforced value.

## 5. M4-C29: fixed populations and precision gate

Materialize complete populations before their outcomes:

- DEVELOPMENT: two generators per candidate family/noise cell, as a corrected
  replacement of the exploratory v4 screen;
- CALIBRATION: **16 independent generators per candidate family/noise cell**;
  and
- CONFIRMATION: reserve **48 independent generators per confirmatory cell**,
  but do not generate, score or inspect their outcomes in this order.

Role must enter seed derivation. Reusing an integer index across roles is
permitted only when role-separated bytes and ids are proven; no inherited text
may claim disjoint numeric ranges if code uses role namespaces instead.

Use CALIBRATION to estimate generator-level dispersion after averaging nested
model-seed repetitions. Freeze thresholds and hyperparameters there. The
confirmation population is fixed at 48 rather than selected after calibration.
If a predeclared upper confidence bound for generator-level SD exceeds 6
associations, return `M4_CONFIRMATORY_PRECISION_NOT_SUPPORTED`; do not weaken
the effect size or multiplicity family. Every ineligible family remains as a
typed slot in the frozen family.

## 6. M4-C30: executable M0/M1/M2 prediction ladder

Implement the prediction question before CALIBRATION outcomes. Because the
endpoint can hit the batch cap, use a discrete-time survival formulation over
the 64 acquisition batches instead of silently treating a capped endpoint as
an uncensored scalar.

Use generator-grouped splits only. Freeze:

- M0: family/noise nuisance terms plus parameter count;
- M1: M0 plus checkpoint loss and elapsed task-training updates;
- M2: M1 plus the frozen trajectory and description measurements;
- one model family and regularization-selection rule using DEVELOPMENT only;
- integrated Brier score as primary prediction loss;
- calibration at predeclared acquisition horizons; and
- a paired generator-level M2-minus-M1 decision with measurement time and CPU
  cost reported separately.

M2 advances only if it improves unseen-generator integrated Brier score after
multiplicity correction and is non-inferior in calibration. Descriptor cost is
part of the Pareto report and may not disappear from the comparison. If there
are too few DEVELOPMENT/CALIBRATION generators for this analysis, return an
explicit insufficiency result instead of fitting it.

## 7. M4-C31: restricted endpoint and resource-stop semantics

Define the primary intervention endpoint as the **restricted acquired count**:
`min(true endpoint, 64 batches)`. A cap hit therefore has the observed
restricted value 64 batches while still publishing `CAP_REACHED`; do not claim
the unrestricted endpoint was observed.

Only a cap hit is censored for the optional unrestricted time-to-failure
analysis. `WALL_STOP`, `RSS_STOP`, `STOP_REQUESTED`, numerical failure and
interruption are incomplete/failed scientific units. They remain in the
denominator and never enter a survival test as ordinary right-censoring.

Remove the undefined Gehan fallback. If an unrestricted secondary analysis is
retained, implement and test an exact generator-paired censor-aware method
before CALIBRATION. The restricted endpoint remains the primary analysis in
all cases.

## 8. M4-C32: runner, verification and durable custody

Extend the single runner so the sealed design can enumerate DEVELOPMENT,
CALIBRATION and reserved CONFIRMATION units, while this order permits execution
of only the first two roles.

Required corrections:

- exact recursive schemas for ledger, records, checkpoints, summaries and
  report;
- output root 0700 and every evidence object 0600;
- create each intervention log once with `O_EXCL`, retain the descriptor for
  append/fsync and never reopen after verification;
- a per-batch durable state checkpoint so partial intervention work resumes
  from the verified predecessor rather than restarting and appending;
- replay of screen and intervention units before any resumed work;
- one canonical terminal report included in exact inventory; arbitrary
  `RUN_REPORT*` lookalikes refuse;
- second invocation is either read-only idempotent verification or refusal
  before every write, never a random extra report; and
- `--execute` may return zero only after invoking the fresh verifier and
  obtaining exact population, inventory and accounting equality.

Resource stops must produce typed terminals. An interrupted append with no
durable predecessor is `UNCERTAIN` and requires explicit disposition; it may
not be silently resumed.

## 9. M4-C33: adversarial and mutation battery

Freeze PRE regressions reproducing at least:

1. treatment/control association tapes differ because arm enters the seed;
2. control genesis differs from treatment genesis;
3. training-majority proportion is used as held-out accuracy;
4. sigmoid is declared while a linear head executes;
5. standard-normal/binary associations are injected into every family;
6. STOP rows select a checkpoint and also score retention;
7. one of four declared checkpoints is absent;
8. a CALIBRATION id or byte stream overlaps another role;
9. only two calibration generators are used to choose confirmation N;
10. M2 advances without an executable calibration comparison;
11. a wall/RSS/stop event is counted as right-censored evidence;
12. partial intervention JSONL is appended after restart from genesis;
13. an intervention artifact is group-writable;
14. an unknown report field or `RUN_REPORT` lookalike escapes inventory;
15. execute exits zero while fresh verification fails;
16. v5 is scored before its commit/push boundary;
17. a CONFIRMATION generator is constructed or inspected in this order; and
18. a producer edits an aggregate and repairs its digest.

Run guard-specific mutations against productive code. A broad upstream refusal
does not count as a bite for the targeted guard.

## 10. M4-C34: corrected DEVELOPMENT execution

After the v5 pre-outcome commit is pushed, execute the complete corrected
DEVELOPMENT screen and the same four diversity-selected mechanics units under
the new protocol. Preserve v4 evidence as history; write a new root and never
overwrite it.

Adjudicate the controls before proceeding:

- the easy positive control must pass its frozen criterion;
- random-label generalization must not be licensed as structured learning;
- all paired tapes and genesis bindings must be exact; and
- every scientific terminal must pass the fresh verifier.

If any condition fails, stop with `M4_V5_DEVELOPMENT_GATE_FAILED`.

## 11. M4-C35: bounded CALIBRATION execution

If and only if C34 passes, execute the sealed 16-generator CALIBRATION
population on CPU. Freeze from CALIBRATION:

- family/width eligibility without changing optimizer budgets;
- learnability margins;
- generator-level dispersion and its upper confidence bound;
- M0/M1/M2 nuisance encoding and any permitted regularization choice; and
- the exact list of CONFIRMATION slots, including typed ineligible slots.

No CONFIRMATION array or outcome may be generated, loaded or scored. A
structural test must enforce this boundary. Stop after producing a candidate
calibration adjudication for Musashi review.

## 12. M4-C36: return

Return one packet with:

- exact PRE/POST outputs and final pushed tips;
- v4-to-v5 field map and the pre-outcome commit/push proof;
- corrected population and resource census;
- association-tape, genesis and checkpoint-lineage digests;
- corrected DEVELOPMENT learnability table and four probe timings;
- CALIBRATION results and frozen thresholds, if C34 passed;
- precision and M0/M1/M2 calibration disposition;
- focal, mutation and final-suite counts from the final tip; and
- read-only B4/T2 service state and restart count.

Stop with exactly one of:

- `M4_V5_CALIBRATION_READY_FOR_EXTERNAL_MUSASHI_REVIEW`;
- `M4_V5_DEVELOPMENT_GATE_FAILED`;
- `M4_CONFIRMATORY_PRECISION_NOT_SUPPORTED`; or
- `M4_V5_PROTOCOL_IMPLEMENTATION_BLOCKED` with the exact failed acceptance
  item.

Do not run CONFIRMATION, use GPU, create a DOIN gene, touch financial/live data,
alter B4/T2/M3 or author external review authority.

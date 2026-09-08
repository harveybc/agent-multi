# 44. Data-Centric Signal, Model Information and Capacity Program

Status: ACCEPTED PROGRAM; M0-M2 bounded CPU pilot executed and independently
recomputed; T2 correction active; B4 remains isolated and unchanged

Date: 2026-09-07

## 1. Owner Decision and Controlling Principle

The trading and DOIN program is data-centric. Data acquisition, temporal
meaning, signal conditioning, feature retention and representation quality are
first-class experimental objects. Architecture and hyperparameter search do
not substitute for establishing what information reaches the learner.

Every research decision accepted as viable must become one of the following:

1. an executable work-plan item with an owner, prerequisite, deliverable and
   falsification rule;
2. a deferred item with a named factual dependency and an automatic opening
   condition; or
3. a rejected item with a scientific reason.

An accepted item may not disappear merely because it was first discussed in a
proposal, interview document or exploratory note.

The project goal of making the inputs "as good as possible" is operationalized
as a Pareto problem. No representation is assumed universally perfect. For a
named task, regime and resource budget, a retained input must demonstrate its
value while respecting causality, calibration, tail preservation, latency,
reproducibility and cost.

## 2. Relationship to Existing Plans

This plan binds four previously separate bodies of work:

- work plan 43: causal transformation evidence and DOIN admission;
- the thirteen-step information-to-knowledge roadmap in `predictor`;
- `PROPUESTA_DOCTORAL_MEMORIZACION_GENERALIZACION_DIMENSIONAMIENTO.md`;
- the STEP-05 source coding, entropy and MDL protocol.

The executable transformation ladder remains:

```text
T0 contracts -> T1 known truth -> T2 public utility -> T3 selector
             -> T4 DOIN admission -> T5 financial confirmation -> live shadow
```

This plan adds a model-side information lane and a complete per-feature
adjudication lane. It does not reinterpret B4, T0, T1 or any completed result.

## 3. Current Position

| Item | State | Next factual transition |
| --- | --- | --- |
| T0 operator mechanics | Accepted in scope | Preserve identity |
| T1 known-truth denoising | Externally reviewed for T2 | Preserve publication |
| T2 public utility | Active; execution-custody correction in progress | Complete correction, review, then execute sealed public screen |
| B4 Screen B | Running under frozen v7 authority | Finish and adjudicate; never retrofit preprocessing |
| T3 selector | Closed | Opens only if T2 supplies enough heterogeneous outcomes |
| T4 DOIN genes | Closed | Opens only for publicly eligible operators |
| T5 financial confirmation | Closed | Opens after B4 and T4 under a new identity |
| Model capacity/information lane | M0-M2 pilot accepted as bounded range evidence | M3-M4 confirmatory design follows the dependency rules in section 7 |
| STEP 04-13 implementations | Scheduled below | Open by explicit dependencies, not by memory |

## 4. Per-Feature Evidence Card

Before a new post-B4 financial experiment, every candidate input feature must
receive a versioned evidence card. A family-level statement is insufficient
when variables have different sampling, missingness or noise behavior.

Each card records:

1. logical source identity, license, acquisition and revision behavior;
2. observation interval, timestamp meaning, finalization and availability;
3. units, numerical support, missingness, duplicates and stale intervals;
4. known-truth SNR where a generator or calibrated measurement permits it;
5. on natural data, explicitly named noise diagnostics without relabeling them
   as true SNR;
6. quantization sensitivity and smallest useful numerical resolution;
7. marginal and conditional entropy/code-length diagnostics;
8. redundancy and conditional coding gain relative to other features;
9. raw, transformed and residual target utility on held-out tasks;
10. phase, amplitude, delay, extreme and regime preservation;
11. compute, memory, latency and output-width cost; and
12. one disposition: retain raw, retain transformed, retain raw plus residual,
    route conditionally, quarantine, or reject.

Paid sources pass the same card. Price and vendor reputation are provenance
facts, not evidence of incremental utility.

## 5. Model Capacity, Memorization and Description Lane

### M0. Definitions and measurement contract

Freeze distinct names and units for:

- theoretical classification capacity under stated assumptions;
- empirical random-association capacity;
- sample-specific memorization;
- held-out generalization and recovery of the generating rule;
- model description-length upper bound;
- data entropy/code-length estimate;
- residual learning capacity under a retention constraint; and
- cumulative exposures and optimizer updates.

Kolmogorov complexity is not computable and will not be reported as an exact
measurement. Lossless compressed length, prequential code length and
quantized-model description length are estimator-specific upper bounds. Their
serializer, quantizer, compressor, headers and precision are part of the
experimental identity.

### M1. Theoretical calibration and known-pattern bank

Start with binary threshold neurons with K inputs under the assumptions of the
MacKay/Cover experiment. Sweep K, the number of random binary associations,
seed and training budget. Compare the empirical saturation transition with the
approximately two associations per weight reference; do not transfer that
constant to multilayer or recurrent models.

Then use small MLPs on:

- Boolean rules with canonical structural complexity and randomized labels;
- structured rules plus a known fraction of sample-specific exceptions;
- clean and noisy variants with the realized corruption retained; and
- controlled temporal signals already licensed by T1: sinusoids, chirps,
  amplitude modulation, discontinuities, regime changes and state-space data.

Tasks, not adjacent samples or seeds, are the independent units. Split by
generator identity. MNIST or another natural benchmark may be a later realism
check, never the theoretical calibration target.

### M2. Training-trajectory instrumentation

At predeclared logarithmic checkpoints record:

- train, calibration and untouched evaluation loss;
- generating-rule recovery and sample-specific memorization;
- calibration, generalization gap and existing stopping-rule state;
- parameter count, exact-zero and near-zero fractions;
- fixed-grid weight histograms and their entropy;
- canonical serialized and compressed checkpoint lengths;
- quantization, pruning and low-rank distortion curves;
- matrix spectral norm, stable/effective rank and participation ratio;
- activation effective dimension and checkpoint-to-checkpoint CKA;
- bounded gradient/Fisher and top-Hessian sketches only where their measured
  overhead fits the pilot budget; and
- wall time, memory and metric overhead.

At the existing early-stopping event, fork the offline experiment. The
selection branch stops. A separately labeled diagnostic branch continues for a
bounded number of checkpoints to observe post-stop memorization and overfit.
That branch cannot select or promote a model. The untouched evaluation split
does not control stopping.

Graph eccentricity, diameter, modularity and path statistics are recorded for
topology-changing systems such as NEAT. They are not treated as learned
quantities for a fixed dense MLP. Weighted-graph variants require a threshold
sensitivity analysis.

### M3. Data information accounting

For every canonical dataset view distinguish:

- physical storage length;
- lossless compressed length under frozen compressors;
- entropy-rate and conditional code-length estimates;
- target-relevant predictive code-length improvement;
- unique observations and repeated training exposures; and
- code length of raw, transformed and residual streams.

Repeated epochs increase computation and exposure count, not the independent
information content of the dataset. A noisy stream may require more compressed
bits while carrying less target-relevant information; compression alone never
licenses a feature.

### M4. Residual-capacity intervention

After learning a structured task, present a held-out bank of random
associations under a fixed retention constraint on the original task. Measure
how many additional associations are learned before the original-rule loss or
calibration crosses its predeclared margin. This intervention, not repeated
weights alone, operationalizes residual learning capacity.

Run matched controls from random initialization and from checkpoints before,
at and after the stopping point. Separate additional capacity from optimizer
plasticity and catastrophic forgetting.

### M5. Architecture extension

Advance in this order:

1. one binary threshold neuron;
2. one-hidden-layer MLP;
3. deeper MLP;
4. small GRU with fixed unroll/horizon contracts;
5. NEAT, where topology metrics genuinely vary; and
6. bounded probes of the current actor, critic and feature extractor only after
   the preceding estimators are stable.

Weight sharing in recurrent or convolutional models prevents a naive additive
capacity calculation. Each result is conditional on architecture, precision,
optimizer, horizon and training protocol.

### M6. Utility gate

The measurements earn a place in the optimization program only if they improve
at least one decision under held-out tasks:

- earliest useful stopping point;
- smallest model reaching a fixed generalization target;
- detection of imminent sample-specific memorization;
- abstention when no candidate is supportable; or
- reduction of compute-to-target.

Required controls are parameter count, validation-loss early stopping,
partial-learning-curve extrapolation and compatible zero-cost proxies. Costs of
the capacity calibration and diagnostics are included and amortization is
reported. If the new measurements do not improve a decision, they remain a
negative scientific result and do not become DOIN genes or production gates.

## 6. Completion of the Thirteen-Step Program

The thirteen steps are not all implemented. "Protocol closed" means the
question and controls were documented, not that the experiment ran. Their
execution order is now explicit.

| Step | Executable disposition | Opening condition |
| --- | --- | --- |
| 01 sampling | T0 observation contract and per-feature card | Active/accepted in scope |
| 02 noise/SNR | T1 known truth plus per-feature natural-data diagnostics | Lab complete; public and financial applications pending |
| 03 denoising | T1 complete, T2 active | Finish sealed public screen |
| 04 quantization | CPU resolution/companding sweep per feature | Begin after T2 executor correction; no need to await a positive T2 result |
| 05 entropy/source/MDL | Data M3 plus model M0-M2; predictive residual controls | Begin after STEP 04 serializer/alphabet is frozen |
| 06 time-frequency | Causal magnitude/complex branches with phase separate | T2 identity/control machinery reusable and STEP 04 complete |
| 07 detection | Matched-filter and learned-detector bank | At least one representation with known-truth detection headroom |
| 08 equalization | Named-distortion correction versus identity | A measured source distortion exists |
| 09 common/private | Train-only common/private decomposition | Conditional redundancy survives STEP 05 public holdout |
| 10 synchronization | Availability metadata now; correction later | A reproducible lag with uncertainty and no future pull exists |
| 11 robust redundancy | Existing extractor corruption track | T2 closes and clean baseline is frozen |
| 12 adaptive routing | Identity/abstention plus eligible modes | At least two modes and a measured oracle routing gap |
| 13 allocation | Static then adaptive branch budgets | Multiple branches are independently useful |

No step may be omitted silently. A failed opening condition is its current
scientific result and must be recorded.

## 7. Waves and Resource Order

### Wave A: active now

- finish T2 execution-custody correction and public screen;
- let B4 run under its frozen contract;
- execute M0-M2 bounded CPU pilot after the current T2 correction return; and
- prepare STEP 04's canonical scalar-resolution design without scores.

### Wave B: after T2 execution is trustworthy

- adjudicate T2;
- execute STEP 04 CPU screen;
- execute M3-M4 on the known-pattern bank; and
- run M6 utility analysis before any new architecture sweep.

### Wave C: evidence-dependent representations

- execute STEP 05, then licensed STEP 06-10 branches;
- build T3 only if simple fixed rules do not suffice; and
- publish per-feature cards for the financial contract.

### Wave D: DOIN and financial use

- admit only licensed, bounded operators through T4;
- run T5 after B4 under a new data and experiment identity;
- evaluate feature-by-feature and branch interactions under paired budgets; and
- keep negative and identity controls in every population.

### Wave E: live

- reproduce artifacts and batch/incremental parity;
- run raw and transformed paths in read-only shadow;
- compare drift, latency and action divergence; and
- request separate approval for any Paper/Demo canary.

## 8. M0-M2 Execution Record

The bounded CPU pilot was completed on 2026-09-07 under design v3
`5df1c6dfde6b06267d44c50355c71211396318c523ce7ebea221650422947c2a`.
It contains 195 units: 45 threshold-neuron calibrations, 90 Boolean MLP
units and 60 temporal MLP units. The scientific result of every unit was
reconstructed from the design, specification and frozen code; producer
self-digests were not treated as scientific authority.

Disposition: `M0_M2_PILOT_ACCEPTED_AS_BOUNDED_RANGE_EVIDENCE`.

The result does not establish exact Kolmogorov complexity, intelligence in
bits, localization of knowledge in individual weights, residual capacity, a
general capacity law for multilayer networks, or eligibility for DOIN/live.
Its accepted uses are:

- calibrating the single-neuron experiment near the Cover/MacKay transition;
- demonstrating separable measurements of data exposure, memorization,
  generalization and model description;
- costing the diagnostics before a confirmatory experiment;
- selecting the confirmatory controls; and
- rejecting raw checkpoint compression as a standalone measure of knowledge
  or remaining capacity.

The complete public summary is
`docs/audits/evidence/MODEL_CAPACITY_M0_M2_PILOT_V3_SUMMARY_2026_09_07.json`.
The next design is recorded in
`docs/research/model_capacity/M3_M6_CONFIRMATORY_DESIGN_DRAFT_2026_09_07.md`.

## 9. Non-Negotiable Boundaries

- B4 is never retrofitted with these measurements or transformations.
- The current T2 design is not changed by M0-M2.
- No paid or financial dataset is accepted by reputation alone.
- No estimated SNR on natural data is labeled ground truth.
- No compressed byte count is labeled knowledge, intelligence or exact
  Kolmogorov complexity.
- No topology metric is promoted without predictive or decision utility.
- No feature is removed solely because it is redundant marginally; conditional
  and target-relevant value must be measured.
- No null result is bypassed by expanding the search after seeing it.

## 10. Program Completion

The data-centric program is complete only when:

1. each admitted feature and operator has a current evidence card;
2. the thirteen steps have an executed, rejected or factually blocked record;
3. model-capacity and data-information diagnostics have passed or failed M6;
4. DOIN searches only licensed finite spaces;
5. financial confirmation uses new identities and held-out evidence; and
6. live use follows shadow parity and separate owner approval.

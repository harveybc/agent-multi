# Musashi to General Satoshi: M0-M2 Model Capacity and Information Order

Date: 2026-09-07

Priority: P1 immediately after the active T2 C48-C56 correction return

Authority: owner explicitly ordered the accepted capacity, memorization,
compression and data-information work to become executable work. This order
requires no additional owner phrase for the bounded CPU work below.

Controlling plan:
`docs/work_plan/44_DATA_CENTRIC_SIGNAL_MODEL_INFORMATION_AND_CAPACITY.md`

## 1. Sequence

1. Do not interrupt or mix identities with the active T2 C48-C56 work.
2. After its return is sealed, inspect the current implementations in
   `preprocessor`, `agent-multi`, the T1 known-truth bank and the existing
   capacity proposal.
3. Freeze a PRE showing which M0-M2 measurements are absent, incomplete or
   merely prose.
4. Implement and execute the bounded CPU pilot in this order.
5. Stop before any confirmatory GPU sweep or integration into B4, T2, DOIN or
   live code.

## 2. M0: Measurement Contract

Create one typed schema and validator that separates:

- theoretical reference capacity;
- empirical random-association capacity;
- sample-specific memorization;
- held-out generating-rule recovery;
- checkpoint description-length upper bounds;
- data code-length/entropy-rate estimates;
- residual capacity under retention; and
- observations, repeated exposures and optimizer updates.

Requirements:

- no field named exact Kolmogorov complexity;
- canonical serializer and fixed quantization grids are identity-bound;
- compressor implementation, version, header policy and checksum are bound;
- bool-as-number, NaN/inf, duplicate keys and producer-declared verdicts refuse;
- all derived aggregates are recomputed from observation/checkpoint records;
- model and data code lengths use distinct fields and units.

## 3. M1: Calibration Bank and CPU Execution

Materialize before scoring:

- threshold-neuron tasks over a bounded grid of K values and random binary
  associations around the MacKay/Cover transition;
- one-hidden-layer MLPs over a logarithmic width grid;
- canonical Boolean rules with clean labels, known random-label fractions and
  held-out rule identities;
- a bounded temporal subset from the accepted T1 generators, including clean,
  Gaussian-noise and structured-noise regimes; and
- matched random-label and identity/null controls.

Use task identity as the independent unit and seeds as nested repetitions.
Record training failure and non-saturation in the denominator. Do not call a
finite-budget failure a capacity limit.

Pilot budget:

- CPU only;
- at most 2 wall-clock hours total;
- at most 1 GiB peak RSS per worker;
- concurrency chosen so B4 GPU training and its data feed are not starved;
- externally observable progress, ETA, stop-file and typed terminal states;
- no network, venue, services, sealed-2025 data or financial scores.

The pilot is authorized to produce mechanics and range evidence. It is not a
confirmatory estimate and cannot license an optimizer gene.

## 4. M2: Dynamic Checkpoint Instrumentation

Instrument at logarithmic checkpoints and at the existing stopping event:

- train/calibration/evaluation log loss;
- rule recovery and sample-specific memorization;
- parameter count and exact/near-zero fractions;
- quantized histogram entropy;
- canonical raw and compressed checkpoint lengths;
- quantization, pruning and low-rank distortion curves;
- effective/stable rank and spectral norm;
- activation participation ratio and bounded CKA; and
- measured overhead for every diagnostic.

At the stopping event create two separately identified branches:

1. the selection branch terminates normally;
2. a diagnostics-only branch continues for a bounded post-stop window.

The second branch cannot select, overwrite or promote the first. Test data
cannot drive either stopping rule. All branch lineage must be explicit.

Graph eccentricity/diameter/modularity are omitted for fixed-topology MLPs
except as static metadata. Provide an adapter contract for later NEAT use, where
topology changes, but do not execute NEAT in this order.

## 5. Required Adversarial Tests

At minimum freeze and kill:

1. a producer labeling compressed length as exact Kolmogorov complexity;
2. changing quantization or compressor without changing identity;
3. counting repeated epochs as new independent data information;
4. fitting a threshold-neuron curve that never reached saturation and calling
   its endpoint capacity;
5. using the evaluation split for early stopping;
6. allowing the post-stop diagnostic branch to become selected;
7. a recomputed self-digest after changing one checkpoint metric;
8. swapping a checkpoint between task/model/seed identities;
9. omitted failed tasks or seeds;
10. exact repeated weights being interpreted automatically as spare capacity;
11. non-finite spectral/compression metrics; and
12. an aggregate that does not rederive from records.

Tests must call the production derivation and consumption paths. A source-text
assertion is supplementary, not sufficient.

## 6. Return Package

Return one packet containing:

- PRE reproductions;
- implementation identities by repository and file digest;
- sealed pilot design committed before scores;
- complete unit ledger and terminal-state counts;
- observed capacity/saturation curves with uncertainty and explicit
  non-saturation cases;
- checkpoint trajectories before, at and after stopping;
- metric overhead and resource use;
- every mutation and the exact tests it broke, measured before prose;
- bounded conclusions and nulls; and
- a proposed confirmatory design, not executed.

Required final disposition is one of:

- `M0_M2_PILOT_COMPLETE_CONFIRMATORY_DESIGN_READY_FOR_REVIEW`;
- `M0_M2_MECHANICS_COMPLETE_SIGNAL_INCONCLUSIVE`; or
- `M0_M2_PILOT_FAILS_UTILITY_OR_IDENTIFIABILITY`.

No success label may imply that description length measures intelligence,
knowledge stored in individual weights or exact Kolmogorov complexity.

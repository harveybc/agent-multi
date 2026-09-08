# M3-M6 Confirmatory Design Draft

Status: proposed after the M0-M2 pilot; not executed; no optimizer or live authority

Date: 2026-09-07

## 1. Question

Can measurements collected during training improve a concrete decision about
when to stop or how large a model to use on an unseen task, and can residual
learning capacity be measured by intervention while preserving previously
learned behavior?

This question does not ask for exact Kolmogorov complexity or a scalar measure
of intelligence. Model descriptions, memorization, generalization and residual
capacity remain different outcomes.

## 2. Lessons carried from the pilot

1. The threshold-neuron transition was visible near two random label
   associations per input, but three seeds and small K cannot confirm Cover's
   counting result.
2. Raw checkpoint compression had only weak descriptive association with
   train/evaluation score and almost none with exception memorization.
3. Full model-description instrumentation consumed most diagnostic CPU time.
4. Calibration stopping reduced evaluation score on average during the
   post-stop continuation, but it also stopped every XOR control near chance.
   Failure to learn and lack of model capacity must therefore be separated.
5. Colored noise changed temporal generalization materially relative to white
   and impulsive noise at the same declared SNR.

Consequently, full compression, pruning and SVD diagnostics are sampled only
at initialization, selection and the post-stop endpoint. Cheap train/calibration
curves, gradient summaries and exception memorization remain more frequent.

## 3. Hypotheses

### H1: single-neuron calibration

For random points in general position and random binary labels, an
optimization-independent linear-separability test will reproduce Cover's
finite-sample separability probabilities over a grid around N/K = 2. The
perceptron learning outcome is secondary and measures optimizer failure, not
capacity.

Falsification: simultaneous confidence intervals fail to cover the exact
counting probabilities after the Monte Carlo error budget is met, or the
feasibility solver and an independently checked solver disagree.

### H2: residual capacity under retention

After a model learns a structured task, the number of additional random
associations it can fit before crossing a frozen retention margin depends on
the checkpoint and cannot be inferred from parameter count, repeated weights
or compressed length alone.

Falsification: the intervention is not reproducible, retention and acquisition
cannot be separated, or simple controls predict the endpoint as well as the
full measurement set on unseen task generators.

### H3: decision utility

A selectively chosen subset of trajectory measurements improves either
early-warning of sample-specific memorization or selection of the smallest
model reaching a fixed generalization target over validation loss, parameter
count and partial-learning-curve controls.

Falsification: nested evaluation on unseen task generators shows no
incremental utility after accounting for calibration cost, or the method
cannot abstain when all candidates are unsupported.

## 4. Experimental stages

### C1. Exact threshold reference

- K in {32, 64, 128}; N/K concentrated on {1.25, 1.5, 1.75, 2.0, 2.25,
  2.5, 2.75}.
- Random Gaussian points in general position and independent binary labels.
- At least 200 labeling tasks per cell, increased only by a frozen Monte Carlo
  precision rule.
- Primary outcome: linear separability from a deterministic feasibility
  solver with an independent solver check on a fixed sample.
- Reference: Cover's finite-N counting probability; the 2K statement is an
  asymptotic transition, not a universal equality for every finite training
  algorithm.

### C2. Structured learning and exceptions

- Boolean identity, majority, DNF and parity families plus random-label nulls.
- Temporal sine, chirp, amplitude-modulated, discontinuous and state-space
  generators under white, colored, impulsive and heteroscedastic noise.
- Separate generator identities for train, calibration and evaluation.
- Optimizer controls must first demonstrate that each structured family is
  learnable; otherwise the cell is `OPTIMIZATION_LIMITED`, never
  capacity-limited.

### C3. Residual-capacity intervention

From initialization and checkpoints before, at and after calibration stopping:

1. freeze an original-task retention margin;
2. add random associations in blinded batches;
3. train under a fixed update budget;
4. evaluate original-task retention and new-association fit after each batch;
5. stop at the first confirmed retention violation or acquisition failure; and
6. repeat from a random-initialization control with matched compute.

The endpoint is a conditional empirical intervention result. It is not called
unused bits or remaining intelligence.

### C4. Utility gate

Fit decision rules on development generators and evaluate on unseen generator
families. Compare:

- validation-loss early stopping;
- parameter count and elapsed updates;
- partial-learning-curve extrapolation;
- cheap differential/activation descriptors;
- sparse 8-bit description length, pruning and rank descriptors; and
- the full candidate set with calibrated abstention.

The independent unit is the task generator, not a seed or adjacent temporal
sample. Seeds are nested repetitions. Calibration, diagnostic computation and
failed runs remain in the cost denominator.

## 5. Statistical contract

- Freeze primary metric, margins and multiplicity family before C1/C2 scores.
- Use task-level paired contrasts and task-level bootstrap or randomization;
  never treat checkpoints or seeds as independent units.
- Report uncertainty by family and pooled only when pooling is licensed.
- Separate model-selection, calibration and untouched confirmation tasks.
- A missing, failed or non-learnable task remains in the denominator with a
  typed state.
- A negative utility result prevents all new measurements from becoming DOIN
  genes or production gates.

## 6. Resource and integration boundary

C1-C4 begin on CPU and may request GPU only after measured preflight. They do
not alter B4, T2 or completed financial evidence. Any later use in DOIN enters
through work plan 44 M6 and T4 after public utility, cost and reproducibility
are established. The NEAT extension must implement
`NEAT_MODEL_INFORMATION_ADAPTER_CONTRACT.md` and receives its own design.

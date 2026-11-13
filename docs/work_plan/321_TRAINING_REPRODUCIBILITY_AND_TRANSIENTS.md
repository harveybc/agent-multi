# Training Reproducibility And Early Transients

Owner requests diagnosis before the next critic refinement or floating pass.
This overrides320's immediate action/state attribution order: establish training
reproducibility and explain early checkpoints first. No new experiment launched
by this document. Source299 and its71 inputs remain unchanged; STL paused.

## Existing Evidence Reviewed

Source: sealed320 native-head results and checkpoints, runtime `bddfb7db`,
identity `00d2f39584defa061924fb26869e3dbd3eb47b6c1ff9533e4aec291b14cfd67c`.
The four fits train only two freshly initialized heads on frozen pretrained
encoders. They are not full-data SAC retraining runs.49/98 FIT anchors from
the respective TRAIN years are reused in batches128.64 optimizer updates
therefore mean167.18/83.59 sample-equivalent passes, not64 epochs and not64
passes over the full four-year dataset.4096 updates mean10699.76/5349.88 passes.
These are nominal exposure counts, not independent sample counts.

| Fold/seed | Stop regret at initialization | Selected update | Selected stop regret | Best stop regret after patience expires |
| --- | ---: | ---: | ---: | ---: |
| A/19 |0.289771|960|0.263909|0.286192|
| A/43 |0.320892|1408|0.301294|0.305903|
| B/19 |0.202453|64|0.215644|0.229651|
| B/43 |0.220521|1024|0.199364|0.241203|

All four curves continued to4096 for the fixed-budget control. None recovered
the selected stop-block minimum after the stopping event within this budget.
This does not exclude eventual later recovery. Initialization is measured but
excluded from checkpoint eligibility; B/19 consequently selects a trained
checkpoint worse on its stopping block than its own initial head. This is a
diagnostic warning, not a reason to deploy the random head or retrospectively
rewrite320's selection rule. It does not invalidate320's like-for-like paired
comparison, but prevents interpreting an early selected score as learning gain.

In B/19, FIT regret improves0.255988 ->0.086706 from selected to4096, while
audit regret worsens0.213729 ->0.231560. Its audit value RMSE improves slightly
0.683019 ->0.677701, but action-gap RMSE worsens0.535451 ->0.563015.
Value accuracy and action ordering need separate instrumentation.

Existing selected/fixed weight comparisons show relative Frobenius changes
of roughly89-142% in first-layer head weight matrices and23-42% in final-layer
weight matrices. These are descriptive magnitudes, not causal attribution:
layer norms/scales differ, and the functional consequences must be measured.
Encoders did not train in320, so encoder drift cannot explain its differences.

Control inference replay and checkpoint reload are exact. Identical independent
native training trajectories have not yet been compared; do not equate exact
inference with verified optimization reproducibility. Different-seed variation
is a separate question. No claim of irreducible noise or insufficient history.

## Diagnostic Order And Acceptance Criteria

1. **Identical-run reproducibility.** Repeat the frozen320 B/19 protocol twice
   in separate processes on the same device. Check source, labels, masks,
   initialization, minibatch order, dtype, dispatch and modes. Save model,
   optimizer and RNG states at0,1,2,4,8,16,32,64 and every64 afterwards.
   Compare tensor digests, losses, gradients, updates and outputs; if unequal,
   locate the first divergent update before any learning-rate experiment.
   Also test interrupted/restarted training against uninterrupted training.
   Same-device exactness and cross-device tolerances are distinct contracts.

2. **Functional weight diagnosis.** On fixed FIT/stop/audit inputs measure MSE,
   common-value and action-gap errors, action margins, tie proximity, argmax
   changes, gradient norms, clipping frequency, update/weight ratios, optimizer
   moment magnitudes and hidden activation changes. Decompose the FIT gradient
   into common-value and centered-gap terms and compare directions. Compare
   early/post-early/final checkpoints within each seeded trajectory. Frozen
   encoders are invariance controls. Layer transplants and interpolations can
   localize functional changes but are hybrid diagnostics, not deployable models
   or proofs that a single layer caused the entire training failure.

3. **Small controlled optimization test.** After exact reproducibility passes,
   compare constant LR3e-5 with3e-6 on identical initial weights and minibatch
   prefixes, preserving labels, full diagnostic support and architecture.
   Include a longer prespecified budget and a longer low-LR arm with comparable
   cumulative step size to distinguish slower progress from real improvement.
   Continue beyond patience for diagnosis only, with the original selected
   checkpoint locked. Choose budgets/checkpoint metrics before launching, not
   after inspecting the curves. Report step0 as a baseline, not automatic
   eligibility for promotion. Both seeds and folds are required before a
   general training recommendation.

4. **Known learnable controls.** First use the same real frozen embeddings with
   targets generated by a fixed teacher representable by the tested head. Test
   teacher realizability, training decrease and held-input recovery against
   initialized and constant baselines. Then, if representation learning needs
   isolation, use synthetic multiscale signals with causal known targets and
   separately held phases/frequencies. Do not use future samples in synthetic
   features or transfer synthetic success to a trading-profit claim. Retain
   original architecture/modes/optimizer where the control permits; document
   every changed factor. These controls must not join the live feature store.

5. **Coverage only after mechanism checks.** A49/98-anchor critic calibration
   may be undercovered even when four years of market data exist. Measure more
   representative TRAIN anchors as a paired coverage experiment if evidence
   points there; do not jump straight to buying longer history or increasing
   all-network capacity. Full SAC additionally requires replay, bootstrap,
   target-network, entropy and actor/critic update-state reproducibility checks
   not exercised by a supervised frozen-head diagnostic.

## Traceability

| Concern | Required discriminating evidence |
| --- | --- |
| Same data, different results | Same-seed independent and resume tensor/output trajectories |
| Missed late recovery | Fixed-budget curve past patience, early checkpoint kept immutable |
| Coarse optimizer steps | Gradient/update ratios plus paired LR and exposure controls |
| Good values, bad trading actions | Gap errors, action margins and native action regret |
| Suspect module | Same-input functional interventions and invariant encoder hashes |
| Dataset versus implementation | Known representable teacher and causal temporal controls |

Deliverables: reproducibility table, component-level trajectory plots, initial/
early/late comparison and a justified next contrast. No source policy promotion
or new floating pass until the diagnostic conclusion is explicit.

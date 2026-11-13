# Training Forensics Execution Contract

Implements owner-approved321 before actor refinement or floating selection.
Source299,71 inputs, original318 annual targets/masks,320 native heads retained.
No source actor/encoder fitting,2024 inference,TEST access or STL admission.

## Frozen Sequence

1. Prepare authenticated frozen embeddings, native initial tensors and labels.
   Verify320 fixed endpoints on the reconstructed cache and source invariance.
2. B/19 with LR3e-5: two independent4096-update processes and another process
   stopped at512 then resumed in a fresh process to4096. Compare every update's
   loss/gradient norm/gradient digest, model/optimizer/RNG digests at checkpoints,
   curve values and outputs. Also compare4096 parameters to320. A mismatch
   stops dependent experiments and reports the first divergent item.
3. After that gate, folds A/B x seeds19/43: constant LR3e-5 for8192 updates and
   LR3e-6 for81920 updates, with the8192 low-LR endpoint retained. Same initial
   weights and minibatch prefixes; the10x exposure is a rough cumulative-step
   comparison, not equivalence under AdamW or weight decay. RegretStopper20,
   delta1e-8 and64-update selection schedule unchanged. Continue beyond its
   trigger diagnostically without changing its selected checkpoint.
4. Four real-embedding known-teacher controls (A/B x19/43),8192 updates, LR3e-5;
   teacher is the same208->8->3 representable head, frozen and independent of
   the student. Include its exact realizability residual, initialized and
   constant FIT-mean baselines. A fifth control uses causal multiscale sinusoid
   lag vectors and a representable teacher, with disjoint phase intervals.
   This fifth test isolates head optimization, not full TCN learnability.

Dense checkpoints:0,1,2,4,8,16,32,64, then every64 through8192; extended tail
metrics/stopping remain every64, but tail full-state checkpoints every1024
and at81920. Save early selected checkpoints independently. Every saved state
includes parameters, optimizer moments, RNG, step and frozen stopping state.

## Instrumentation

Track FIT/stop/audit action regret, value RMSE and gap RMSE; step0 is descriptive,
not eligible for selection. Record common-value versus centered-action MSE and
gradient norms/cosine, gradient clipping frequency, actual update/weight norms,
hidden activation drift, action margin/tie counts, greedy switches and weight
drift per layer. Audit metrics and gradient diagnostics never drive fitting.
Known-teacher controls are artificial-only evidence, not deployable features.

## Tests Before Launch

| Requirement | Test |
| --- | --- |
| Value/gap decomposition | MSE identity and gradient additivity |
| State digest sensitivity | Detect changed weight, optimizer moment and RNG |
| Restart equivalence | CPU interrupted AdamW state restoration equals uninterrupted |
| No synthetic future access | Multiscale lag fixture prefix invariant |
| Known realizable teacher | Teacher evaluated on same input gives exact target |
| Same experiment identity | Seals, schedule and checkpoint comparison in GPU integration |

Report reproducibility separately from generalization. Interpret weight changes
through functional outputs, not a claim of knowing causes from tensor values.
The small49/98-anchor diagnostic does not certify full SAC replay/target-network
or entropy-training reproducibility; those are separate later tests if needed.

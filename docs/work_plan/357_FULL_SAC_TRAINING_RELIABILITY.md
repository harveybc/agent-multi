# Full SAC Training Reliability: Bounded First Diagnosis

Owner priority2026-11-15: phase2 stays OPEN; no DEAP or live work. Preserve
selected299,71 inputs,28 groups, full2020-2023 TRAIN and2024 validation,
window13/stride2, fixed TRAIN scaler and the native business engine. STL stays
paused. G3 remains an independent pinned run, not a substitute for this gate.

## Existing Evidence And Scope

347 reproduced256 protected critic-head transactions and18 checkpoints exactly.
Its two optimizer swaps did not recover the missed action; that is not evidence
about unrestricted end-to-end SAC.356 verifies construction and fixture updates,
not convergence. Do not repeat those studies or infer a universal LR defect.

This first bounded experiment uses the COMPLETE SAC, not frozen head regression.
Collect one full native TRAIN traversal with the preserved frozen actor and the
existing warmup policy flag (the pinned recipe uses policy actions, not random
warmup). Freeze that real transition dataset per seed.
This separates optimizer effects from changing experience. It is NOT an online
training epoch or a proof that closed-loop training is repaired. Validation is
evaluated only, never collected into replay. No synthetic transitions here.

## Predeclared Procedure

Three independent seeds7/19/43, one per GPU worker. Within each seed all arms
use identical initial parameters, optimizer moments, real replay, batches and
dropout RNG. Cross-device bit identity is NOT required; within-device repeat
and cold-resume identity are required. Do not rank hardware/seed against seed.

1. Authenticate bundle, data and engine hashes. Publish baseline full accounts.
2. Verify splitting the inherited AdamW into actor/critic/temperature groups
   at UNCHANGED learning rates reproduces8 original updates exactly.
3. Verify64 continuous updates versus a cold restart at32: loss traces,
   model, optimizer, sampler and RNG must match exactly. Save the first
   mismatching field/step on failure and stop dependent intervention work.
4. Sequential arms: unchanged control, actor LR x0.1, critic LR x0.1,
   temperature LR x0.1, all LR x0.1.512 updates per arm, original batch size.
   These are diagnostic interventions, not selected/optimized hyperparameters.
5. Save state and weight/function diagnostics at0/1/8/32/64/128/256/512;
   checkpoint every32 updates for crash recovery. Report complete real TRAIN
   and validation accounts at0/64/256/512. No early-best selection or extension.

The fixed replay traverses the whole TRAIN support;512 updates is explicitly a
bounded optimization budget, not a reduced dataset or shortened input window.
Publish actual sampled coverage. A role LR effect is conditional on the fixed
replay, inherited moments and other settings. No model promotion, feature-pass
restart, phase2 closure or DEAP launch follows automatically.

## Test Design Before Implementation

| Requirement | Behavioral witness |
|---|---|
| No parameter silently fixed | Actual optimizer groups/names/LRs; unknown role rejected |
| Preserve inherited moments | Unchanged grouped optimizer reproduces original updates |
| Cold restart | Identical next batches, dropout draws, weights, optimizer and RNG |
| Real support only | Native TRAIN-only transitions, pinned hashes, full support/count receipt |
| No mixed comparisons | Same seed/replay/start state/schedule per arm; identity check on resume |
| Weight interpretation | Per-role/module drift and gradients plus fixed-input logits/Q/actions |
| Honest failure | Failed identity/replay gate prevents subsequent interventions |
| Automation | Locked worker, atomic checkpoints, persistent service, finite work and result |

Unit tests: optimizer roles/moments, changed-LR consumption, exact update and
cold resume, identity mismatch and unknown parameter refusal. Integration:
real71-input preflight and on-device qualification. System: three pinned workers
run the finite queues and leave complete receipts. Aggregate only authenticated
results; absent/failed workers must never be counted as successes.

Next after this bounded evidence: identify whether divergence exists, or whether
the identical optimizer consistently damages actions. Then define the smallest
full online paired continuation test justified by the measured effects. A lower
LR that merely freezes the model is not a successful learning correction.

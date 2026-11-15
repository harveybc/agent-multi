# Full SAC Training Reliability: Bounded First Diagnosis

## Mandatory Reporting Clarification

All15 arms completed512 optimizer updates by FIXED_UPDATE_BUDGET, not patience.
There were zero online training epochs: one frozen collection traversed all
TRAIN, followed by minibatch updates on fixed replay. Batch128 gives65,536
sampled transitions, not65,536 distinct observations. Full accounts were
evaluated only at updates0/64/256/512. Lines connecting those four points are
visual guides, not observations between evaluations. This experiment cannot
exclude recovery after longer training or qualify a low-LR convergence recipe.

Every future status/table must state: run type, completed/max epochs (when
applicable), optimizer updates, evaluation cadence, early-stopping enabled,
monitor metric, min_delta, warmup, patience current/limit, selected epoch and
stop reason (budget/patience/error/manual/running). Mark inapplicable fields as
such, never infer epochs from replay draws. Show the final checkpoint separately
from the selected best checkpoint. Preserve the complete curve, including the
early transient, and align weight/function diagnostics to the same step index.

Next long-training design must give the reduced-LR arms enough predeclared
budget, retain best checkpoints throughout and report patience explicitly.
The short357 results identify sensitivity only; they do not reject late recovery.
No long run is launched by this reporting correction.

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

## Deployment Evidence

Pinned worker revision:eaa90f22. Three persistent services are active on the
RTX5090(seed7), RTX4090(seed19) and RTX4070(seed43). Each queue owns its output,
lock, immutable source export and32-update recovery checkpoint. User lingering
and boot enablement are verified on all three. Worker memory is capped at5GiB;
Torch/BLAS threads are bounded to avoid competing with existing work. G3 is
running separately on its original runtime and was not stopped or modified.

All three INITIAL accounts reproduced the preserved baseline exactly:

| Metric | TRAIN | Validation |
|---|---:|---:|
| Return % |16.05822355000215|18.369945750000117|
| Sharpe |0.7819691259276358|2.690712133251012|
| Trades |745|207|
| Orders/week |3.5645933014354068|3.8333333333333335|

Identical support SHA256 on all workers:
`28d6088a92173592f39e221c1c2ab1efc9094e614eeccb11c9ae23cf14878c98`.
This is an inference/account control, NOT a repeated-training success.
Actual batch size is128, inherited unchanged from the current training design.

Output root on each worker: `full-sac357-20261115/seed_<seed>` under the
external experiment directory. `SPEC.json` binds code/data/model identities;
`DATA.json` authenticates the full native TRAIN transition set;
`QUALIFICATION.json` gates interventions; arm `HISTORY.json`, checkpoints and
`RESULT.json` retain updates, weights, losses and full account metrics.
The read-only collector checks checkpoint hashes and common experiment identity
and emits `aggregate/REPORT.md` and `RESULT.json` only after all three finish.
Missing, failed or mismatched results cannot mark the fleet complete.

Current status: full TRAIN collection running, no intervention result yet.
Original-versus-grouped update parity and cold-resume tests passed on CPU;
on-device qualification runs automatically after each full collection.

Verification:47 focused CPU tests passed, including existing native/recovery
tests and new negative collector tests (missing worker, mixed code identity,
duplicate seed); scoped Ruff passed. The collector's final completion flag is
written only AFTER common-contract validation and result publication.

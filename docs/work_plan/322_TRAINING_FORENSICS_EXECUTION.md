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

## Preserved Control Limitation And Supplement

While322 runs, A/19 known-teacher FIT is observed to choose only action1.
Its value-regression results remain valid but cannot establish learning all
three decision regions. Keep every original result. Add five separate controls
after the original queue: A/B x19/43 on the same real embeddings, plus the
multiscale sinusoid control. The representable teacher uses one hidden SiLU
activation s and outputs `(l-s,0,s-u)/(u-l)`, with l/u its FIT-only1/3 and2/3
quantiles. No stop/audit labels enter calibration. Require each FIT action to
cover at least20%; reject degenerate activation ranges. Both twins share this
exact function. This changes the teacher, not real labels or the candidate
model. Unit tests check balanced coverage and invariance to held-input changes.
Publish this supplement separately; do not retrofit or erase the original run.

## Completed Execution

Original queue:16 units,540.40 seconds, runtime `b4747654`, identity
`e1d5e7197a8969aa3a65ffa5b2a0af278f74df70d538cce14acba691ddab9c6e`.
Balanced supplement:5 units,71.31 seconds, runtime `1c187805`, identity
`bcb56fc708a5765de97e0cde14cead8a388a077bb1ebc7530fd0768c8f279c3c`.
Both sequential services exited0.22 focused tests passed.325MB of full states,
optimizer moments, gradients and curves remain outside the repository.

### Reproducibility

- Both independent B/19 processes and the512-update interrupted/restarted
  process agree exactly across4096 update traces, checkpoint model/optimizer/
  RNG/output digests and full curve records. Final weights match320 exactly.
- All four extended baseline runs match their320 weights at update4096.
  Low/base arms start with identical tensors within each fold/seed.
- Independent stopping reconstruction passes for all21 units. The independent
  analysis produced on the execution host and local CPU is byte-identical.
- The saved optimizer LR,weight decay and final step were verified for all21
  units; the smaller LR actually reached AdamW and did not remain fixed.
- This establishes reproducibility of this frozen-head diagnostic, not the
  entire historical SAC training/replay pipeline or arbitrary cross-GPU fits.

### Late Recovery Is Real, But Not Sufficient

Five of eight real-target runs improve their stop-block regret after the
original patience has expired. One of those five also improves audit regret.
The following late checkpoints are descriptive minima, not promoted policies.

| Arm | Original selected | Later recovery | Stop regret selected -> later | Audit regret selected -> later |
| --- | ---: | ---: | --- | --- |
| A/19 low LR |64|11648|0.289771 ->0.265076|0.222980 ->0.275653|
| A/43 base LR |1408|7680|0.301294 ->0.260340|0.216413 ->0.242840|
| A/43 low LR |64|18240|0.320892 ->0.295567|0.210989 ->0.244751|
| B/19 low LR |64|11520|0.206956 ->0.204197|0.224644 ->0.226955|
| B/43 low LR |64|7552|0.220521 ->0.204660|0.307524 ->0.231767|

All low-LR runs select64 and trigger patience at1344. A tenfold slower optimizer
was given the same1280-update patience budget. The discrete greedy-regret
metric can remain flat while outputs move; min_delta1e-8 does not fix that
quantization. B/19's untrained initial head still beats every observed stop
checkpoint even after81920 low-LR updates. Do not call every early solution
an optimizer local minimum or assume every longer trajectory must recover.

At equal8192-update exposure, low LR improves audit regret in2/4 pairs.
At81920 low-LR updates versus8192 base updates, it improves3/4 pairs. Neither
comparison is a consistent four-pair improvement. Full native-twin FIT regret
falls to0-0.02005 while stop regret remains0.20936-0.33859 at the respective
final endpoints. More updates reliably fit these anchors; that is not enough
to generalize across these blocks.

### What The Weight Analysis Shows

Initial common-value MSE gradient norms are3.31-6.85 times the centered
action-gap gradient norms. The two parameter-space gradient directions oppose
each other at most measured checkpoints. Both FIT components nevertheless
decrease, so this is competing training pressure, not proof that optimization
is broken. No gradient clipping occurs in the eight real-target runs. Maximum
measured one-update/weight norm ratios are about0.101% at base LR and0.017% at
low LR; no numerical explosion was observed, although small parameter changes
can still cross an action boundary.

Sixty-four layer interventions compare early/late functional states. Restoring
the second twin's early final linear layer in the late model improves audit
regret7/8 and both stop/audit6/8. Other single-layer swaps have mixed effects;
none improves both blocks in every case. This identifies that output mapping
as a candidate for a prespecified confirmation, not an established uniquely
faulty component. Twin numbering is initialization-specific, not a business
role. No hybrid was deployed or used to change the source policy.

### Known-Function Controls

The original random teacher's almost-single-action task was retained and not
presented as a sufficient action-learning test. The supplement's exact teacher
has FIT action counts17/16/16 for A,33/32/33 for B and128/128/128 for sinusoids.
All five teacher realizability residuals are exactly0.

| Balanced teacher | FIT RMSE initial -> final | Audit RMSE final | Audit constant baseline |
| --- | --- | ---: | ---: |
| Real embeddings A/19 |1.531697 ->0.066524|1.539345|1.530002|
| Real embeddings A/43 |1.640010 ->0.073856|1.638164|1.530002|
| Real embeddings B/19 |1.466409 ->0.193373|1.467892|2.337171|
| Real embeddings B/43 |1.546989 ->0.210122|1.446088|2.337171|
| Causal multiscale lags |0.716323 ->0.024327|0.024428|0.591857|

The sinusoid control learns all three decision regions (audit predicted counts
66/66/60; true67/64/61), audit regret0.000360. This validates this head optimizer
on a known temporal-lag input function, not the entire TCN's trainability.
The real-embedding teacher substantially fits TRAIN but still fails to beat a
constant in foldA. A known representable function can therefore show this
generalization gap on the sparse historical embedding panel without label noise.
This supports investigating coverage/support and objective geometry before
requesting more calendar years; it does not isolate coverage as the sole cause.

## Disposition And Next Action

Preserve actor299 and71 features; no new floating pass or parameter transplant.
Retain the complete trajectories rather than selecting retrospectively from
audit. The next useful diagnostic is a paired FIT-coverage curve on the known
balanced teacher using additional existing TRAIN embeddings, followed by real
native-label coverage if that discriminates the problem. Keep architecture,
initialization and optimizer fixed; specify stopping budget in relation to LR
and exposure, and distinguish learning progress from checkpoint eligibility.
Include initialization and constant baselines in every future calibration.
Do not add an arbitrary warmup or claim that patience20 is now universally fixed.
The source's historical mixed-COT limitation from311 remains unchanged.

Evidence: [reproduction](generalization/training_forensics_20261113/REPRODUCIBILITY.json),
[independent analysis](generalization/training_forensics_20261113/ANALYSIS.json),
[original results](generalization/training_forensics_20261113/RESULT.json),
[balanced controls](generalization/training_forensics_20261113/BALANCED_TEACHER.json),
[training curves](generalization/training_forensics_20261113/TRAINING_CURVES.png).

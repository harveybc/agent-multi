# Parallel Phase2 Selection and Module Weight-Space Diagnosis

Hardware attribution correction2026-11-09: the launch used a CUDA ordinal,
not a verified physical device. The RTX5090 attribution below is unverified;
see226 for the discovered ordinal mismatch and enforced device-name gate.
No numerical result or source checkpoint is changed by this correction.

Owner priority2026-11-08: demo integration must not displace feature selection
or direct analysis of learned weights. Continue both independently of Retsu.

## What Already Happened

Steps197/198 and211 performed algebraic channel removal, constant-response
preservation and exact retained-weight transfer. Those are explicit weight-space
operations. They are not a comprehensive diagnostic of every module, nor a
demonstration that semantic weight edits improve trading. The critic anchor220
was gradient-based training, not gradient-free weight-space programming.

The linked article reports early weight-reading/editing successes and failure
cases, sometimes using CPU weight inspection plus bounded forward evaluations.
This motivates measurable interventions, not a claim of guaranteed insight or
improvement: https://rikiyat.github.io/blog/weight-space-programming/

## A. Phase2 Continues on the Primary GPU

Use the completed selected70 epoch0 bundle as the parent, not the degraded
epoch40 actor. Execute the existing ordered deletion pass followed by reinsertion
through feature_selector0.2.4. Keep frozen weights, full TRAIN/VALIDATION support,
13 observations/stride2, fixed TRAIN scaling, costs and business contract.
Remove improvements/ties; reinsert strict improvements; activity must exceed
two orders/week in both periods. Preserve the original checkpoint for paper.

The first prepare attempt found the old environment's missing two_pass_greedy
entry point. Reuse the already pinned0.2.4 plugin directory via PYTHONPATH;
do not change shared installed packages or bypass plugin identity checks.
Successful identity:
`906c0dc6795b237722c12704b4fd243591a7e7b2d79244325d78dc223162a96e`.
Runtime: `1d3c5f71`, same authenticated source as the completed parent.
The independent GPU service passed original account/probe checks and completed
67 fresh evaluations and both ordered passes. No unrelated process was stopped.
Result:70 ->69 ->69 columns in28 groups; RAW_CLOSE_LOG_RETURN was removed.
Full results and the physical69-channel bundle are in step225 below.

After selection, preserve its masked winner. Structural weight transfer must
pass action/account parity before any refinement. Do not automatically repeat
the known degrading head-only fit just because a GPU is idle.

## B. CPU Weight-Space Diagnostic

Compare effective TorchRL parameters, not stale standalone actor copies, at
selected70 epoch0, recovery epoch40 and the4096-update diagnostic critic anchor.
Inspect groups/extractors, fusion, positional attention, core, actor head, online
critics and target critics. Report tensor movement, finite values, norms, rank
and singular values; keep common action offsets separate from action differences.
On the same fixed TRAIN probe capture activation ranges and final-head
contributions. Raw magnitude alone is not feature importance or proof of cause.

Read the actual small head matrices/biases when interpreting diagnostics.
Derive any proposed edit from an identified computation and test it on an
isolated copy. No arbitrary weight adjustment based only on a large norm.
The source checkpoint, recipe, scaler and account semantics remain immutable.

## Test and Decision Gates

| ID | Required check |
|---|---|
| W10 | Matrix/tensor summaries detect nonfinite data and distinguish zero rank from a rank-one matrix |
| W11 | Delta accounting reports unchanged tensors exactly and groups role/component consistently |
| W12 | Common-action linear projection leaves categorical softmax and action differences invariant |
| W13 | Effective copied actor/critics reproduce native logits/Q on the fixed TRAIN probe before inspection |
| W14 | Source file hashes unchanged; no optimizer step or dataset fitting during the diagnostic |
| W15 | Every proposed behavior-changing edit gets full paired native accounts, activity and directional metrics before selection |

Run focused CPU tests before diagnostics; use GPU for full support evaluation
and later explicitly specified training contrasts. Do not infer trading
improvement from a critic-only calibration gain or a function-preserving edit.
Retain Retsu's forward work as a parallel track, not a prerequisite here.

## Completed Execution

Both GPU selection passes, CPU module inspection, four GPU head interventions
and physical selected-channel export are complete. See
[225_PHASE2_69_AND_WEIGHT_SPACE_RESULTS.md](225_PHASE2_69_AND_WEIGHT_SPACE_RESULTS.md).
Seventeen focused tests passed, followed by full-support real GPU parity checks.
Source70 remains unchanged; selected69 is a separate inference bundle, not a
freshly retrained model. No degraded head transplant was selected.

Next research action: qualify bounded refinement from the preserved policy,
with explicit actor drift/activity checks and a critic calibration control;
the long degrading head-only schedule is not automatically repeated. Selection
and forward integration remain independent. Completing these short diagnostic
jobs does not mean a long-running training job is still active.

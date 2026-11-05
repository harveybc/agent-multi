# Attention baseline and regularization placement

## Authorized successor: early-core BatchNorm

Owner direction2026-11-05 adds R4 only after R3 finishes, on the same secondary
accelerator. No additional control or dropout run. R1 remains unchanged.
R4 moves the single critic BatchNorm from the flattened head input to after
the first causal convolution of the first core residual block, before SiLU.
The shortcut, second convolution, second core block and actor are unchanged.
Normalize each (channel,time-position) independently across the batch: flatten
[B,C,T] to [B,C*T] for BatchNorm and reshape back, never average across T.
Use the same affine, eps1e-5, momentum0.1 and running-statistics settings as R3.
Weights and replay start fresh; retain common initialization, data/support,
scaler, attention, seed, optimizer, monitor and business rules.

Acceptance before dispatch: tests verify exact placement and single application,
TRAIN and eval suffix-perturbation invariance, per-position running statistics,
unchanged common initialization, actual twin-critic updates, eval freezing,
cold reload and interrupted-run parity. Dispatch must wait for the predecessor
service to finish successfully and authenticate its result before starting R4.
The R3 zero-trade selected checkpoint is an inadmissible activity outcome, not
evidence of generalization. Moving BN is a placement experiment, not a claim
that replay/target distribution mismatch has been solved.

Deployment2026-11-05: code ed586112, CPU61 passed/14 skipped. Enabled resumable
successor service reports WAITING_FOR_PREDECESSOR, zero restarts, while R3 is
in epoch40 with patience19/20. The launcher authenticates R3 source/input/result
and exported checkpoint, prepares R4 atomically, checks GPU readiness, runs
CUDA acceptance, then starts the sole authorized fresh R4 fit. CPU tests also
reject missing, failed and nonzero-exit predecessors. No overlap with R3 and
no changes to R1. GPU acceptance and training have not yet run at this check.

Owner direction2026-11-05: retain causal patch attention as the development
baseline and isolate regularization mechanisms and their placement. This is
an engineering choice, not a claim that attention has established predictive
or generalization superiority. Preserve the no-attention control and frozen
compact93 reference. Do not alter the two running immutable201 runtimes.

## Evidence and limits

At the latest reviewed checkpoints, attention epoch20 has TRAIN1637.72% and
VALIDATION-15.62%; no-attention epoch35 has TRAIN830.14% and VALIDATION-16.63%.
These are different selected epochs from one seed; better fitting does not
establish better useful pattern recognition. Compare matched epochs as well
as checkpoint-selected outcomes. Retain Sharpe, drawdown, activity and cost.

Invalidated earlier campaigns are not evidence for this comparison. This work
remains in feature-selection phase2, with the current93-input representation.

The prior warmed-up108-input stopping contrast (195) favored validation-only
over guarded mean in observed validation return: -3.4537431% vs -16.46431855%.
Keep validation net return/week as checkpoint criterion; TRAIN is measured
but cannot compensate for declining validation. This is one observed contrast,
not proof of a universal stopping metric.

## Required mode correction before new arms

Current `app/three_signal_learning.py:emit` and bundle probes use
`torch.inference_mode()` without explicit module evaluation mode. Current
dropout probability is zero and BatchNorm is absent, so this is not evidence
of stochastic-dropout corruption in201. It is an unsafe extension point for
new regularized arms. No new arm may launch until tested mode handling exists.

For full TRAIN score, VALIDATION, action emission and exported inference probes:
disable dropout, freeze all BatchNorm running statistics, make no optimizer or
replay updates, and restore every prior module mode afterward, including errors.
Evaluation must not advance training RNG. Refit or periodic learning does not
change this distinction: gradient phases train; score/inference phases evaluate.
MC dropout/Bayesian posterior sampling is a separate explicitly declared mode.

SAC loss has online actor, online twin critics and delayed critic evaluations;
do not assume one global train()/eval() call defines all three. Specify training
sites for stochastic masks and target evaluation, authenticate each twin's BN
buffers, update target buffers intentionally, and prove replay/save/load parity.
Adding BatchNorm alone is not a CrossQ implementation.

## Authorized contrast (owner correction2026-11-05)

Keep attention,93 features,13x2 window,scaler,support,costs,wd0.01,batch128,
learning rate,seed and business contract fixed. Fresh training only for R1 and
R3. Reuse the completed attention baseline as control; do not retrain R0.
Use the core output flattened from[B,16,13] to[B,208] immediately BEFORE
the existing dense8 head. This is downstream of temporal feature extraction,
not noise on source data or on the final3 logits/Q outputs.

| Arm | Actor | Online critics | Placement |
|---|---|---|---|
| R0 | none | none | completed attention baseline; reuse history, no new fit |
| R1 | none | Dropout p0.10 | twice inside each core residual branch |
| R3 | none | BatchNorm1d208 | after core flatten, before dense8; critics only |

The alternative post-core dropout arm R2 is not scheduled. Only R1 and R3
are authorized. R1 runs on the primary accelerator and R3 on the secondary
accelerator concurrently. No successor training is automatically authorized.

R3 starts with affine=True,eps1e-5,momentum0.1,track_running_stats=True;
these are declared starting values, not optimal values. Statistics update only
from TRAIN replay during declared online-critic training. Evaluation uses saved
statistics, never validation batches. Each temporal coordinate is a separate
feature; unlike applying BatchNorm1d directly to[B,C,T], this does not average
different time positions together inside the window. Replay-distribution
shift remains a measurable risk. Batch-size1 inference must work.

Do not combine dropout and BatchNorm in this first contrast. Combining them is
not mathematically forbidden, but separate arms identify effects more clearly.
R1 keeps the identity shortcut untouched and the existing SiLU activations.
It applies dropout after the first activation and after the second convolution.
If useful, subsequently compare dropout0.05/0.20.
Only then consider the attention/core boundary or group extractors. These
follow-up locations must be separate arms, not simultaneous hidden changes.
Dropout on latent activations is not equivalent to moving-average smoothing:
the former is stochastic training regularization disabled for ordinary inference;
the latter transforms inputs at training and inference.

## Acceptance tests before dispatch

1. Repeated eval outputs deterministic; different TRAIN masks produce stochastic
   outputs; both use unchanged inputs. All modes restored after normal/error exits.
2. BatchNorm buffers change only in authorized TRAIN updates. VALIDATION cannot
   alter them; one-row and batched inference agree within declared tolerance.
3. Actor/critic/target placement and effective configuration match the arm;
   baseline common initialization unchanged; no regularizer on final logits.
4. Actual TorchRL functional twin-critic CUDA update and checkpoint restoration
   preserve buffers, modes, finite gradients and cold-reload predictions.
5. Both train and eval preserve core prefix causality; no mean over temporal
   positions before causal layers. Native trades and costs still pass controls.
6. Same source/support/scaler/periods and matched monitor semantics in every arm.
   Compare full native TRAIN/VALIDATION, trade count, buy/sell wins, early closes,
   net return, Sharpe, DD and inference overhead. These runs do not read TEST.

Warm-up issue from201 is separate: retain the best inference checkpoint from
epoch1 while delaying patience increments until epoch21. Before dispatch,
specify and test which earlier best initializes the patience reference, including
tie/min_delta boundaries. Compare against201 at matched epochs and replay the
new monitor over its existing TRAIN/VALIDATION history without training or
additional evaluation; retain its original selected result separately. The
reused control predates explicit train/eval mode handling; mathematical dropout0
and no-BN behavior is unchanged, but no bit-identical new training trajectory is
claimed across the mode implementation or device change.
Save the best global checkpoint and separately the best checkpoint from epoch20
onward. This diagnostic retention does not reset patience or change the primary
selection rule. A transient early validation peak is not proof of learned skill;
inspect activity, drawdown and replication. An epoch spans the full TRAIN account,
roughly24025 decisions and6006 replay updates, not a single gradient iteration.

## Bayesian sequence

Preserve the separate Bayesian-head contrast from201. Do not mix it into
R0-R3. Compare normal versus Bayesian head on the chosen architecture with
fixed controller first; a later experiment can use uncertainty for sizing.

## Sources and execution state

- [PyTorch Dropout](https://docs.pytorch.org/docs/stable/generated/torch.nn.Dropout):
  evaluation is identity; training randomly masks and rescales activations.
- [PyTorch BatchNorm1d](https://docs.pytorch.org/docs/stable/generated/torch.nn.modules.batchnorm.BatchNorm1d.html):
  running statistics and track_running_stats govern evaluation behavior.
- [DroQ](https://arxiv.org/abs/2110.02034): critic dropout plus LayerNorm is an
  established research direction, not evidence that our proposed placement wins.

Status2026-11-05: exactly R1 and R3 are running on their assigned accelerators.
Both services are enabled, with zero restarts at the dispatch check. CPU acceptance:
50 passed,13 skipped. GPU acceptance:31 passed on each assigned accelerator.
Both input preflights authenticated before native training started.
The completed attention/control runs remain unchanged.

## Dispatch and diagnostic artifacts

Owner-corrected allocation: fast worker R1; secondary worker R3. R0 is reused;
R2 is not launched. Each service resumes its own atomic epoch checkpoint after
an interruption and stops when its one authorized arm finishes.
Never overlap these workers with the preceding attention/control training.
Each run writes TRAIN_VALIDATION.png, TRAINING_DIAGNOSTICS.png and EPOCHS.csv
from already computed full-epoch accounts. No extra scoring, TEST access or
training RNG consumption is permitted. Return uses separate aligned axes for
TRAIN and VALIDATION so large TRAIN gains cannot hide validation deterioration.
Diagnostics include drawdown, Sharpe, orders/week, win rate, stopping score and
patience. Mark the chosen checkpoint and warmup; missing metrics stay missing,
never become zero. CSV records available per-epoch update diagnostics, explicitly
labelled as last-update values rather than epoch means. Refresh by overwriting
the same artifacts atomically; no per-epoch PNG accumulation or OLAP blobs.
Backfill current attention and control from their immutable HISTORY.json files.

Tests before dispatch: artifact values match history; generation is repeatable,
works with one epoch and missing metrics, preserves RNG/source files, uses no
GPU and marks the selected checkpoint (not just peak validation). Core and
post-core regularization require real TorchRL twin-critic update/reload tests.

CPU acceptance2026-11-05:50 passed,13 skipped (GPU/native fixtures).
New regularized recovery tests cover all four arms, exact resumed weights,
best-from-epoch1 retention and independent post-warmup bundles. Registration
order keeps common twin-critic parameters identical across arms: TorchRL's
functional expansion otherwise consumes extra random draws for BN parameters
before initializing the dense head. Optional BN is registered after the head
but evaluated before it. Config receipts name critic-only regularization and
its actual probability/BN setting rather than retaining misleading defaults.

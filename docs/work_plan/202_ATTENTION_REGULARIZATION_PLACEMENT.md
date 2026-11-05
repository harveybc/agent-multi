# Attention baseline and regularization placement

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

Plan47 describes dropout/L2/BatchNorm in older dense SAC critics with15 inputs.
Plan65 describes critic-dropout removal under the old COMMANDS5 contract.
Neither establishes the result of a core-placement comparison on current
93-input discrete SAC, grouped TCNs and post-fusion attention. Those documents
prove historical designs/deployments, not current-architecture final metrics.

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

## Initial small contrast

Keep attention,93 features,13x2 window,scaler,support,costs,wd0.01,batch128,
learning rate,seed and business contract fixed. Fresh training for all arms.
Use the core output flattened from[B,16,13] to[B,208] immediately BEFORE
the existing dense8 head. This is downstream of temporal feature extraction,
not noise on source data or on the final3 logits/Q outputs.

| Arm | Actor | Online critics | Placement |
|---|---|---|---|
| R0 | none | none | matched control with new verified mode handling |
| R1 | none | Dropout p0.10 | after core flatten, before dense8 |
| R2 | Dropout p0.10 | none | same location, actor only |
| R3 | none | BatchNorm1d208 | same location, critics only |

R3 starts with affine=True,eps1e-5,momentum0.1,track_running_stats=True;
these are declared starting values, not optimal values. Statistics update only
from TRAIN replay during declared online-critic training. Evaluation uses saved
statistics, never validation batches. Each temporal coordinate is a separate
feature; unlike applying BatchNorm1d directly to[B,C,T], this does not average
different time positions together inside the window. Replay-distribution
shift remains a measurable risk. Batch-size1 inference must work.

Do not combine dropout and BatchNorm in this first contrast. Combining them is
not mathematically forbidden, but separate arms identify effects more clearly.
If useful, compare dropout0.05/0.20 and then place the same mechanism on the
residual branches within the core; leave the identity shortcut untouched.
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
   net return, Sharpe, DD and inference overhead. TEST remains closed.

Warm-up issue from201 is separate: retain the best inference checkpoint from
epoch1 while delaying patience-based stopping until epoch20. Before dispatch,
specify and test which earlier best initializes the patience reference, including
tie/min_delta boundaries; apply the same rule to R0-R3. Do not compare new arms
to201 as though checkpoint selection had remained identical.

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

Status: design and acceptance requirements recorded. Regularized implementation
and new GPU jobs are NOT claimed complete or launched.201 continues unchanged.

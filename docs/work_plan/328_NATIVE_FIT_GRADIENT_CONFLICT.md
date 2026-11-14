# Native FIT Gradient Conflict

Owner approved the bounded successor to327 on2026-11-14. TESTS_PASS_PREPARED.
17 focused tests pass, including failure-path parameter restoration.
No training, optimizer step, source mutation, new features or model promotion.

## Frozen Design

Use all eight327 arms at their shared matching322 initial control and saved
8192/32768 endpoints:24 scoped observations. These are initial/primary/late,
not the64-update early transient: that point has metrics but no saved weights.
Do not reconstruct or choose a checkpoint based on stop/audit performance.

For each fold use only original FIT and the disjoint added FIT from324.
Expanded FIT is their union, original weight1/4 and added weight3/4.
Never evaluate stop, audit,2024 or TEST rows. Frozen encoders were trained
on all TRAIN, so this is an internal fitting diagnosis, not new generalization
evidence. Preserve source299,71 inputs,49 return candidates and paused STL.

For each twin's three-action squared error, decompose
MSE = mean-action level error + centered action error. Centered error is
one third of the mean squared error over the three pairwise action gaps.
Compute each component's gradient over the arm's actual trainable parameters:
54 output-only, or3398 full-head. Report full active vector and hidden/output
subvectors separately. Frozen parameters are excluded, not assigned fake
nonzero gradients.

Measure original/added alignment, original/expanded alignment and level/gap
alignment. Negative gradient dot product means descending one increases the
other locally; publish cosines, norms and normalized directional derivatives.
A zero norm yields undefined cosine, never a fabricated conflict. Check loss
and gradient decomposition, and expanded-gradient weighted-union identities.

Verify directional derivatives with symmetric finite differences at1e-4 and
1e-5 along normalized descent directions of added MSE, expanded MSE and
expanded level error. Evaluate original/added total/gap losses. Use CPU
float64 copies of the saved float32 tensors, restore every tensor exactly,
and publish numerical discrepancies. Keep checkpoint/data file hashes fixed.

This diagnoses raw-loss geometry, not the realized minibatch AdamW update:
momentum, preconditioning, weight decay and discrete greedy regret are not
equivalent to a full-batch MSE gradient. No differentiability claim for greedy
regret, no causal claim about a whole training trajectory from one cosine.

## Requirement To Evidence

| Requirement | Test / runtime check |
| --- | --- |
| Correct total/level/gap decomposition | Translation and analytic-gradient tests |
| Correct original/added weighting | Disjoint support tests, weighted-gradient identity |
| Correct conflict sign | Known opposite and aligned vectors, zero-norm case |
| No parameter mutation | Before/after tensor equality including failed perturbation |
| Correct derivative scale | Central finite differences at two fixed steps |
| Same authenticated weights | Parent seals, checkpoint hash, trainable-scope counts |
| No held-out calculation | Explicit FIT subset construction and exclusion assertions |

Use evidence to propose one FIT-defined preservation/ranking or replay-weight
contrast. Do not change the loss or deploy a new model within this diagnosis.

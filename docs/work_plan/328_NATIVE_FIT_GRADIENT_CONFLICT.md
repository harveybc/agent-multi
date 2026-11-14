# Native FIT Gradient Conflict

Owner approved the bounded successor to327 on2026-11-14. COMPLETE_NO_PROMOTION.
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

## Results And Independent Verification

Runtime commit `0326c6bc`, identity
`7cb718972f3c7786a586bc30a767bb5b4c9fa984bb287fcb8daba965bc67b005`.
24 scoped observations completed in1.01 seconds on CPU. Source unchanged;
no optimizer steps, GPU use or held-row evaluations.17 focused tests pass.

An independent NumPy implementation differentiated both linear layers and
SiLU explicitly, without PyTorch autograd. All216 gradient vectors agree
(maximum absolute difference8.535e-16).360 alignment summaries and576
finite-difference witnesses were verified. Maximum numerical directional
error4.035e-8; loss discrepancy2.221e-16. Signs are numerically supported,
not an inference from a training plot alone.

| Scope | Updates | Old/new MSE conflict | Expanded descent harms old MSE | Expanded descent harms old gap | Level descent harms old gap |
| --- | ---: | ---: | ---: | ---: | ---: |
| Output only | 0 | 3/4 | 3/4 | 3/4 | 0/4 |
| Output only | 8192 | 4/4 | 4/4 | 4/4 | 0/4 |
| Output only | 32768 | 4/4 | 4/4 | 4/4 | 0/4 |
| Full head | 0 | 3/4 | 3/4 | 2/4 | 2/4 |
| Full head | 8192 | 4/4 | 0/4 | 1/4 | 4/4 |
| Full head | 32768 | 4/4 | 2/4 | 3/4 | 4/4 |

The old/new centered-gap gradients also conflict in4/4 pairs at both trained
endpoints for both scopes. Such opposing gradients can occur normally near
a compromise optimum: their existence alone is not a programming defect or
a proof of the cause of the entire training trajectory.

The detailed result is more specific. With the hidden layer frozen, descending
expanded MSE locally worsens original MSE and original gap error in all trained
endpoints. With full heads, the combined direction helps original MSE in4/4
at8192 despite opposing old/new gradients; it harms original gap error in
only1/4 there. Therefore it would be incorrect to say every expanded update
necessarily damages the old examples.

Full-head common-level correction locally harms old gap error in4/4 at8192
and32768, through the hidden subvector. The corresponding output-subvector
cross term is numerically orthogonal. Shared hidden representation therefore
provides a measured route for level/gap interference, but freezing it is not
a solution:327 already showed output-only fitting loses old decisions.

## Scalar Loss-Weight Feasibility Witness

Consider only the initial full-head checkpoints and the raw descent direction
of `lambda * expanded_level_MSE + expanded_gap_MSE`, with lambda>=0.
For old-gap MSE not to increase to first order, the necessary inequality is
`lambda * dot(g_old_gap, g_exp_level) + dot(g_old_gap, g_exp_gap) >= 0`.
The saved gradients imply:

| Fold / seed | Necessary lambda bound |
| --- | --- |
| A /19 | lambda <=16.334771 |
| A /43 | lambda >=0.490191 |
| B /19 | lambda <=-3.440100: impossible for nonnegative lambda |
| B /43 | lambda >=40.770032 |

B19 has both dot products negative (-0.000547952 and -0.001885008).
Thus lowering the common-level weight alone cannot make this specific local
gap-preservation condition hold in every initial case. Even without B19,
A19 and B43 bounds are incompatible. This is NOT a proof about finite AdamW
updates, discrete regret, eventual learning or the feasibility of a broader
constrained direction. Do not treat these inequalities as deployable weights.

## Next Step

Before another long fit, test a bounded update-preservation mechanism against
the actual proposed AdamW parameter displacement on FIT, not only a projected
raw gradient. Check the changed decisions on original FIT and learning on the
expanded FIT; retain the unchanged control and report infeasible/stalled steps
instead of silently accepting zero learning. Momentum, preconditioning and
finite-step effects must be included. A small gradient or lower common-value
MSE alone cannot certify safe action ordering. This next mechanism has not
been implemented or launched by328; no loss or source actor was changed.

Evidence: [full results](generalization/native_fit_gradients_20261114/RESULT.json),
[summary table](generalization/native_fit_gradients_20261114/RESULTS.md),
[independent backpropagation check](generalization/native_fit_gradients_20261114/INDEPENDENT_VERIFY.json),
[scalar feasibility calculation](generalization/native_fit_gradients_20261114/SCALAR_FEASIBILITY.json).

# Actor Gradient Coupling Diagnosis

2026-11-11. Owner approves identifying the source of282's broad suppression of
beneficial opening margins. This is a diagnostic, not a loss/strategy change.

## Frozen Scope

Authenticate the complete282 receipt, its69-input source bundle and151 TRAIN
anchors. Use the recorded first FIT minibatch for each of the six arms; the38
MONITOR anchors never contribute gradients. Reconstruct the actor's logit
Jacobian against its four trainable head tensors. No representation, features,
periods, source weights, live model or business rule changes. No TEST reads or
new floating pass in this step.

Decompose each direction-minus-NOP margin into:

1. Independent-logit descent at that observation (three-action softmax effect).
2. That observation's own shared-head gradient, without other observations.
3. Weighted same-observation and cross-observation contributions from the exact
   sampled FIT batch, including repeated sample multiplicities.
4. Each head tensor's contribution to the batch gradient projection.
5. First-step AdamW adaptive update and decoupled weight decay, checked against
   the saved ACTOR_0001 checkpoint; compare linearized and actual margin changes.

Repeat the algebra with native realized-return Q targets as an offline oracle
control. This is not an input, a live action rule or a trained candidate.
Separate Q and entropy, and retain the distinction between a margin change and
an actual order. Report all six arms, both roles and all state strata, including
native-beneficial/Q-beneficial comparisons and all selected controls.

## Requirements And Tests

| ID | Requirement | Prespecified evidence |
|---|---|---|
|G1|Exact actor objective|Closed-form logit gradient equals autograd; common Q offsets invariant|
|G2|Interference attribution|Independent-logit fixture has no cross-row influence; shared fixture reproduces cross-row reversal; self+cross equals total|
|G3|Duplicate batch treatment|Repeated indices reproduce autograd mean, not unique-row averaging; MONITOR never sampled|
|G4|Optimizer separation|Fresh-state AdamW formula equals isolated native optimizer step with decay; saved first-step parameters replay exactly|
|G5|Authenticated observational replay|Source receipt, input support, panel logits and first minibatch match282; no source mutation|
|G6|Complete reporting|Six arms plus native-target controls, tensor projections, residual errors and explicit next gate; no promotion|

## Execution

Run sequentially on the primary GPU when free, bounded by a finite supervisor.
Replay optimizer steps only in disposable in-memory copies to reproduce the
recorded transition; export diagnostic arrays/reports, never replacement model
weights. Stop rather than reinterpret an identity, gradient or replay mismatch.
The selected subset remains69 features/28 groups and the previous qualified
reference and six282 checkpoints remain intact.

## Completed Evidence

Runtime46905161 completed six arms in6.35seconds on the primary RTX5090,
with2.2GB peak process memory. The collector authenticated all three receipt
files, including the full decomposition arrays. Every saved first-step actor
checkpoint reproduced exactly. No new checkpoint was exported. The additional
final CPU report took0.30seconds. All34 focused tests pass, including finite-difference
probability derivatives and duplicated-minibatch weighting.

Both services exited0. Analytical AdamW first-step parameter differences from
the real optimizer are at most2.93e-8 (floating-point rounding). Linearized
versus actual margin differences are at most7.36e-6. The source receipt and
weights remained unchanged. This is an exact diagnostic of the first sampled
update, not proof of every future training trajectory.

### Three Different Effects

The probabilities below refer to native-optimal openings among NOP/BUY/SELL on
the same-state frozen-policy continuation, not to perfect market foresight or
new orders. There are73 such FIT openings and24 MONITOR openings. The monitor
is held out from the gradient, and its own-row results are hypothetical
diagnostics only. First we replace Q by the measured realized return, removing
critic estimation error without changing the actor or batch.

Number of native-optimal opening probabilities pushed down:

| Scope / control | Independent logits | Own-row shared-head step | Full FIT-batch shared-head step |
|---|---:|---:|---:|
| FIT, native-return oracle, seed19 |2 /73|11 /73|69 /73|
| FIT, native-return oracle, seed43 |2 /73|11 /73|61 /73|
| MONITOR, native-return oracle, seed19 |0 /24|3 /24|24 /24|
| MONITOR, native-return oracle, seed43 |0 /24|3 /24|23 /24|
| MONITOR, absolute critic19 |16 /24|18 /24|24 /24|
| MONITOR, absolute critic43 |17 /24|21 /24|24 /24|

1. **Critic/local categorical effect remains.** With predicted Q, even the
   independent-logit direction lowers16-17 of24 optimal monitor opening
   probabilities. The native-return control lowers none there. Pairwise
   calibration therefore did not finish the ranking repair.
2. **Shared-parameter interference is measured independently of critic error.**
   Under local logit descent the oracle improves21 monitor optimal-opening
   probabilities, with3 negligible changes at the declared1e-10 threshold and
   zero appreciable declines; the FIT-batch head update reduces23-24. Own-row versus
   cross-row decomposition isolates this coupling; MONITOR contributes no
   training gradient. In absolute19,26 FIT direction comparisons have positive
   native and predicted advantages over NOP, yet their shared batch margins
   decrease. This is not merely a sign mistake in those Q comparisons.
3. **AdamW is not an implementation failure here.** The first step matches the
   saved parameters exactly. Its actual nonlinear update reduces23/24 and22/24
   optimal monitor opening probabilities for absolute19/43, versus24/24 under
   infinitesimal batch SGD. It changes the details, but suppression already
   exists before adaptive preconditioning. Decoupled weight decay's projected
   L2 magnitude is0.076-0.092% of the adaptive term on these flat FIT/MONITOR
   cohorts, so it is not the main driver in this probe.

Margin changes alone would overstate the conclusion. In absolute19 the actual
step reduces all24 useful monitor margins against NOP, but only23 corresponding
probabilities; the third action also moves. The report checks both quantities.
Even with exact Q, a three-action policy-gradient step need not increase the
probability of every individually best action: this occurs in2 of73 FIT cases.
Do not turn pairwise comparisons into an invented binary objective.

### Where The Shared Effect Lives

For the absolute critics, the first head layer's weight matrix accounts for
61.3-66.7% of the summed absolute projections on FIT's optimal-opening margins,
and53.8-56.5% on MONITOR. The final weight matrix accounts for25-32% approximately;
the remaining contribution is from biases. These are decomposition magnitudes,
not isolated causal ablation improvements or percentages of trading loss.
Freezing one layer would change the update and must be tested separately.

The FIT margin Jacobian has226 rows and1699 head parameters, with numerical
row rank226 at relative singular threshold1e-6. Its leading direction contains
27.85% of squared sensitivity, the first five61.64%. Thus this local panel does
not exhibit a hard rank limit that makes independent margin changes impossible.
It does not prove adequate global capacity or prescribe increasing network size.

### Interpretation And Next Contrast

The update is optimizing a shared expected-return objective, not promising to
improve every row. Even the oracle batch's mean expected native return rises
on the flat monitor while most optimal-opening probabilities fall: gains on
other decisions compensate. This is a measured learning tradeoff, not a broken
accounting engine, reversed optimizer sign or evidence that inputs lack signal.
It helps explain why refinement can improve the aggregate while suppressing
particular profitable openings and why calibrating Q alone is insufficient.

The next narrow contrast should target **how a desired per-observation policy
change is translated into shared parameter changes**, retaining the same critic,
features and source actor. Compare native head AdamW with a damped, bounded
function-space projection and a last-layer-only control, using TRAIN only and
prespecified small drift budgets. First require finite-difference/local-target
agreement, then full native trading, activity and closing checks in both seeds.
These are proposed tests, not a claimed fix or an already-running experiment.
Do not infer that reducing batch size, removing entropy or globally enlarging
the network will solve the measured coupling. Keep the improved282 candidate
separately eligible for export qualification and a frozen floating pass; this
diagnostic does not revoke its actual account gains.

Evidence: [manifest](generalization/actor_coupling_20261111/MANIFEST.json),
[gradient summary](generalization/actor_coupling_20261111/GRADIENT_SUMMARY.json),
[probability and spectrum report](generalization/actor_coupling_20261111/PROBABILITY_SUMMARY.json).
Worker identity:
`49cbbecb0f8e6701475bcce863838331ae2bbad9ecb6477370110a5a3b7ec0ce`.
Report identity:
`f4b8c27c03549cccec4e187e2c8fb59d15f2d93f951a0e0e069e56d8d227f9ac`.

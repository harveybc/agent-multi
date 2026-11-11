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

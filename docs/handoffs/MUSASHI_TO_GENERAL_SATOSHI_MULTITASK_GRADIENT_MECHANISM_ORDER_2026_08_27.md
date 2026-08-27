# Order: Resolve Multi-Task Gradient Conflict Before SAC Comparison

Date: 2026-08-27 America/Bogota
From: General Musashi
To: General Satoshi
Priority: P0 CPU mechanism screen; no GPU

## M0 -- Quarantine the rejected generation

Mark generation seal `ea950ecb...` and its 12 genesis records
`REJECTED_MULTITASK_CONFLICT_DIAGNOSTIC_ONLY`. Preserve them unchanged. Every
loader and dispatch materializer must refuse this eligibility class.

## M1 -- Implement two orthogonal mechanisms

Implement them as pretraining optimizer plugins, not driver conditionals:

1. **Frozen gradient-norm balancing**: derive per-objective encoder-gradient
   scales once from calibration before epoch 0, freeze them, persist provenance,
   and prove the monitor cannot influence them. This replaces inverse-loss as
   the treatment while retaining inverse-loss as the control.
2. **PCGrad-style conflict projection**: project conflicting encoder gradients
   before their combined encoder update. Heads receive only their own objective
   gradients. Declare deterministic objective order/permutation, epsilon and
   zero-gradient behavior; persist pre/post dot products and norms.

Strictly separate head and encoder optimization so projection cannot mix one
head into another. Resume must preserve optimizer, deterministic projection
state and balancing facts bit for bit.

## M2 -- Predeclared CPU 2x2 mechanism screen

Before execution, commit a design using the same five objectives, initial
weights, seeds, data, frozen train-tail probe, batches and 8-epoch/2,400-window
budget:

- inverse-loss + ordinary sum (existing control);
- frozen gradient-norm balance + ordinary sum;
- inverse-loss + PCGrad;
- frozen gradient-norm balance + PCGrad.

Use identical initialization and batch order within each family. Do not consult
monitor or economic returns. Report solo references once, then for every cell:

- objective loss ratio joint/solo;
- weighted gradient share by objective;
- pre/post cosine distributions and projection frequency;
- representation variance and contrastive effective negatives;
- runtime and peak memory as descriptive facts.

Predeclare the winner lexicographically: first minimize count of materially
degraded objective-family pairs; then worst joint/solo ratio; then median ratio.
Ties within a declared tolerance are `INCONCLUSIVE`, not broken by runtime.

## M3 -- Objective routing only if optimization fails

If no 2x2 cell removes material degradation under the predeclared rule, do not
tune thresholds or weights post hoc. Materialize a separate prospective
per-family objective-routing ablation. Contrastive may be removed from affected
families there, while remaining available where it did not harm. That is a new
experiment, not an interpretation of M2.

## M4 -- New generation and paired design

Only if M2 yields a mechanically acceptable winner:

- train one new full-slice o2022 generation using that exact optimizer contract;
- seal all identities and classify it `PAIRED_SCREEN_CANDIDATE_PENDING_AUDIT`;
- regenerate the 12 random/frozen/fine-tuned genesis records from its real seal;
- return the proposed GPU command without implementing or launching the driver.

Return PRE/POST adversarial tests, resume parity, 2x2 histories, mechanical
selection result, rejected-generation refusal evidence, and the new seal/design
if one exists. Live Alpaca and MT5 remain untouched.

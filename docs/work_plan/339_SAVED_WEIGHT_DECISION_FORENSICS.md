# 339: Saved-Weight Loss And Decision Forensics

Status: COMPLETE_VERIFIED; read-only, no model promoted.

## Results And Interpretation

Runtime commit `ca30d3f3`. All4 cases,8 arms,20 saved points,24 transitions
and384 four-block hybrids completed. Execution of the bounded analysis took
2.28s, not a training run. A separate CPU run repeated every case. The GPU
run adds full-FIT CUDA gradient witnesses. Code development, tests and
interpretation are not included in that execution time.

Evidence: [tables](generalization/weight_forensics_20261114/RESULTS.md),
[all measurements](generalization/weight_forensics_20261114/RESULT.json),
[row events](generalization/weight_forensics_20261114/EVENTS.csv),
[component plot](generalization/weight_forensics_20261114/COMPONENTS.png),
[cross-host verification](generalization/weight_forensics_20261114/VERIFICATION.json).
GPU identity `2994608b1953d5b3a4e8fbabba6fdfc62320948fefac6bebec5db2b4d8ce4270`;
CPU identity `8e0967126c7db47e66cc4ee1ab64dab7ba3708484803ece2b62bd0a6bbc64be9`.

### Main Finding: Shared Updates Can Reverse Correct Decisions

Complete-protection source-to256 results over the FULL expanded FIT panel:

| Case | FIT rows | Cost before -> after | Greedy regret before -> after | Better / worse / equivalent |
|---|---:|---:|---:|---:|
| A/19 |196|0.100499732 ->0.095285494|0.156435285 ->0.143093188|6 /2 /3|
| A/43 |196|0.095305457 ->0.090771205|0.141511388 ->0.137134156|4 /1 /1|
| B/19 |392|0.111643395 ->0.109345482|0.193833237 ->0.189182553|6 /2 /3|
| B/43 |392|0.115912840 ->0.113151363|0.231144661 ->0.222863006|8 /4 /3|

These are native same-state reward-target costs, NOT account profits or new
validation returns.338's primary table used stable-added FIT only, explaining
the different regret values.43 changes are case/row occurrences, not43
independent market observations. All9 harmful changes lose a previously
target-optimal action on the expanded, not originally protected, FIT rows.
The original protected decisions remain guarded as verified in338.

At the source checkpoint, the own-example shared-weight gradient favors
preserving the better action in9/9 harmful endpoint changes. In4/9, gradients
from OTHER examples overwhelm that favorable contribution. CUDA32 confirms
the signs. This is measured gradient interference, not an inferred absence
of useful input information.

Example A/43,row257: BUY -> SELL loses1.062454 native target-return units.
Own-example contribution to the BUY-minus-SELL margin is+0.118546, other
examples contribute-0.351846, total-0.233300. The strongest negative role
contribution is other flat/openable rows, not necessarily conflicting tasks.
Simply splitting opening from closing heads would not remove this witness.
These numbers are local directional derivatives, not realized P&L changes.

### The Surrogate Can Improve While The Executed Action Worsens

In3/9 harmful source-to256 changes, the SAME ROW's cost falls, not just the
average cost. A/19,row316 changes BUY -> NOP. Probabilities in NOP/BUY/SELL order:

- Before: `[0.382607,0.383653,0.233741]`.
- After: `[0.391096,0.381188,0.227716]`.
- Targets: `[0.141937,0.576651,-0.400867]`.
- Row cost improves0.174950796 ->0.173976562 while greedy regret rises0.434713.

Reducing probability of the worst action helps the probability-weighted loss,
even though NOP overtakes the best action. No numerical or accounting bug is
needed for this mismatch. It does NOT mean cost loss is universally useless:
all four complete arms improve their aggregate FIT regret. It means loss
alone cannot certify each executed decision or the useful refinement path.

### No Universal Bad Layer

All16 before/after combinations of the four hidden/output blocks were
evaluated per transition. Exact Shapley values account for interactions;
their sum equals the measured change. Components do not have independent
global merits. A/43's second critic blocks oppose the net FIT improvement,
whereas A/19 benefits from all four blocks. B/43's first output block opposes
the improvement while both hidden blocks help. Do not permanently freeze or
reset a block merely because of one retrospective attribution.

Selected local witness A/43,row257 has margin attribution:
Q0-hidden-0.0172423; Q0-output+0.0085852;
Q1-hidden-0.1071290; Q1-output+0.0122469. Sum-0.1035392 matches the
observed margin change. This identifies the finite endpoint-weight effect,
not the unique training-step cause. No hybrid was selected or promoted.

### Optimizer History And Nonlinearity Remain Distinct

Saved AdamW effective-moment directions oppose the better-action margin
in7/9 harmful cases. In3 of those, full-FIT plain gradient descent initially
favors that margin. Endpoint displacement aligns more strongly with the
initial effective moment direction (cosine0.866-0.923) than with the initial
full-FIT negative gradient (0.245-0.516). This is a diagnostic association:
moments summarize minibatches, and the recorded direction excludes the new
gradient, weight decay, clipping and finite protection projection. It is
not proof that AdamW itself is broken or that resetting it fixes the result.

B/43,row530 is especially informative: first-order prediction along the
total endpoint displacement is+0.008856, but actual margin change is-0.010740;
the minimum twin changes. A tangent at the source cannot explain the entire
finite path. The existing complete-twin protection repair remains necessary.

### Hypotheses Not Supported By These Measurements

- No action disagreement in20 CPU32/CPU64 reconstructions, nor in their
  full-FIT CUDA action checks. Two points each have one minimum-twin routing
  difference; GPU/full-double gradient cosines remain >=0.99972577. These
  small derivative differences are recorded, not called bitwise equality.
- NumPy64/Torch64 forward error <=7.11e-15; analytic gradient error<=8.33e-17;
  route-stable finite-difference error<=6.68e-10. All20 points have a smooth
  finite-difference witness. Cross-host common discrete evidence is exact;
  1958 continuous fields differ by at most6.94e-12, not byte-identically.
- No gradient clipping in the16 endpoints' recorded continuations.
- No collapsed matrix rank: hidden effective ranks6.94-7.30 of8; output
  effective ranks2.69-2.96 of3. Minimum hidden activation standard deviation
  across FIT is0.10655. These are not capacity-sufficiency proofs.
- No exact duplicate twin embeddings among the196/392 FIT rows. This does
  not refute316's different-position identifiability witnesses elsewhere.
- Encoders, inputs and actor did not change here. This analysis attributes
  critic-head changes; it does not claim to have exhausted all possible
  diagnoses of the temporal representation or the whole RL system.

### Next Bounded Diagnostic

Before another long fit or unchanged floating pass, replay the authenticated
complete-arm trajectories to their first harmful action crossings. Separate
the actual sampled-batch gradient, updated AdamW displacement, projection
and accepted finite step. Compare them against full-FIT/own-row directions
at THAT checkpoint, not only the source tangent. Include the row530 twin
switch witness and row316 loss/greedy mismatch. Fix the support, seeds,
native targets and original complete/single/batch protection. No held labels
choose an update, and no actor promotion comes from this diagnostic alone.
Only then contrast the responsible update rule or execution-aware criterion.
This is the next planned task, not an already executed new training run.

The analysis and reporting are complete.50 focused regression tests pass
after final reporting changes; scoped Ruff passes. The initial launch had
an SSH quoting error in the device-name argument and never entered the
analysis; it was stopped and corrected. Both actual executions completed.

## Original Frozen Protocol

Owner requests detailed analysis of the weights following338. This is a
read-only diagnostic: no optimizer steps, actor updates, new inputs, new
floating pass,2024/TEST reads or live promotion. Source299 remains unchanged.
Analyze all four cases and both338 arms, not only the best or worst endpoint.

## Questions And Measurements

1. Reconstruct all saved native heads at source,64,256 in independent NumPy64
   and Torch CPU32/64. Measure margins, twin routing, action changes and
   numerical disagreements before interpreting improvements/regressions.
2. For source->64,64->256,source->256, measure weight/activation drift,
   singular values and exact four-block interventions: hidden/output of each
   twin. Evaluate all16 combinations. Exact Shapley decomposition of changed
   action margins and FIT metrics must sum to the endpoint change. Hybrids
   are diagnostic interventions, never candidate models or trained policies.
3. Derive per-row cost gradients analytically and verify against Torch
   autograd and directional finite differences. For every changed FIT action,
   separate independent-logit, own-example shared-weight, other-example and
   role-group effects on its old-versus-new action margin. Compare these LOCAL
   derivatives with the actual finite displacement, without pretending that
   the endpoint difference is a single gradient step or the full training path.
4. Measure alignment of saved AdamW moments and effective moment directions
   with full-FIT gradients and endpoint movement. Moments are minibatch-history
   quantities, not the exact full-FIT gradient. Do not infer a bug from cosine
   disagreement alone.
5. Attribute discrete regret changes to rows and roles. Held internal TRAIN
   rows are descriptive only: no gradient fit, coefficient choice, candidate
   selection or intervention adoption from held results. Whole-TRAIN encoder
   pretraining prevents calling these independent end-to-end holdouts.

## Test Design

| Requirement | Evidence |
| --- | --- |
| Native reconstruction | NumPy/Torch forward and analytic-gradient parity |
| Correct surrogate derivative | Finite differences and common-logit shift invariance |
| Exact component attribution | Additive/interacting known functions; permutation enumeration and efficiency identity |
| Shared-gradient decomposition | Own+other=total and role groups sum; conflicting-example fixture |
| No silent geometry mistake | CPU32/64 action disagreements and margin magnitudes recorded per changed row |
| No mutation or data drift | Parent seals, checkpoint/source hashes and unchanged actor verified at end |
| Honest interpretation | Local tangent predictions separated from finite interventions; no claim of exhaustive weight decoding |

Freeze this protocol before analyzing the new measurements. Existing338
metrics/curves motivated the questions, not a particular layer verdict.
Write reproducible JSON/CSV/plots and a report with measured causes, limits and
the next bounded test. Do not blame inputs or recommend abandoning the task.

Prelaunch:64 focused CPU tests pass, including native3398-parameter point
reconstruction, held-label perturbation with unchanged FIT gradients, exact
hybrid decomposition and common-shift/finite-difference checks. Finite-difference
epsilon reduces from1e-6 to1e-10 only to preserve the same minimum-twin branch;
if no smooth witness exists, report it explicitly rather than calling it a
gradient error or relaxing tolerance. Full-FIT CUDA32 gradients and role effects
are additional numerical witnesses, not optimizer steps or minibatch replay.

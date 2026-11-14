# 339: Saved-Weight Loss And Decision Forensics

Status: IMPLEMENTED_TESTED_BEFORE_EXECUTION.

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

# 341: Expanded FIT Optimal-Decision Protection

Status: IMPLEMENTED_TESTED_BEFORE_EXECUTION.56 focused tests and scoped Ruff pass.

Owner authorizes continuation after340. Use all ten sealed first-crossing
proposals, without changing their source weights, batches, AdamW proposal or
original protection contract. No actor/input change or held-label fitting.

## Contract

1. Establish target-optimal action SETS from native expanded FIT outcomes.
   Exact ties remain equally valid; no arbitrary ordering of tied targets.
2. Protect every row whose current action is optimal in all five inference
   geometries. Rows with all three outcomes tied need no constraints. Incorrect
   predictions remain free to change. Keep every original anchor guard.
3. For each eligible row and each suboptimal action, require one optimal
   witness action to beat it through BOTH chosen critics. At each relinearized
   center choose the currently highest-Q allowed action; no inequalities are
   imposed between equally optimal actions. The added boundary is zero,
   with the actual finite action-set check authoritative after float32 casting.
4. Retain original interior thresholds (2x original margins), primal solver,
   maximum3 repairs,8 scales and4x proposal-norm cap. Never relax original
   guards. Accept only positive motion with strictly lower full-FIT cost and
   non-increasing full-FIT greedy regret in every geometry. Rejection is a
   result, not permission to change the contract after seeing it.
5. Publish paired original-protection control versus new protection per saved
   proposal. Checkpoint any admitted diagnostic state; never promote it.

## Test Design And Traceability

| Requirement | Test/evidence |
|---|---|
| Do not freeze wrong predictions | Eligibility excludes non-optimal predictions; negative fixture |
| Preserve tied optima | No pair between optimal labels; finite guard allows switching between them |
| Correct derivative | Manual pair Jacobian versus Torch autograd for both twins and alternative minimum |
| Finite behavior, not just linear | Boundary fixture and original/single/batch checks after casting |
| Zero threshold contract | Zero allowed, negative/nonfinite rejected; no disguised positive constant |
| Useful update | Nonzero displacement, lower cost, no regret increase; zero-motion/unchanged fixture rejected |
| Reproducibility | Parent seals, cold endpoint checks, repeat one paired repair from saved weights |
| Scope | FIT-only labels/gradients; actor/source hashes unchanged; no2024/TEST reads |

Feasibility gate for the next bounded continuation: at least8/10 useful
updates and at least one per fold/seed case, all finite guards valid. Report
actual decision improvements separately from loss improvements; a flat action
metric is not called better trading. If the gate passes, the next task is a
matched short continuation from334, not a long fit or another unchanged
floating pass.341 itself is the fixed-proposal contrast and verification.

# 341: Expanded FIT Optimal-Decision Protection

Status: COMPLETE_INDEPENDENTLY_VERIFIED.56 focused tests and scoped Ruff pass.

## Measured Result

Runtime `a68c0ac4`, verifier `3258d1c5`;
identity `e61891e71590f0823a830b53bc708410d7906f561c365eeab0311552959de200`.
Ten paired proposals completed in11.93s. All ten original controls reproduce
their saved states/receipts exactly and violate at least one newly protected
decision. All ten new candidates have positive motion, lower cost and unchanged
FIT actions/regret in all five geometries. Nine use scale0.5, one scale1.0.
111-175 target-optimal rows are protected per witness, versus49/98 original
anchors. No thresholds were relaxed after seeing results.

Independent verification checked5616 added Jacobian constraints against
Torch autograd, maximum difference1.78e-14; all ten endpoints, optimal sets,
original guards and proposed optimizer/RNG lineage pass. First repair repeats
exactly and all ten endpoints cold-reload exactly. Gate passes10/10 with all
four cases represented. This proves safe nonzero local movement, NOT improved
greedy decisions or account trading. [342](342_OPTIMAL_PROTECTION_SHORT_CONTINUATION.md)
tests whether accumulated updates improve decisions under the same contract.

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

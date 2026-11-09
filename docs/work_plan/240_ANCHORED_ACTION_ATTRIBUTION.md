# Anchored Action Attribution

Owner authorized2026-11-09 following239. No training, threshold fitting or
model promotion. Reuse fixed256-update KL endpoints for r19/r43 and unchanged70.
Compare both TRAIN and VALIDATION, including all changed decisions, not just
the harmful ones. Exact native account attribution is the primary evidence.

## Sequence

1. Authenticate239 collection and source identities. Reload parent and both KL
   endpoints. Check emitted categorical signals against archived execution
   tapes and verify the frozen corrected critics against calibration checkpoints.
2. Reuse the tested native same-state suffix interventions and chronological
   telescoping attribution. Every contribution must reconcile to full account
   difference; isolated contributions are separately labeled and never summed.
3. At every changed row report parent/current logits, probabilities, margins,
   per-row KL, twin Q values and minimum-Q gap(new minus old). Compare Q ordering
   to the realized discounted reward gap from identical-state parent-policy
   continuations, separately from the undiscounted account cash difference.
4. Distinguish opening changes from early-close changes using actual position
   and commands; publish helpful, harmful and cash-neutral cases plus totals.
   Complete both periods/seeds even when outcomes differ. Generate CSV and MD.

No causal claim about market predictability follows from a Q ordering error.
No probability margin threshold is chosen using validation, no state is added
to model observations, and no new training or strategy rule follows automatically.
Position exists only in the native simulator and retrospective diagnostic.

## Test Gates

| Requirement | Test/evidence |
|---|---|
| Reject incorrect dimensions, NaNs and mismatched actions | event helper fixtures and native tape parity |
| Parent/current margins and KL calculated with correct signs | hand-calculable opposite-action fixture |
| Cash and discounted-reward differences remain distinct | deliberately opposing-sign fixture |
| Correct opening vs early-close classification | open-position and flat fixtures |
| Full cash sum exact, snapshots unchanged | existing native attribution regressions plus real four accounts |
| No fitting or source mutation | zero optimizer calls, source/checkpoint hashes before/after |

Execution uses inference on the original GPU class for signal parity and CPU
for native suffixes. Full private event state stays outside the public repo.
Public report contains compact aggregates and selected event metrics only.

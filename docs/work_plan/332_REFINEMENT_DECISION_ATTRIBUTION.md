# 332: Refinement Decision Attribution

Status: IMPLEMENTED_AND_VERIFIED_BEFORE_EXECUTION.

Automatic continuation after331's extension gate refused longer fitting:
identify the remaining decision regressions rather than requesting another
user approval or extending a loss-only improvement.

For all eight256-step endpoints, replay their full expanded-FIT native GPU
outputs against the saved starting state. Attribute changes in action regret
row by row and by operational role, excluding any stop/audit/validation label.
Split native MSE exactly into common-level and centered action-gap error.
On CPU64 copies, inspect the already-tested original/added-FIT gradients with
finite-difference witnesses, and split the saved weight displacement into
hidden/output layers. Do not infer causation from weight norm alone.

Acceptance checks: native metric replay within1e-12; sum of row regret changes
equals aggregate regret change; MSE decomposition exact within1e-12; frozen
source and parent seals unchanged; finite differences pass the existing328
gradient checker. No optimizer updates, new target, feature change or promotion.

Deliver an explicit measured diagnosis distinguishing common-level improvement,
action-gap improvement and discrete action quality. This is a diagnostic, not
a retrospective checkpoint selection or claim of trading improvement.

Prelaunch:22 attribution/interior/projection tests and7 gradient tests pass.

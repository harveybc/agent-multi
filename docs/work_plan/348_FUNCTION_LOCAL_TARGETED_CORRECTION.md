# 348: Function-Local Targeted Corrections

Status: IMPLEMENTED_READY.50 focused tests and scoped Ruff pass. Each paired
target must also reproduce its saved344 state with identity-metric repair
before running lambda1. Automatic independent CPU verification follows.

Compare ONLY the10 verified accepted344 corrections, at their frozen witnesses
and targets, against one fixed locality variant. This is a conditional probe,
not an estimate across every attempted correction. No actor/optimizer training,
additional target search, feature change,2024/TEST read or live promotion.

For each target, G is the manual Jacobian of all three unordered min-Q action
gaps at every OTHER expanded-FIT row, evaluated at the original weights.
Common shifts of all action values are excluded by construction. No held row
or held label enters G or update selection. Fix lambda=1, not a tuned sweep.
With P parameters and s=||G||_F^2, H=I+(P/s)G^T G. If s=0 use H=I.
Each linearized repair minimizes

    0.5 ||delta - raw_AdamW_delta||^2 + 0.5 (P/s) ||G delta||^2

under the SAME original-anchor, optimal-set and target constraints. H remains
fixed at the original state across relinearizations. Cholesky whitening maps
this SPD quadratic to the existing independently checked primal projection;
verify original-space constraints after mapping back. Finite tests use actual
nonlinear outputs in all five geometries. Original motion limit4x, three
relinearizations, eight halving scales, target margin, strict FIT cost/regret
improvement and exact old decision guards remain unchanged.

Measure collateral gap drift as mean squared difference of the three action
gaps, excluding the target, separately in all five geometries. Label a new
candidate pairwise-qualified only if it passes every existing finite gate,
strictly reduces collateral drift in ALL five views and has no higher FIT
regret than its saved344 control in ANY view. Lower drift alone is insufficient.
Record failures without widening limits. Held partitions are descriptive only.

|Requirement|Acceptance evidence|
|---|---|
|Correct gap function|Common-Q-shift invariance and target exclusion unit tests|
|Exact SPD objective mapping|Closed-form unconstrained and constrained2D solutions; identity metric negative control|
|Reject invalid numerical inputs|Nonfinite matrices, zero motion and inconsistent shapes rejected|
|Original gates unchanged|Original-space residual, raw norm cap, five-view finite checks and target correctness|
|Matched control|Cold344 candidate and its original assessment reproduce exactly|
|Independent derivatives|Torch64/min-Q autograd versus manual G; target Jacobian checks|
|Repeatability|Every accepted candidate cold-check and exact repeated solver/receipt|
|Honest paired selection|Unit tests reject better drift with worse regret or a failed geometry|
|No promotion|Actor299 hash and all parent seals unchanged; continuation_allowed=false|

Independent CPU verification follows automatically and checks finite NumPy
metrics, stored target identity, collateral drift and pairwise verdicts. No
long fit follows from a single-witness success. Preserve71 inputs,28groups,
49 cumulative return candidates; STL remains paused.

# 348: Function-Local Targeted Corrections

Status: COMPLETE_INDEPENDENTLY_VERIFIED.50 focused tests and scoped Ruff pass.
All10 identity-metric controls reproduce saved344 states. Automatic CPU
verification and a separate local CPU verification both pass byte-exactly.

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

## Results

Runtime/verifier:d54377cc. Identity
`ba756e655725f93626cd811365247f399670103ccb9bc8db7f867b144eb467aa`.
Ten paired probes finished in29.18seconds with no restarts. Six local variants
pass the original finite gates; five ALSO pass the stricter paired rule of
lower collateral drift in all five views and no higher FIT regret than344.
All six accepted candidates cold-reproduce and repeat exactly. All ten
identity-metric repairs reproduce their original344 candidate weights.

|Witness / target|Local finite gate|Paired verdict|Other-FIT gap MSE reduction|FIT regret control / local|
|---|---|---|---:|---:|
|A19/32 /332|Pass|Qualified|98.46%|0.149094201 /0.149094201|
|A19/32 /316|Pass|Qualified|96.99%|0.148733808 /0.148733808|
|A43/238 /246|Pass|Qualified|95.94%|0.129825620 /0.129825620|
|A43/238 /253|Rejected|Not qualified|Not admitted|0.128855336 /not admitted|
|A43/238 /243|Rejected|Not qualified|Not admitted|0.129312137 /not admitted|
|B43/88 /457|Pass|Qualified|96.42%|0.229738595 /0.229738595|
|B43/88 /402|Rejected|Not qualified|Not admitted|0.229806033 /not admitted|
|B43/144 /402|Rejected|Not qualified|Not admitted|0.225138172 /not admitted|
|B43/152 /434|Pass|Rejected: worse paired regret|95.17%|0.224879363 /0.225979739|
|B43/152 /402|Pass|Qualified|97.10%|0.225204422 /0.225204422|

The drift numbers are MEAN SQUARED differences of other-row action gaps, not
profit, accuracy or percent changes in action probabilities. For the five
qualified pairs the corresponding RMS reduction is79.86%..87.60%. Each keeps
the target correction and the same aggregate FIT regret as its saved control.
All five qualified pairs also have EXACTLY the same complete FIT action vector
as their controls in each of the five geometries (zero action differences).
These ten controls come from FIVE historical witnesses in THREE cases
(A19,A43,B43); none is a B19 witness. This is not a direct rerun of347's two
missed-gain trajectories or a generalization result.

The four rejected proposals have original-space motion ratios5.570,7.895,
6.408 and4.173, exceeding the unchanged4x limit. The second also reports
unsuccessful SLSQP line search, despite a small residual. No limit or solver
acceptance condition was relaxed. The valid but unqualified target434 is an
important counterexample: locality alone would sacrifice another gain. The
paired regret condition prevents adopting it.

For all six finite candidates, internal TRAIN stop/audit actions and greedy
regret are identical to the corresponding controls; soft costs vary. No
held-action improvement is claimed. The encoder used TRAIN, so these internal
partitions are not end-to-end held-out validation. Neither2024 nor TEST was read.

## Verification

- Gap Jacobian versus independent Torch64 autograd: max error2.22e-15.
- Target Jacobian max error2.00e-15.
- Original-space normalized linear constraint residual<=1.04e-13 across
  proposals, including rejected proposals; finite guards decide admission.
- Ten exact identity-metric controls; six cold checkpoints and six repeated
  directions/solver receipts exact.
- CPU verifier reconstructs native NumPy outputs, target/optimal-set guards,
  cost/regret, all five stored geometry metrics, locality and paired decisions.
- A separate local CPU run produces VERIFICATION.json byte-identical to the
  automatic verifier. Parent seals and actor299 hash are unchanged.
- Checkpoints explicitly prohibit continuation and contain no optimizer.

## Decision And Next Bounded Step

Function locality is useful on these fixed examples but is not a universal
replacement. Use a conservative candidate-selection rule in the NEXT test:
first let the original345 procedure select its target/correction exactly as
before; then try lambda1 locality for THAT SAME target. Substitute it only
when all original gates pass, other-FIT drift is lower in all five geometries,
and paired FIT regret is no worse. Otherwise preserve the original correction
exactly. Never use held scores to choose the variant or target.

Implement/test that rule before a separately pinned16-attempt continuation
from the four334 sources, against saved345 controls, with the SAME total four
target searches per case and explicit projected-AdamW moments. A locality
attempt is additional compute, not an additional target search; report both.
Cold/resumed checks and internal transfer reporting remain required. No actor
update, longer fit or unchanged floating pass follows from these probes alone.
Keep71 inputs,28 groups and49 cumulative return candidates unchanged.

[Table](generalization/function_locality_20261114/SUMMARY.csv),
[results](generalization/function_locality_20261114/RESULT.json),
[verification](generalization/function_locality_20261114/VERIFICATION.json),
[identity](generalization/function_locality_20261114/CONFIG.json).

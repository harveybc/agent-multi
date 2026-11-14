# 332: Refinement Decision Attribution

Status: COMPLETE_MEASURED_ATTRIBUTION_NO_PROMOTION.

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
Final combined regression suite:43 passed in3.62s. Repository correctness-rule
Ruff check passes for all five newly added code/test files in331-332.

## Results

Runtimece67baf0, identity
`122f32e0a1fdaeb30e384ddc43ec37db34e80140e5c00b6112bbc141471ae39b`.
Eight checkpoints analyzed in0.763s with zero optimizer updates. Independent
NumPy backpropagation reproduces72 gradient vectors to5.52e-16 maximum absolute
error;192 finite-difference witnesses pass (max error1.94e-8). Native GPU
expanded-FIT metrics replay exactly and row/role sums reconcile.

| Interior case | Helpful decision changes | Harmful decision changes | Common-level share of MSE reduction | Expanded regret change |
| --- | ---: | ---: | ---: | ---: |
| A19 | 6 | 4 | 89.24% | -0.001668 |
| A43 | 5 | 1 | 85.20% | -0.007371 |
| B19 | 1 | 4 | 87.37% | +0.002391 |
| B43 | 2 | 2 | 91.28% | +0.000698 |

All nonzero decision changes occur on flat states permitting both opening
directions. No measured early-close improvement occurs in this contrast.
Both the common-level AND centered action-gap MSE improve in all four interior
cases. Thus it would be incorrect to say merely removing the common level
necessarily solves the remaining regressions. Mean squared action-gap error
can improve while a few consequential argmax decisions worsen. The actual
row witnesses are retained, not inferred from a global loss curve.

Old/added gap-gradient cosines at interior endpoints are negative in all four
cases (-0.2845,-0.4918,-0.4999,-0.3984), but these are local unprojected
derivatives. The new finite protection demonstrably handles original decisions;
these cosines do not prove that no safe learning direction exists or explain
every harmful change. Weight displacement norms are descriptive, not causal
proof of a particular bad layer.

## Batch-Geometry Witness And330 Clarification

At the saved B43 starting state, exact same embeddings for original row66 give
action2 when evaluated with the239-row guard, but action1 in the392-row expanded
FIT batch and in a one-row call. The two leading Q values differ by about
6e-8; there are no input changes, dropout, stochastic sampling or retraining.

A fresh native GPU replay of both330 successful-regret proposals (offsets2,4)
shows that their ONLY action change is this original row66, from1 to2 in the
expanded batch. Its regret reduction0.1829129573 /392 accounts for the whole
reported0.0004666147 improvement. The numbers were reproduced correctly;
their interpretation must be narrowed: this is numerical decision stabilization,
not a newly learned decision on the added-FIT rows.

The same row explains one of B43's two helpful changes in331; the other is an
added-FIT row. All four interior256 endpoints show zero original-FIT action
disagreements across guard, expanded and single-row geometries in this new
witness. Scores still differ by up to1.31e-6, so equality of scores is not
claimed. The robust decision margin, not exact float equality, is relevant.

## Operational Conclusion

The authorized sequence continued without another user approval: repair,
eight paired fits, automatic budget decision, replay, and detailed attribution.
No technical blocker or missing user decision remains for this block. All
jobs finished successfully; the original actor/model is unchanged.

Further authorized diagnostics should distinguish stabilization from learning,
check all three batch geometries, and contrast a decision-cost-aware objective
against this authenticated MSE control before another long refinement. A pure
gap-MSE substitution is not justified as a proven fix by these results. No
model is promoted and no feature-selection pass is restarted on that assumption.

## Evidence

-[Full attribution, changed rows and gradients](generalization/refinement_attribution_20261114/RESULT.json)
-[Runtime identity](generalization/refinement_attribution_20261114/CONFIG.json)
-[Independent gradient verification](generalization/refinement_attribution_20261114/INDEPENDENT_VERIFY.json)
-[Native batch-geometry and330 witnesses](generalization/refinement_attribution_20261114/BATCH_WITNESS.json)

# 335: Rank-Preserving Value Calibration

Status: IMPLEMENTED_VERIFIED_BEFORE_EXECUTION.

334 improved cost-only FIT decisions but increased absolute Q error. Calibrate
copies of its eight cost-only endpoints (four cases x256/1024), without
optimizer updates, actor changes, new observations or any held-label fitting.
This does not repair mixed decision transfer: a rank-preserving transform
cannot change greedy action regret in exact arithmetic.

Use the SAME positive affine transform on every action and both twins:
Q'=a*Q+b. Then min(Q'1,Q'2)=a*min(Q1,Q2)+b and argmax is unchanged for a>0.
Bake it into each twin's final weight and bias, preserving hidden tensors.

Two predetermined variants, not a coefficient search:

1. Shift-only: a=1, b=mean(y-Q) over expanded FIT and both twins.
2. Protected affine: a=max(cov(Q,y)/var(Q), a_min), b=mean(y)-a*mean(Q).
   a_min is the maximum required-margin/current-gap ratio over all original
   protected FIT rows and guard/expanded/single GPU32/CPU32/NumPy64 views.
   It is computed from inputs/predictions/frozen margins, not held labels.

Use NumPy64 expanded-FIT scores to solve the coefficients. No retuning after
finite-network evaluation. Serialize all16 candidates, including failures.
Accept this diagnostic calibration only if actual original guards pass,
all expanded-FIT actions remain unchanged in every geometry, and MSE does not
increase in GPU32/CPU32/NumPy64 (tolerance1e-10). Report strict improvement
separately. Do not relax margins or adjust coefficients on a failed case.

## Test Design

| Requirement | Evidence |
| --- | --- |
| Correct closed-form coefficients | Recover known transform; constrained optimum and grid witnesses |
| Positive shared scale | Reject invalid scale, target shape, nonfinite data and infeasible source margins |
| Exact mathematical rank invariance | Twin-min and argmax identity in double precision |
| Real parameter transform | Hidden tensors bit-identical; baked affine forwards match formula |
| Data separation | Coefficients only expanded FIT; stop/audit readouts descriptive, no selection from them |
| Robustness | Original margin guards plus all-FIT and held action-geometry checks |
| Reproducibility | Parent seals/source identity, candidate tensor hashes, cold reload and independent NumPy metrics |

Read TRAIN stop/audit only after fitting; encoder was pretrained on whole
TRAIN, so these are not an independent end-to-end generalization test.
Keep source299 actor,71 inputs,28 groups and49 cumulative return candidates.
No2024/TEST, new floating pass, model/live promotion or automatic SAC adoption.

Prelaunch:38 focused CPU tests pass in3.71s, scoped Ruff correctness checks
pass. The first test run caught an overly broad hidden-tensor name predicate
in the test; replaced by exact hidden-layer keys, without changing the
implementation or weakening rank/parameter invariants.

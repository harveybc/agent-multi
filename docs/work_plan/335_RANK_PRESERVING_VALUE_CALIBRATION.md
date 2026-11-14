# 335: Rank-Preserving Value Calibration

Status: COMPLETE_VERIFIED_13_OF_16_DIAGNOSTIC_CALIBRATIONS_PASS.

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

## Completed Results

Runtime `f9ba2352`, identity
`7baa1a87b259b0c40623c86da6d8eafe2ff79aa12667908fceed975aa4ea8492`.
All16 candidates completed in3.143s, supervisor exit0,0 restarts. Eight source
endpoints and16 calibrated cold reloads reproduce exactly. No optimizer/actor
updates, no model promotion; the chain has completed and no worker is running.

| Variant | Candidates | Strict FIT MSE improvement | FIT actions unchanged | Pass margins |
| --- | ---: | ---: | ---: | ---: |
| Shared offset | 8 | 8 | 8 | 8 |
| Shared positive affine | 8 | 8 | 8 | 5 |

Every source has at least the valid shift-only candidate. Both variants also
preserve all evaluated TRAIN stop/audit actions in five execution geometries.
This improves FIT value calibration without discarding ranking; it cannot
improve decision transfer or trading results by itself.

Three affine candidates fail a fixed margin despite unchanged argmax:

| Case | Failed checks | Worst margin deficit |
| --- | --- | ---: |
| A43/1024 | CPU guard/expanded/single, GPU single | 7.583e-8 |
| B43/256 | GPU guard | 2.367e-8 |
| B43/1024 | NumPy guard/expanded | 8.694e-9 |

All three have the analytic slope bound active. They remain rejected: no
rounding of deficits to zero or guard relaxation. Maximum baked-forward
deviation from ideal a*Q+b is1.193e-6. Exact-arithmetic rank invariance is
not a sufficient finite-precision margin certificate.

## Independent Verification

17 seals,16 candidates,64 hidden tensors bit-identical,96 metric scopes.
Independent least-squares coefficients and manual NumPy forward metrics agree
within4.441e-16. No FIT/held overlap and no independent guard disagreements.
Local CPU MSE differs from producer CPU by at most5.141e-10; regret is exact
and margin-surplus difference<=4.441e-16. The verifier initially demanded
1e-12 for CPU values, diagnosed that small arithmetic difference, and then
separated value tolerance from strict decision/margin acceptance. It did not
change the experiment's1e-10 MSE gate or any margin. Do not claim CPU value
arithmetic is byte-identical across machines.

Evidence: [table](generalization/affine_calibration_20261114/RESULTS.md),
[results](generalization/affine_calibration_20261114/RESULT.json),
[verification](generalization/affine_calibration_20261114/INDEPENDENT_VERIFY.json).

## Stall Attribution And Next Action

Read-only classification of334 B43's trace:567 trailing rejected attempts
after the last acceptance at457;329 finite-network guard failures,238 solver
feasibility failures,0 terminal norm-cap failures. Solver normalized residuals
are1.0035e-7..2.5396e-7 against the fixed1e-7 requirement; norm ratios
0.9391..0.9651 are below4x. This distinguishes numerical feasibility from
RAM/GPU availability. It does not prove that a more accurate solve would
also pass the nonlinear finite guard.

Next: bounded paired replay of these rejected actual AdamW proposals with an
independent higher-accuracy direction solve, unchanged margins/norm cap/data,
real finite-network checks and decision-cost scoring. No tolerance relaxation,
new user decision, unchanged floating pass or automatic promotion. This next
diagnostic is not launched by335. Preserve299,71 inputs,28 groups,49 cumulative
return candidates and paused STL. Calibrated fixed-label values are not proof
of full Bellman/actor training correctness.

Trace receipt: [classification](generalization/affine_calibration_20261114/STALL_CLASSIFICATION.json).

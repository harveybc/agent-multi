# 334: Optimizer Memory Contrast

Status: COMPLETE_VERIFIED_FIT_IMPROVEMENT_TRANSFER_MIXED.

333 changed the loss while retaining AdamW moments learned under MSE. Test
whether those moments impede adaptation. This is not an established cause.
Reuse all12 authenticated333 inherited-state controls; run12 fresh-optimizer
fits (MSE/cost/hybrid x foldsA/B x seeds19/43). Same exact initial weights,
3398 trainable parameters, all expanded FIT rows, schedules starting at8192,
batch128, LR3e-5, decay.01, clip10, temperatures, scoring masks and numerical
guards. Reset AdamW first/second moments AND its step count, nothing else.
Source299 actor, encoders,71 inputs and paused STL remain unchanged.

## Test Design And Gates

| Requirement | Acceptance |
| --- | --- |
| Reset only optimizer | Exact head tensor digest unchanged; fresh state empty/step0 |
| Preserve inherited mode | Exact loaded state and count; invalid modes refuse |
| Fair loss-by-loss pairing | Initial tensor/support/baseline receipts exact against333 |
| No false improvement | Same stable-added FIT scoring; original actions protected |
| Reproducibility | Every endpoint cold replay;3 resumed transitions per arm |
| Autonomous continuation | Fixed256 attempts; qualify1024 automatically or record failure |

For each of the THREE objectives separately, qualify extension only if all four
endpoints are robust, all accept>=32, none worsens stable-added regret versus
its own start by>1e-8, and>=3/4 improve by>1e-8 versus BOTH that start AND the
same objective's inherited-state333 endpoint. Extend only qualifying fresh
arms to1024. The256 matched comparison remains the decisive contrast;1024 is
an additional trajectory, not a matched256 treatment-effect comparison.
No coefficient search, retrospective endpoint selection or held-data tuning.

Keep AdamW weight decay and proposal projection unchanged. A reset effect
cannot distinguish first moment, second moment and bias-correction age; that
requires another contrast. Cost-only still lacks calibrated Q offsets. No
actor promotion, floating-selection restart or trading claim from this test.

Prelaunch:51 focused CPU tests pass in3.68s; scoped Ruff correctness checks
and git diff checks pass. The runner's default inherited mode is preserved;
fresh mode requires an authenticated inherited control and refuses mismatched
support, source, temperature, budget, optimizer hyperparameters or initial
weights. Both modes keep checkpoint/resume and automatic extension logic.

## Descriptive Transfer Audit

After the frozen training/extension chain, evaluate ALL recorded256 endpoints
and any1024 endpoints on324's TRAIN-only stop/audit panels, versus their
initial states and inherited333 counterparts. No selection/refit from this
readout. Added after observing the256 gate, before reading those panels.
They are label-disjoint from FIT, but the fixed encoder was pretrained on
whole TRAIN: this is not an independent end-to-end generalization result.
Freeze starting numerical stability masks without using labels, report full
and stable action regret, changes by role, and five execution geometries.
No2024/TEST or account replay is part of this small diagnostic.

## Completed Results

Runtime `ebb061ca`, identity
`7603f573875f72abf0cad6e4166b6a3328c21915819f903245fa65b6b486b41e`.
All12 short fits finished. Cost-only passed the predeclared gate in4/4 cases;
its four1024 continuations started automatically, without user intervention.
MSE and hybrid did not qualify. Total502.74s,6144 attempted updates,4811
accepted, supervisor exit0 and0 restarts. No current worker remains running
after completion and collection.

Stable-added FIT regret, lower is better, not trading performance:

| Fold/seed | Start | Inherited cost256 | Fresh cost256 | Fresh cost1024 |
| --- | ---: | ---: | ---: | ---: |
| A19 | .300857 | .285340 | .266233 | .208441 |
| A43 | .297515 | .293475 | .242987 | .188652 |
| B19 | .299639 | .298168 | .287018 | .252696 |
| B43 | .326445 | .328914 | .310984 | .304705 |
| Unweighted mean | .306114 | .301474 | .276806 | .238623 |

The matched256 comparison improves8.1826% on average,4/4 cases. The1024
trajectory continues improving all four relative to256, but has4x the update
budget: do not call that an equal-budget reset effect. Late B43 accepts395
of1024 attempts and its last acceptance is step457; all567 subsequent attempts
are rejected. A19 accepts655, A43 881, B19 869. Protection still constrains
learning; it has not been removed or silently weakened.

This isolates a real optimizer-state effect under this specific cost-only
loss. It does not prove that optimizer memory explains all SAC refinement
failures, or that resetting AdamW universally helps. MSE reset has mixed
decision effects; hybrid still regresses B43. Cost-only MSE increases: better
ranking is not equivalent to calibrated absolute Q values.

## Independent Verification

13 seals,28 serialized endpoints, exact initial weights/supports/baselines
against333, all optimizer counts verified. Independent NumPy metric error
<=1.111e-16;0 protected-action failures.28 native cold replays and36 resumed
transitions are exact.12 starting masks and all28 stable-added endpoints have
0 disagreements across five CPU/GPU/single/NumPy geometries.52 focused tests
pass after the reporting fix below, in3.92s.

The transfer audit evaluates64 scopes with8 frozen numerical masks. Independent
manual NumPy forward reproduces all64 within4.441e-16;0 FIT overlap and0 action
geometry disagreements. Its identity is
`d9855861adf340903b34ab8d91bf7cefc8016891d303a679920cf259d4b71353`.
All readouts are TRAIN-only, with a previously pretrained encoder.

| Mean stable regret | Start | Inherited cost256 | Fresh cost256 | Fresh cost1024 |
| --- | ---: | ---: | ---: | ---: |
| Internal stop | .263512 | .260413 | .269151 | .269353 |
| Internal audit | .231290 | .229515 | .227313 | .228387 |

Transfer is mixed: final stop improves versus start only in B43; final audit
improves in3/4 (A19 worsens). No joint generalization claim or actor promotion.
The full per-role and per-case results include harmful changes, not only means.

## Reporting Correction

Visual inspection caught that the original shared x-axis stopped at256,
clipping the1024 continuation despite correct saved JSON/checkpoints. The
plotter now takes the maximum step over ALL objectives and has a regression
test. Published CURVES.png is a corrected derivative generated outside the
sealed run; the original sealed artifact was not modified. No refit needed.

Evidence: [table](generalization/optimizer_reset_20261114/RESULTS.md),
[training results](generalization/optimizer_reset_20261114/RESULT.json),
[replay](generalization/optimizer_reset_20261114/REPLAY.json),
[verification](generalization/optimizer_reset_20261114/INDEPENDENT_VERIFY.json),
[geometry](generalization/optimizer_reset_20261114/GEOMETRY_AUDIT.json),
[transfer](generalization/optimizer_reset_20261114/TRANSFER_RESULT.json),
[transfer verification](generalization/optimizer_reset_20261114/TRANSFER_INDEPENDENT_VERIFY.json),
[corrected curves](generalization/optimizer_reset_20261114/CURVES.png).

## Exact Next Action

Before another actor update or floating pass, separate ranking improvement
from value calibration: test a shared positive affine output calibration on
expanded FIT, verify actual finite-network actions/margins across all recorded
geometries, and report both Q error and regret. No new input, arbitrary data
filter, optimizer reset of the production model, or held-label fitting.
This calibration is not launched or certified by334. If it preserves the new
ranking but fails the value/transfer check, retain299 rather than silently
installing the cost-only critic. Investigate B43's late protected-step
infeasibility as a separate diagnostic, not a reason to loosen guards.
No user choice is needed to implement these bounded tests.71 inputs,28 groups,
49 cumulative phase2 return candidates and paused STL remain unchanged.

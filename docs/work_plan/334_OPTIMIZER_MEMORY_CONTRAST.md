# 334: Optimizer Memory Contrast

Status: IMPLEMENTED_VERIFIED_BEFORE_EXECUTION.

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

# 333: Decision-Cost Loss Contrast

Status: IMPLEMENTED_VERIFIED_BEFORE_RUN.

## Scope And Frozen Design

Continue332 automatically, not another intermediate approval. Source299 actor,
71 inputs, frozen encoders and targets remain unchanged. No2024/TEST, live
promotion, data thinning or feature-selection restart. Four fold/seed starts
are the same protected8192 states used in331, with identical AdamW moments,
batch schedules, LR, decay, clipping and interior correction. This is a native
critic diagnostic, not a new trading result or proof of Bellman calibration.

Three objectives (12 paired fits):

- `mse`: exact existing twin-head MSE.
- `cost`: q is the twin minimum; costs c_a=max(y)-y_a. With temperature tau
  frozen to the median strictly positive cost over expanded FIT, use
  L_cost=tau*mean(sum(softmax(q/tau)*c)). This is a smooth decision surrogate,
  not the actual discrete regret or a calibrated uncertainty estimator.
- `hybrid`: L_MSE+L_cost, coefficient1 fixed before execution. Both terms have
  squared-return units. No coefficient search or held-panel choice.

The temperature is fitted once on the same expanded FIT targets, never on a
minibatch or held data. Refuse zero/nonfinite scales. Preserve mean offsets in
MSE/hybrid; cost-only is explicitly insensitive to common Q offsets and cannot
be promoted as a calibrated value critic on this evidence alone.

## Guards And Honest Attribution

Keep331's fixed original actions,1e-5 margin,2x interior target,4x norm cap and
at most three linearizations. Every proposal must pass original239 and
expanded-FIT geometry in GPU32, CPU32 and NumPy64. At saved checkpoints also
verify original rows individually in GPU32 and CPU32. Do not call legacy
starting states robust when they fail this new check.

Before training, freeze a stable-added-FIT scoring mask: added rows only, whose
starting argmax agrees in native expanded GPU, CPU, NumPy and single-row GPU
evaluation. No labels determine this stability mask. Training still consumes
ALL expanded-FIT rows. Report full regret and this stable-added regret separately
so repairing old row66 cannot masquerade as new decision learning.

The MSE control is rerun because admission now checks additional batch
geometries. Compare its native256 metrics to331, recording any difference;
never silently assume equivalence.

## Automatic Chain

1. Run all12 arms for256 attempted updates, logging accepted/rejected steps,
   objective components, clipping, guards, gradients and fixed scoring supports.
2. A cost/hybrid arm qualifies for1024 only if all four endpoints pass all
   saved-checkpoint guards, each accepts>=32 steps, none worsens stable-added
   regret by>1e-8 versus its start, and at least3/4 improve by>1e-8 versus BOTH
   their start and the matched MSE control. No choice based only on loss.
3. Extend all qualifying objectives AND the matched MSE control to1024, otherwise
   write the rejected gate. No intermediate user approval.
4. Automatically cold-replay all endpoints, reproduce three transitions after
   step64 for each arm, generate JSON/table/plots and seal the result. If no
   objective passes, emit a measured negative contrast, not a queue waiting for
   the user's next instruction. Source actor is never promoted here.

## Test Design

| Requirement | Evidence |
| --- | --- |
| Correct cost gradient | Finite differences; moving probability to a lower-cost action reduces loss |
| No hidden scale/adapter bug | Deterministic TRAIN-only tau, shift invariance, exact hybrid decomposition |
| Existing MSE unchanged | Default-versus-explicit MSE AdamW parity |
| Transaction remains safe | Custom-loss rollback and optimizer resume parity |
| No artificial learning claim | Stable mask excludes original rows and mismatched baseline decisions |
| Extension respects decisions | Deterministic gate rejects loss-only, degradation or incomplete cases |
| Reproducibility | Per-unit identity, atomic checkpoints, cold/restart replay, source hashes |

Numerical guards verify finite states; a smooth surrogate is not assumed to
solve discrete decision quality until the measured contrast says so.

Prelaunch:50 focused CPU tests pass in3.75s, including exact MSE AdamW parity,
new loss finite differences, stable scoring and resume/rollback. Scoped
repository correctness-rule Ruff checks and CLI help pass.

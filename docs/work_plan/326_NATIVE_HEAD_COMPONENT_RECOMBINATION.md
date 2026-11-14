# Native Critic Head Component Recombination

Owner authorizes the next bounded diagnostic after325. No training, actor
update, new observations, source mutation or promotion. Source299,71 inputs,
49 cumulative phase2 return candidates and paused STL remain unchanged.

## Frozen Design

Within each foldA/B and seed19/43, compare the322 control head to both324
endpoints (8192 and32768 updates). The same initialization underlies each
paired fit. Swap matching components in both twins together, never across
seeds. The hidden component is Linear208->8 plus SiLU inside the critic head;
the output component is Linear8->3. Include weights AND biases. Frozen temporal
encoders, feature support, label function and twin-min aggregation do not change.

Four combinations per context:

| Hidden component | Output component | Role |
| --- | --- | --- |
| Control | Control | Authenticated original control |
| Expanded | Expanded | Authenticated complete324 endpoint |
| Control | Expanded | New hybrid, internal representation preserved |
| Expanded | Control | New hybrid, previous output map preserved |

Eight paired contexts and two hybrids each produce16 new diagnostic heads.
Reconstruct both full heads as self-combinations and require exact tensor and
output parity. Do not infer that algebraic layer attribution uniquely assigns
causal credit: hidden/output components may have coadapted during each fit.
Measure the per-twin interaction identity
Q_NN-Q_NC-Q_CN+Q_CC=(h_N-h_C)(W_N-W_C)^T. Do not apply this linear identity
after the nonlinear twin-min operation.

## FIT Gate And Interpretation

Primary endpoint8192; secondary32768 is reported separately. A hybrid passes
the FIT diagnostic only if original-FIT regret is no worse than the control
(tolerance1e-8) AND expanded-FIT regret improves by more than1e-8. A consistent
result requires all four fold/seed pairs for the same hybrid and endpoint.
No decision threshold is fitted. Stop/audit and native-state strata are
descriptive and cannot choose a hybrid for deployment. Even a consistent FIT
result is not actor qualification or proof of trading improvement.

If recombination fails, do not conclude that freezing a component during
training cannot work: an unadapted output map can be incompatible with its
new hidden representation. Use the evidence to define, at most, a matched
warm-start output-only versus full-head fit from the SAME control checkpoint,
same expanded FIT and fixed budgets, with the unchanged control retained.
That later contrast is not part of this no-training diagnostic.

## Tests And Evidence

| Requirement | Evidence |
| --- | --- |
| Exact copying, including bias | Schema/dimension tests, per-key donor provenance |
| No alias or donor mutation | Clone independence tests, donor digests before/after |
| No hidden architecture change | Reject extra/missing keys, wrong shape/dtype/device, nonfinite tensors |
| Same original function | Self-combination state and output parity |
| Same native evaluations | Replay12 stored endpoints with original batch geometry |
| Real interaction accounting | Analytic bilinear identity test and native residual |
| No audit-driven gate | FIT-only gate tests, including tolerance and empty/invalid metrics |

Save each hybrid state separately, all metrics and FIT decisions, raw outputs,
interaction residuals and a comparison table. Preserve all failures. No new
2024 or TEST inference and no full-account return claim.

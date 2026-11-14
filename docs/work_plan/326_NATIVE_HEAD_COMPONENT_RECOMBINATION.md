# Native Critic Head Component Recombination

Status: COMPLETE, no hybrid accepted or model promoted. Runtime commit
`4dd29868`; execution identity
`954530afa82c383de8006736468767135858dcb3626e5427c845e8b2caa33012`.

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

## Completed Results

All16 hybrids ran in1.41 seconds on the primary GPU using cached embeddings.
This is inference over small saved heads, NOT another full training run.
17 focused tests passed. Independent verification checked17 seals,128 donor
tensors including biases and128 metric scopes. A separate float64 NumPy CPU
forward reproduced every greedy decision, with maximum value difference
2.188132e-6 from GPU float32.12 original endpoint replays were GPU-exact.
The bilinear interaction identity had maximum residual1.353670e-6.

Mean regrets across four paired fold/seed contexts (lower is better):

| Components | Donor updates | Original FIT | Expanded FIT | Audit, descriptive | FIT passes |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original control | 8192 | 0.008043 | 0.232719 | 0.236451 | reference |
| Complete expanded head | 8192 | 0.099032 | 0.123486 | 0.249055 | reference |
| Control hidden / expanded output | 8192 | 0.013302 | 0.238665 | 0.229387 | 0/4 |
| Expanded hidden / control output | 8192 | 0.107318 | 0.144359 | 0.246023 | 0/4 |
| Control hidden / expanded output | 32768 | 0.026084 | 0.242835 | 0.239268 | 0/4 |
| Expanded hidden / control output | 32768 | 0.118331 | 0.126234 | 0.242797 | 0/4 |

Preserving the original hidden component recovers much of original-FIT
behavior relative to the complete expanded head, but the unadapted mixed
head does not jointly preserve original decisions and improve expanded FIT.
None of16 hybrids passes. The lower audit regret of one aggregate is not a
promotion criterion and does not override the predeclared FIT gate.

This rejects simple checkpoint component substitution as the tested repair.
It does not reject fitting a new output map against a frozen hidden layer:
the donated output map was fitted against a different hidden representation.
The next bounded contrast is warm-start output-only (54 parameters) versus
full native heads (3398 parameters), starting from the SAME authenticated
control, same expanded-FIT targets and paired sampling, with original-FIT
preservation measured throughout. Keep control and fixed8192/32768 endpoints;
no new feature removal or actor update before interpreting that contrast.
That training is not launched by326.

Source299 and71 inputs are byte-preserved. No new trading return, live
qualification or validation2024 result is claimed. Evidence:
[full paired table](generalization/head_recombination_20261113/RESULTS.md),
[complete metrics](generalization/head_recombination_20261113/RESULT.json),
[independent verification](generalization/head_recombination_20261113/INDEPENDENT_VERIFY.json).

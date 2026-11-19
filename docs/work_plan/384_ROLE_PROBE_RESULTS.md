# 384: window recognition is not an economic entry command

[Design](384_ROLE_PROBE.md). Four sequential GPU fits finished at patience 80;
initial LR 0.0003. The market-only latent has 208 coordinates from the original
actor head input. Its cold logits were checked before fitting new heads.
No live orders, no 2024/TEST scoring. The optimization-era ZigZag implementation
was recovered from its historical Git blob and checked against the original
SHA256; newer implementations were not silently substituted.

| Role | Fit / internal 2023 ticks | Epochs / best | Calibrated Brier | Fit prevalence / hour-only Brier | MAE bp / fit median baseline |
|---|---:|---:|---:|---:|---:|
| Open buy |8713 / 3759|109 / 29|0.2368|0.2419 / 0.2407|36.89 / 37.48|
| Open sell |8713 / 3759|87 / 7|0.2341|0.2398 / 0.2394|38.77 / 38.93|
| Close buy |103 / 113|81 / 1|0.2499|0.2637 / 0.2768|27.95 / 28.91|
| Close sell |802 / 818|104 / 24|0.2475|0.2503 / 0.2588|28.30 / 28.27|

Lower is better. Binary probabilities refer to different events for open vs
close: pivot window membership vs positive close advantage. Do not rank
all roles by raw probabilities. Regression is the paired economic advantage
in bp of current equity, not an FX price forecast. Unavailable or incomplete
pivot support is censored. Near episode end both forks share the same
truncated horizon. The native source closes or holds thereafter in both.
One seed, fixed inherited features/weights and pivot parameters; this does
not establish reproducible convergence or fully historical generalization.

In fit, 1879/5154 buys outside the valley window beat waiting, while 1315/3559
inside did not. For sells, 2007/5201 outside the peak window beat waiting
and 1255/3512 inside did not. Proximity and financial benefit are not synonyms.

Cold posterior CPU replay agreed with stored GPU Brier to within 1e-5 for all
roles. Low vs high epistemic SD quartile Brier on 2023: open buy 0.2487 vs
0.2174; open sell 0.2471 vs 0.2074; close buy 0.2500 vs 0.2498; close sell
0.2491 vs 0.2453. Descriptive strata, not a calibration test of realized
economic tail risk. The signal does not support a high-SD abstention rule.
Do not infer that high SD is itself desirable either. Close-buy predictions
barely beat the uninformative 0.5 forecast (Brier 0.25); no maturation claim.

Decision: keep all four checkpoints as diagnostics, retain existing policy.
Next bounded comparison: same weights, same native accounts, advantage mean
alone vs an uncertainty-aware decision rule set on development only; do not
use pivot probabilities as economic confidence. Before that comparison,
check uncertainty for the *economic advantage* against actual advantage
errors on development and address thin close-buy exposure; if calibration
fails, do not deploy an uncertainty penalty. No DEAP or floating restart.
Four checkpoint/normalization filesets, epoch traces, curves and calibration
bins saved in the private full run. DATA manifest sha256
7f98b7627e06b0a3204c3518797f2a57a51820253eaf46a22dc325b7a65ba4b9.
Completion receipt sha256
3f3e63487ec699da7af6b146506cf82af5b6563e99c9f804a36cd544003d454e.

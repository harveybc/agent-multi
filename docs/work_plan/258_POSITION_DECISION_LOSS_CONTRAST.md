# Position Decision-Loss Contrast

Owner authorized continuation of257. Sixteen diagnostic head fits; original
actor and encoders remain frozen. No new market inputs, account feedback,
feature selection, actor transfer or broker activity. Reuse every2342 position
record,866 banks, fixed fit2020-2022 and monitor2023, same native targets and
fixed TRAIN-derived axes. Source pretraining already used2023. No2024/TEST.

## Frozen Comparison

Interfaces progress9/surface81, seeds19/43, market embeddings actual/fit-only
bank-balanced constant. Two new objectives per combination. Existing257
SmoothL1 endpoints are authenticated and reused as controls, not retrained.
Same initial tensors, primary minibatches,4096 updates,batch128,AdamW3e-5,
WD0.01,clip-gradient norm10. One GPU sequential queue and automatic collection.

For native account gap y=Q(CLOSE)-Q(HOLD), s=sign(y), output z:

`L_sign = mean(abs(y) * softplus(-s*z)) / c_sign`.

Targets abs(y)<=1e-8 contribute zero. c_sign is mean abs(y) over fit banks,
with each bank and each state within bank weighted equally. It is frozen
before optimization; no per-batch or monitor normalization.

The paired arm draws a second state j uniformly from the SAME fit bank for
each scheduled i, allowing self-pairs. Only opposite nonzero signs contribute:

`L_pair = mean(abs(y_i-y_j) * softplus(-sign(y_i-y_j)*(z_i-z_j)) * opposite) / c_pair`.

c_pair is the exact fit-bank expectation of abs(y_i-y_j)*opposite for independent
uniform i,j. The two objectives are SIGN=L_sign and
SIGN_PAIR=0.5*L_sign+0.5*L_pair. Coefficient0.5 is a declared equal-weight
design choice, not optimized. Same-sign and neutral pairs contribute zero;
an absent informative pair population fails explicitly. One common partner
schedule per seed is saved and reused across all paired arms.

These outputs are decision scores, NOT calibrated Q gaps or probabilities.
Positive mean-twin score requests close, <=1e-8 holds. Raw score-versus-gap
RMSE is dimensionally inappropriate and is omitted for these new arms.
Retain native decision regret, sign accuracy, counts, harmful original-position
closes and missed account-beneficial closes. No claim of profit from replay
instances. Changing loss scales may interact with the fixed optimizer; this
comparison tests these declared objectives, not every possible learning rate.

## Acceptance Tests Before Execution

| Requirement | Test/evidence |
|---|---|
|Correct sign pressure and cost weights|Analytic zero-score gradients; positive/negative/neutral targets|
|No hidden mean/clip|Bank-balanced fit constants; monitor perturbation invariance|
|Comparable schedules|Exact primary schedule/initial tensor hashes against257|
|Position pairing is causal-role compatible|Same fit bank, same market and direction, no monitor pairs|
|Correct paired gradient|Opposed pair gradient separates scores; same-sign pairs zero|
|Numerical safety|Extreme scores finite; sign inversion invariance|
|No model corruption|Source tensor/hash and all-TRAIN actor parity, exact head reload|
|Meaningful diagnostics|All monitor cases including extrapolation, critical opposed-choice witness|
|Durable execution|Lock, per-arm sealing/restart, FAILED on error, automatic collection|

Report all fixed endpoints and curves at0/64/256/1024/2048/4096. No best-step
selection after inspecting monitor. Primary comparisons: same-interface
SmoothL1 versus SIGN versus SIGN_PAIR, each against its constant-market control.
Learned critical witness and harmful closes remain necessary checks before
actor transfer; a lower aggregate regret alone is insufficient. No automatic
promotion or architecture change follows from this diagnostic.

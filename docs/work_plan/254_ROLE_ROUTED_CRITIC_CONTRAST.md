# Role-Routed Critic Contrast

## Scope And Frozen Decision

Owner approved continuation after253. This is a critic diagnostic, not a
replacement actor, live strategy, or feature-selection evaluator. Preserve all
source checkpoints. No account-state features enter an encoder or head. The
network emits all outputs from market inputs; a separate controller selects
the flat, long or short value output using the native position direction.

Before execution freeze this design. Reuse every comparable bank from252,
including its exact labels, timestamps and support. Fit2020-2022, monitor2023;
2023 was used by source pretraining, so this is not an untouched holdout.
No2024/TEST evaluation. Same70 inputs,13 observations,stride2 and fixed scaling.

## Contrasts

Use the initial calibrated critic's two frozen encoders. Cache the exact
pre-head embeddings once and verify reconstruction against native twin Q.
Fresh heads retain Linear(input,8), SiLU, Linear(8,3), independently seeded.
Six sequential arms: seeds19/43 crossed with:

1. Shared single head per twin.
2. Three independently initialized heads per twin, averaging their outputs
   without position information: parameter-count-matched control.
3. Exactly the same three-head initialization, externally routed by role.

All arms use4096 updates,batch128,fresh AdamW and the calibrated model's LR/WD,
the existing pairwise-plus-mean Huber objective,gradient norm cap10. No encoder,
actor,target-critic or entropy update. All outputs keep NOP/BUY/SELL semantics;
the native controller already maps the opposite signal to closing. No new
forced timing, minimum trade count, risk rule or invalid-action mask.

Sample banks uniformly, then surviving states uniformly, on one identical
schedule per seed. Report full deterministic metrics at0/64/256/1024/2048/4096,
both twins and minimum, by fit/monitor and flat/long/short. Main comparison:
endpoint bank-balanced regret, plus action-gap RMSE and ranking accuracy. Do
not select a favorable checkpoint. Report both seeds, not only a winner.
The equal-state constructed distribution is not natural trading frequency.

## Same-Direction Residual Audit

Existing banks only provide nine repeated-row same-direction groups. Instead
of interpreting sparse coverage as absence of conflict, reconstruct all512
original flat origins. For each direction, attempt native opening at offsets
0,1,3 decision steps, then hold. At ages4/12 compare surviving positions at
identical market rows. Do not manufacture fills, force positions to survive,
replace rejected entries, or vary risk coefficients. Deduplicate identical
native states, record exclusions and actual elapsed hours. Simulate each first
action then the same frozen greedy continuation to the quarter terminal.
This measures residual conflict without training on the new panel.

## Requirement-To-Test Trace

| ID | Requirement | Verification |
|---|---|---|
|R1|No state input|All-head forward accepts embeddings only; controller gather separate|
|R2|Correct routing|All roles, invalid role refusal, permutation-equivariance tests|
|R3|Matched arms|Same state dict for routed/mean; same sample digest and budget|
|R4|No collateral update|Source state/hash and full-TRAIN actor parity before/after|
|R5|Frozen encoder fidelity|Cached heads reproduce native Q within1e-6; finite values|
|R6|Real histories|Native opens only, time/input equality, parent-isolation tests|
|R7|No false compatibility|Deduplicate state; keep survival and unsupported counts|
|R8|Temporal isolation|Authenticated quarterly support and fit-label terminal checks|
|R9|Reproducibility|Persist schedule, initialization/final heads, effective config, curves|
|R10|Recovery|Exclusive lock, sealed arm caches, explicit failure, automatic collector|

No profitability claim follows from a successful fit. Role routing may still
leave incompatible values for same-direction positions with different entry,
barriers, age, balance or size. Report this limitation rather than silently
introducing those variables into observations. Source continuation is fixed;
end-to-end actor transfer requires a subsequent explicit design.

## Pre-Training Execution Correction

Runtime88388fab stopped before any head update at native-state equality. The
state view records entries_day as tuples; JSON receipts deserialize those as
lists. Compare canonical serialized content, preserving every field, rather
than Python container-type equality. Regression accepts this roundtrip but
rejects changed entry counts and cash. The failed collection remains intact;
the corrected runtime uses a new output identity and root, not reused outputs.

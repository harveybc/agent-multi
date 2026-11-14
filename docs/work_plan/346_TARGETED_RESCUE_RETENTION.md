# 346: Retention Through 64 Matched Attempts

Status: IMPLEMENTED_READY_FOR_BOUNDED_EXECUTION. Focused suite:81 passed;
scoped Ruff passed. Producer automatically invokes independent CPU verification.

Resume each of the four sealed345 ENDPOINT_16 states, including its actual
AdamW moments, RNG, accepted-step count, curve, trace and exhausted search
counter. Execute ONLY attempts17..64 on the same covering schedule as342/345.
Original334 sources remain the reference; saved342 ENDPOINT_64 is the control.
Do not refit either control or the first16 steps. Preserve71 inputs,28 groups,
source actor299 and all49 cumulative return candidates. No2024/TEST reads.

The continuation code must have no targeted-search dispatch. It uses only the
ordinary341 finite-protected update; search counter must stay exactly4 in
every added trace row. Keep learning rate, weight decay, loss, original guards,
optimal sets, projection radius, relinearizations and finite scales unchanged.
The sole question is whether existing gains persist during ordinary updates.

## Test Design

|Requirement|Acceptance evidence|
|---|---|
|Valid full-state resume, no budget reset|Unit attacks on identity, source hash, offset, search count, accepted count and optimizer steps|
|No additional searches|Unit suffix checks and verifier rejects targeted arms or target_search records after16|
|Original optimizer/RNG preserved|Loaded optimizer/RNG digest parity before first new update|
|Same batch sequence|Schedule digest matches345; original334 and control342 parent identities match|
|No retraining the prefix|Copy inherited8/16 endpoints under new identity, exact state/optimizer/RNG/trace/curve parity|
|Cold and restarted continuation|Check all8/16/32/48/64 points; repeat17/18 and33/34 exactly|
|Independent result verification|CPU scores, trace chain, optimizer counts, no-search suffix and gains retained from16|
|Negative control|When resumed weights/optimizer/RNG equal342 at16, require final64 state/optimizer/RNG exact|
|No hidden promotion|Source bundle hash unchanged; no actor update, no floating pass|

Report FIT regret versus matched64 control and versus16; individual previously
improved decisions retained/lost; soft cost separately; descriptive internal
TRAIN stop/audit separately. Those internal partitions are not end-to-end
held-out validation because encoders used TRAIN. Do not use held scores to
choose updates or a checkpoint. Stop at64; no automatic further extension.
Producer seals results and invokes the independent verifier automatically.

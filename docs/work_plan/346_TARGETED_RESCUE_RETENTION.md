# 346: Retention Through 64 Matched Attempts

Status: COMPLETE_INDEPENDENTLY_VERIFIED. Focused suite:81 passed;
scoped Ruff passed. Automatic CPU verification and separate local replay passed.

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

## Results

Runtime/verifier commit:81dba6df. Identity
`5b28e01442fa513619f1a1ae18429e84ecb736334d846b45a21bdd034ab86fbe`.
The four cases finished in201.26seconds with zero restarts. Only192 new
attempts were executed (17..64 per case),167 accepted. Including the inherited
prefix, cumulative accepted steps are227/256. No new targeted searches.

|Case|Control64 FIT regret|Retained64 FIT regret|Comparison|Initial gains retained|New accepted /48|
|---|---:|---:|---|---:|---:|
|A /19|0.153583178|0.145327107|Better|3/3|41|
|A /43|0.141511388|0.139623538|Better|1/1|43|
|B /19|0.192433535|0.192573954|Worse|1/1|35|
|B /43|0.229738595|0.229738595|Exact tie|0/0|48|

All FIVE earlier improved FIT decisions survive. At64 there are7 improved
decisions versus the original sources and0 worsened, compared with3 improved
decisions in the controls. These are case-row diagnostic counts, not trades or
independent observations. Two additional improvements appear after16: A19 and
B43; the latter also appears in its ordinary control. Mean FIT regret is
0.176815798 versus0.179316674, **1.39467% lower**. There are2 matched wins,
1 loss and1 tie, not the3 wins/1 tie observed at16. B19 preserves its earlier
gain but misses the slightly better improvement obtained by its control.

The soft cost remains higher than control in A19,A43,B19, identical in B43.
Retention of useful actions and minimizing surrogate loss are different tests.
The optimal-set guard explicitly protects newly optimal FIT decisions; the
retention result verifies this implementation, not independent generalization.

Internal TRAIN audit regret remains identical to control in all four cases.
Internal stop regret is identical in three cases, but A43 is worse:
0.285994797 versus control0.278837543. This is a missed control improvement,
not deterioration from A43's initial0.285994797. B43 retains the same prior
stop deterioration as its control (source0.194676539, final0.205758166).
These are descriptive internal partitions: the encoder used TRAIN, so they
are not end-to-end held-out data. Neither2024 nor TEST was read.

## Verification

- All20 endpoints cold-reproduce (8 inherited,12 newly written).
- All16 repeated new transitions reproduce exactly, including optimizer/RNG.
- Independent CPU verifier checks256 trace entries including the64 inherited,
  optimizer counts, source lineage, unchanged prefix and exhausted-search suffix.
- NumPy forward versus Torch64 maximum error:7.9936e-15.
- Separate local CPU verification reproduces VERIFICATION.json byte-exactly;
  the updated verifier also reproduces345's prior report byte-exactly.
- B43 starts equivalent to control16 and ends EXACTLY equal to control64 in
  weights, optimizer and RNG. The other cases have different starting states;
  their negative_control_exact=false is not a failed equality test.
- Parent seals and actor299 bundle hash remain valid.71 inputs,28 groups and
  all49 cumulative return candidates are unchanged; no model was promoted.

## Next Bounded Step

Do not repeat floating selection with an unchanged actor or launch a longer
fit merely because FIT improved. First replay the saved B19 FIT and A43 stop
divergences against342 to locate the first differing decisions and attribute
them to the changed direction, AdamW history or protection. Use saved states
and identical batches; no held labels in update selection, no guard relaxation,
no new target-search budget. This diagnostic precedes any actor refinement.
The64-step experiment is closed; no further GPU continuation is running.

[Table](generalization/targeted_retention_20261114/SUMMARY.csv),
[raw results](generalization/targeted_retention_20261114/RESULT.json),
[verification](generalization/targeted_retention_20261114/VERIFICATION.json),
[identity](generalization/targeted_retention_20261114/CONFIG.json),
[plot](generalization/targeted_retention_20261114/FIT_REGRET.png).

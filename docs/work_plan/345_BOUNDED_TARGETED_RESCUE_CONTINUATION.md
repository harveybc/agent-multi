# 345: Bounded Targeted Rescue Continuation

Status: IMPLEMENTED_TESTED_BEFORE_EXECUTION.73 focused tests and scoped Ruff
pass.344's independently verified corrections
occur in five witnesses covering three of four cases. Test accumulation before
considering any actor change or longer fit.

Use the four334 sources and the identical first16 batches, LR, AdamW state and
guards used in342. Compare against its saved ENDPOINT_16 controls; no repeated
control fitting. At each step first compute the ordinary341 correction. If it
already strictly improves FIT regret in all five views, use it. Otherwise allow
at most FOUR targeted searches per case (the first four eligible attempts),
each with344's fixed8-candidate/radius4/margin/relinearization rules. Select a
strictly cost-and-regret improving targeted candidate by NumPy FIT regret,
cost,row; otherwise use the ordinary correction or its exact rollback.

Optimizer semantics are explicit: this is projected AdamW. Every admitted
update keeps that step's PROPOSED AdamW moments and increments its counter,
whether the direction is ordinary or targeted. A rejected update restores
weights AND moments. No344 candidate checkpoint supplies optimizer state.
No extra gradient step is taken while examining alternatives. There are16
optimizer attempts per case, but treatment has extra projection compute;
this is not an equal-wall-clock comparison.

Dynamic optimal sets are recomputed each step; newly corrected decisions gain
protection. Original anchors stay frozen. No held-label update selection.
Save8/16 endpoints, cold-check both and replay9/10 from8 exactly with the saved
search counter. Also replay1/2 from the original source, including any targeted
searches. Verify optimizer counts, finite guards and trace chaining.
Require344 verification before launch and preserve all parent seals. Stop
after16 attempts in each case; no budget extension or actor promotion.

Tests: inherited finite guard/derivative/transaction tests; target-choice
preference and fallback fixtures; real cold replay with saved search budget;
independent CPU endpoint scoring and per-step trace verification. Report
matched FIT decisions, descriptive internal TRAIN stop/audit, accepted steps,
targeted searches/accepted rescues and source hashes. No2024/TEST reads.
The producer invokes the independent CPU verifier automatically after sealing;
the supervisor resumes whole incomplete cases with bounded restarts.

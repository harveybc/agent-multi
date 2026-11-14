# Native Critic Head Warm-Start Contrast

Owner authorized execution on2026-11-14 after326. Stage: TESTS_PASS_PREPARED.
Eight sequential fits: foldsA/B, seeds19/43, output-only/full-head.
15 focused tests pass, including exact CPU interrupted/uninterrupted updates,
parameter scopes, immutable frozen tensors and rejection of invalid updates.

## Frozen Protocol

Start BOTH arms at the exact matching322 trained control, NOT at random
initialization or a326 hybrid. Reuse324 authenticated533 native records and
cached frozen embeddings. Expanded FIT has196/392 rows; original FIT49/98.
Original stop/audit masks are unchanged. Source299 actor,71 inputs, labels,
scaling, temporal encoders and strategy remain unchanged. No2024/TEST reads.

Reset AdamW state in BOTH arms: this tests weight warm start, not optimizer
momentum transfer. Same LR3e-5, weight decay0.01, batch128, MSE, gradient norm
limit10 and paired deterministic covering schedules. Train only the final
Linear8->3 in each twin (54 parameters), or both Linear208->8->3 heads
(3398 parameters). No random layer reinitialization. Check frozen tensors
bit-exact at every saved observation. Original source files are immutable.

Primary endpoint8192 additional updates; secondary32768 additional updates.
Fixed budget, no early stopping or audit-based selection. Save measurements
at0,64,256,1024 and every2048 updates, plus optimizer/RNG/checkpoint state
for exact restart. The checkpoint's original8192 updates are additional
history, shared by both arms, not part of these new budgets.

Gate: relative to matching control, original-FIT action regret must not worsen
beyond1e-8 AND expanded-FIT regret must improve by more than1e-8. Consistency
requires4/4 pairs at a fixed endpoint and scope. Publish stop/audit metrics
descriptively, including RMSE and regret, not as a checkpoint selector.
No automatic actor, feature-list or live-model promotion even if gates pass.

## Requirement To Test Map

| Requirement | Pre-execution test | Runtime evidence |
| --- | --- | --- |
| Same initial learned weights | State equality, schema rejection | Initialization hash and exact control metrics |
| Only intended layers train | Scope/count, nonzero changes, frozen exact | Trainable names and per-key deltas |
| Optimizer cannot update frozen tensors | Exact parameter identity check | AdamW scope and frozen checkpoint digest |
| Paired exposure | Shared schedule, deterministic seed | Schedule hashes and expanded-support hashes |
| Restart is not fresh optimization | CPU interrupted/uninterrupted parity | Complete optimizer/RNG snapshots and cold reload |
| FIT controls decisions | Existing326 gate regression | Gate from FIT only at both endpoints |
| Native data and control preserved | Parent identities verified | Seals and source hash before/after |

If output-only preserves old decisions but cannot learn the expanded labels,
do not promote it for stability alone. If full-head wins only on approximation
but loses action ordering, report that distinction. A failure is not evidence
that no market signal exists. Do not restart floating selection unchanged.

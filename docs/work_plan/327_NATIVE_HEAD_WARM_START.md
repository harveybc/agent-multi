# Native Critic Head Warm-Start Contrast

Owner authorized execution on2026-11-14 after326. Stage: COMPLETE_NO_PROMOTION.
Eight sequential fits: foldsA/B, seeds19/43, output-only/full-head.
15 focused tests pass, including exact CPU interrupted/uninterrupted updates,
parameter scopes, immutable frozen tensors and rejection of invalid updates.

## Frozen Protocol

Start BOTH arms at the exact matching322 trained control, NOT at random
initialization or a326 hybrid. Reuse324 authenticated533 native records and
cached frozen embeddings. Expanded FIT has196/392 rows; original FIT49/98.
Original stop/audit masks are unchanged. Source299 actor,71 inputs, labels,
scaling, temporal encoders and strategy remain unchanged. No2024/TEST reads.
These are internal TRAIN diagnostics: the frozen encoders were originally
fitted on the full TRAIN period, so stop/audit are not independent end-to-end
validation. Native targets are the same annual-bounded discounted continuations
under the fixed source actor as324, not targets recomputed for an updated actor.

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

## Execution And Independent Verification

Runtime commit `9585e596`, identity
`435f400b6dd83a85120a6252eb6b8a0a4fb6a86477d2a9bf7f163371168a1bdd`.
All8 fits and16 endpoint reloads completed in239.65 seconds on the primary
GPU. Supervisor exited0, no restarts, no clipped updates. Both arms used
identical control weights, schedules and support within each fold/seed pair.
The CPU verification checked9 seals,16 endpoints,128 metric scopes and32
frozen endpoint tensors. Independent float64 NumPy inference reproduced
action regret exactly; maximum RMSE discrepancy4.163674e-7.15 tests pass.

## Results

Equal-weight means over four fold/seed pairs; action regret is lower-is-better,
in native discounted-reward units, not profit percentages or trade counts.

| Variant | Additional updates | Original FIT regret | Expanded FIT regret | Audit regret | Paired FIT passes |
| --- | ---: | ---: | ---: | ---: | ---: |
| Unchanged control | 0 | 0.008043 | 0.232719 | 0.236451 | reference |
| Output only | 8192 | 0.017945 | 0.231894 | 0.217530 | 0/4 |
| Full head | 8192 | 0.025809 | 0.116318 | 0.227021 | 0/4 |
| Output only | 32768 | 0.078385 | 0.233232 | 0.228335 | 0/4 |
| Full head | 32768 | 0.027790 | 0.037762 | 0.261350 | 2/4 |

The full-head late passes are both seeds in foldA: original-FIT regret
0.000227/0.000089 and expanded-FIT0.000445/0.000657. FoldB still fails both
seeds: original-FIT0.069755/0.041088, expanded-FIT0.070144/0.079800.
Thus the frozen-hidden constraint is not a general repair, and full warm
start can preserve and extend the fitted decisions in some settings, but
does not meet the required4/4 consistency. No head or actor is promoted.

The descriptive audit RMSE for output-only improves from control1.099600
to0.822842 then0.723679, even while original-FIT action regret worsens.
Full-head audit RMSE is0.982990 at8192 and1.442478 at32768. Lower numeric
value error alone still does not ensure preservation or generalization.
Do not promote the output-only arm for its lower audit regret after seeing it.

Intermediate FIT gate passes exist and are preserved in the curves. For
example B19 full-head passes at64,256,2048,4096 but fails both fixed endpoints;
B43 full-head never passes a saved observation. This is not evidence that
only an early transient can work: A43 full-head recovers late at24576 and
retains the gate thereafter, while A19 passes again at32768. No post-hoc
checkpoint or stopping rule is selected from those observations.

## Next Bounded Step

Measure original-FIT versus added-FIT gradient alignment and value-level
versus action-gap contributions at the saved initial/early/late checkpoints,
especially foldB. Use FIT only to diagnose whether updates trade old decisions
for new ones before choosing a preservation/ranking-loss or replay-weight
contrast. This next diagnostic is not launched here. Do not infer insufficient
capacity, absence of signal or a need to discard features from these results.
Source299,71 inputs and49 cumulative phase2 return candidates remain unchanged;
STL stays paused. No2024/TEST inference or new trading-profit claim.

## Evidence

- [All16 endpoint results](generalization/native_head_warm_start_20261114/RESULTS.md)
- [Metrics and state identities](generalization/native_head_warm_start_20261114/RESULT.json)
- [Independent verification](generalization/native_head_warm_start_20261114/INDEPENDENT_VERIFY.json)
- [FoldA seed19 curves](generalization/native_head_warm_start_20261114/A_19_curves.png)
- [FoldA seed43 curves](generalization/native_head_warm_start_20261114/A_43_curves.png)
- [FoldB seed19 curves](generalization/native_head_warm_start_20261114/B_19_curves.png)
- [FoldB seed43 curves](generalization/native_head_warm_start_20261114/B_43_curves.png)

Per-observation metrics and weight deltas are retained in the adjacent
`curves/` directory; resumable weights and optimizer states remain sealed in
the run artifact store, not copied into the source repository.

# Selected72 Critic Calibration

2026-11-11. Continue phase2 from qualified export295, 72 inputs and28 groups.
Keep its actor and all historical controls. This is not a live promotion.

| Requirement | Verification |
|---|---|
| New policy, new targets | Rebuild inherited151 TRAIN anchor continuations with the72-input actor; never reuse69-input returns |
| Critic-only training | Empty actor-arm list, zero actor updates,4096 head updates for each seed19/43 |
| TRAIN only | FIT2020-2022;2023 MONITOR diagnostics, no fit on2024 |
| Preserved policy | Frozen actor/buffers during fit; full valid-window logits identical after calibration and cold reload |
| Native data unchanged | Same full support hash, recipe, fixedTRAIN scaling and original trading contract |
| Reproducible completion | Immutable runtime, finite sequential queue, hashed source and output receipts, no promotion |

Reference: TRAIN15.612725500002252%, VALIDATION18.369945750000117%,
746/207 orders. Critic calibration cannot improve these account results because
the actor stays fixed. Report FIT and MONITOR value RMSE and ranking separately.
Do not interpret lower FIT error as generalization.

After calibration, diagnose executable TRAIN boundaries before any bounded
interior corrective update. Preserve the step294 nonzero feasible projection
and decision-preservation constraints; do not repeat the rejected raw AdamW
actor refinement. A later fresh-context floating pass must consider all48
historical phase2 absences, not only the last pass.

STL recovery and current-hour producer availability are separate workstreams.
Neither a completed historical repair nor this calibration constitutes a
complete live input feed. See the updated handoff to Retsu.

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

## Completed

Runtime45739af3 completed both seeds,4096 critic-head updates each, in198.01s
(finite queue203.03s,exit0). Rebuilt151 native current-policy anchors:113 FIT,
38 MONITOR. All20 sealed output files authenticated. Actor logits over the
entire valid support are unchanged after calibration and cold reload; no actor
update and no account improvement is claimed.12 focused tests and scoped Ruff
passed. Outputs and calibration curves are preserved on both worker and coordinator.

| Seed | FIT RMSE before/after | MONITOR RMSE before/after | FIT pair accuracy before/after | MONITOR pair accuracy before/after |
|---|---|---|---|---|
|19|0.750018 /0.335011|0.672060 /0.728980|52.20% /76.27%|58% /62%|
|43|0.824966 /0.517677|0.685604 /0.702810|52.54% /67.46%|40% /49%|

Monitor regret improves0.218868 ->0.155363 and0.342066 ->0.267099, despite
worsening value RMSE. This qualifies a measured calibration, not a complete
repair of generalization. TRAIN/VALIDATION remain15.612726% /18.369946%,
746/207 trades, exactly the295 research reference.

Identity: `71dfa35d9c1118b3e461e338bb5a97142c3001edd612741fd2e7d9c2dcd282fa`.
Private run: `sac72-critic-calibration-20261111-r1`.
Evidence: [summary](generalization/selected72_calibration_20261111/SUMMARY.json),
[curves](generalization/selected72_calibration_20261111/CURVES.png).

The separate live-input review also reproduced order dependence in the COT
producer, agreeing with Retsu's reported contract mixture. Quantify historical
impact before interpreting a corrected store as this model's original input.
See [handoff](../handoffs/MUSASHI_TO_RETSU_SELECTED72_LIVE_GAPS_2026_11_11.md).
Keep research continuity but no promotion or silent input replacement.

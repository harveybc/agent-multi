# Selected72 TRAIN Boundary Diagnosis

2026-11-12. Owner requests the next step after296 calibration. Read-only
diagnosis, no actor optimizer step or new deployable candidate.

| Requirement | Test / gate |
|---|---|
| Correct current source | Authenticate296 receipt,72-input export,full support and original parent |
| No validation targeting | Only151 inherited TRAIN anchors,113 FIT and38 MONITOR; no posthoc2024 events |
| Executable decisions | Report flat/allowed subset separately from all anchors; never call probes trades |
| Tie correctness | Probability mass over ALL native optimal actions; all-action ties cannot be counted as harmed by redistribution |
| Same contrast | Seeds19/43,first batch128 from fixed8-step covering schedule; Q,min twin, native FIT return and no-entropy controls |
| Shared-weight diagnosis | Analytic logit gradients versus autograd; independent/local,self,cross and first-step AdamW projections |
| No policy modification | Actor parameter equality before/after each arm, source hashes and receipts unchanged; no optimizer step |

A native FIT control replaces only FIT targets, not MONITOR. MONITOR returns
are used only for diagnostic scoring. Jacobian projections are local effects,
not a claim that a finite optimizer step is accepted or improves trading.
First-step AdamW decomposition assumes fresh moments, matching the proposed
refinement restart, and does not authorize returning to unconstrained AdamW.

Execution is CPU-only with capped threads, finite timeout and exclusive output
lock. This small diagnostic does not benefit from distributing GPU training.
Preserve actor15.612726% TRAIN /18.369946%VAL and all calibrated critic snapshots.
After results: choose the next FIT-only interior decision-preserving contrast,
not another unchanged floating pass. Cumulative readmission still includes
all48 absent phase2 features. Live/STL source qualification remains separate.

## Completed

Both seeds completed in2.91s of diagnostic work (6.26s service including
startup), CPU peak1.2GiB.12 focused tests and scoped Ruff pass. The independent
collector authenticated3 diagnostic files and20 source files and reproduced
32 ranking/count/regret checks. Analytic gradients agree with autograd within
1.62e-8. Source weights and receipts are unchanged. No optimizer update,
new trading result or live promotion occurred.

The full frozen store is loaded by the shared loader. The receipt's
validation_read=false denotes no validation observations entering this
diagnosis: no2024 inference, scoring or event selection. All151 probe rows
are inTRAIN;2023 MONITOR is diagnostic only.

Of113 FIT anchors,82 are flat with entry allowed. Under the recorded current-
policy continuations, the actor chooses a native-optimal action in25/82.
This is not a win rate, nor a whole-market accuracy estimate.

| Diagnostic on82 executable FIT anchors | Seed19 | Seed43 |
|---|---:|---:|
| Critic chooses native-optimal action |52|34|
| Critic regret, native reward units |0.096926|0.186532|
| Critic local objective lowers optimal probability mass |27|33|
| Critic shared first-AdamW projection lowers it |52|60|
| Same shared projection without entropy lowers it |52|60|
| Native-FIT local objective lowers it |3|3|
| Native-FIT shared first-AdamW projection lowers it |64|59|

These are local derivative projections, not executed updates or predicted
profit losses. Counts need not track average derivative magnitude. Both seeds
still show critic ranking error and shared-head interference. Replacing critic
targets alone does not prevent that interference; removing entropy alone also
does not resolve this observed count. No claim of exhaustive root-cause closure.

## Next Contrast

Rebuild the FIT-correct protection set and FIT-error targets from this72-input
actor and its refreshed native continuations; do not reuse69-input guard rows
or old margins. Use the already tested interior feasible projection with
nonlinear decision-preservation and trust checks. Compare against this frozen
actor, fixed8-update budget and both schedules; protect optimizer state on
rejection. Diagnose the proposed target direction before increasing training.
Do not switch to unconstrained SAC/AdamW or run another unchanged floating pass.

The unresolved COT source-identity issue still prevents live promotion. These
diagnostics concern the pinned historical store, not a repaired/current feed.

Identity:`b79c3735a3559f2febd42c4b884a924c225679cd77e3c69327a301b0d83c2d38`.
Private result:`sac72-boundary-diagnosis-20261112-r1`.
Evidence:[summary](generalization/selected72_diagnosis_20261112/SUMMARY.json),
[integrity](generalization/selected72_diagnosis_20261112/INTEGRITY.json).

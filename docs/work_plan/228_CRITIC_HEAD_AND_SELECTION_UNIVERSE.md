# Critic Head Contrast and Selection Universe

Owner approved continuation2026-11-09. Preserve selected69 and selected70.

## Requirements and Tests Before Execution

| Requirement | Evidence required |
|---|---|
| Isolate online critic head weights | Both arms clear AdamW memory; same epoch40 actor, encoders, alpha, labels, replay and update schedule |
| Initialize effective twin weights, not unused module copies | Reset both linear layers of each effective critic with native Linear initialization; deterministic seed; test changed heads and unchanged other tensors |
| No false original parity after an intentional edit | Verify native epoch40 calibration before the edit; label step0 after edit separately |
| Preserve original models | Diagnostic checkpoints only; no actor update, no promotion |
| Detect action imbalance | Per-action row counts, RMSE/bias and zero references for all three existing sampling trajectories; absent actions reported as absent |
| Execute on intended hardware | Actual RTX5090 name checked before data loading |

Two sequential arms,4096 updates each,batch128,all24025 TRAIN transitions.
Both start from epoch40 and empty optimizer state. One keeps current online
heads; the other uses native PyTorch Linear initialization,seed7,independently
for each twin. No encoder or target initialization changes. Target soft updates
remain native but do not define the realized-return training labels. Same
sampling/dropout RNG reset after intervention. Reuse of previous numeric results
is descriptive only; run the paired control with this instrumented code.

Verification:32 focused CPU tests passed. Pinned runtime3153f902 launched the
sequential queue on the physically authenticated RTX5090; retained-head control
passed source/calibration checks and reached256 updates with actor/encoder parity.
The queue automatically verifies both receipts and writes the paired comparison.
No numeric result for the fresh-head arm is claimed before its receipt exists.

## What the Current Return Pass Actually Does

Verified in `feature_selector.search:TwoPassGreedy`: `removed` is initialized
empty on every call; pass2 traverses accepted removals from pass1 only.
`tools/run_refit_mask_pass.py:dataset_for` constructs the universe from the
current structural recipe, not the original phase1 admission list.

Last authenticated selection therefore tested70 ->69 ->69. Its return pass
tested RAW_CLOSE_LOG_RETURN only, not every feature removed during all earlier
phase2 cycles. This is an ordered two-pass heuristic, not an exhaustive floating
search or a proof that earlier removals cannot now help.

The actual pinned plugin0.2.4 package digest was independently reproduced:
`cbb9f0da336c2e593c3a0901bfe1b1d5f2dda542eb6936e49e8bd40fa6ec1cfb`.
Its archived return-pass trace contains exactly that one reinsertion, rejected:
score0.2702053639 before versus0.2557127528 with the column restored (VALIDATION
return per week). This verifies execution scope, not just current source intent.

## Cumulative Readmission Design

Keep the immutable phase1-admitted universe and a reasoned exclusion ledger
across phase2 cycles. A separate cumulative return pass should consider all
currently absent phase2 columns (coherent encoding blocks remain atomic).
Acceptance compares each candidate to the current baseline using identical
support, seeds, budget, preprocessing, activity gate and declared score.
Deletion accepts equal performance for simplicity; insertion needs improvement.

Physical channel removal means the current frozen69 model cannot consume an
old excluded feature. Restoring an earlier checkpoint is not an equal comparison
because its surviving weights differ. Preserve old modules for provenance, but
reconstruct missing channels/branches and qualify baseline parity while those
channels are masked; then use a matched bounded refit for both baseline and
readmission candidate. New features with zero connecting weights cannot prove
usefulness through inference alone. Do not label that experiment an unmask-only
pass, and do not claim cumulative readmission already ran.

Phase1 exclusions remain separate: leakage, unavailable data, constants and
invalid transformations must pass their original repair/quality gates before
readmission. Valid features rejected solely for weak proxy evidence or redundancy
can enter an explicit rescue pool, evaluated with the actual model. Do not restore
all phase1 exclusions indiscriminately or change the completed selection results.

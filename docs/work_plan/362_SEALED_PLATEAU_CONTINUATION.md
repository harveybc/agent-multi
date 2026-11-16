# Sealed Plateau Continuation

Owner authorized execution2026-11-16. Preserve the most recent complete359
recovery generation on each worker before suspending the original services.
Do not mutate pinned359 code, its results, checkpoint identity or original queue.
New experiment identity362 references the authenticated old recovery identity.

## Predeclared Execution

Each worker first runs actor-only ReduceLROnPlateau from its own sealed complete
state, then a fixed-LR continuation from the SAME seal for EXACTLY the same number
of additional epochs. Differences between seeds/source arms are not causal LR
rankings. The reference selected299 and source best remain selectable/preserved.

Keep all data, network, optimizer moments, replay contents/writer and sampler RNG,
Torch/NumPy/Python RNG, native business, CPU/GPU dispatch and weight decay unchanged.
New scheduler starts counting at the branch point, preserves historical best,
does NOT apply retrospective reductions, and uses patience=ceil(20/3)=7, factor=.5,
absolute threshold1e-8, cooldown5, actor floor1e-7. Critic and temperature unchanged.
Allow at least20 new epochs, original minimum100 total epochs, and20 epochs after
each actual reduction; retain original hard cap2000. Fixed-LR arm matches the
eventual scheduler budget, not its stopping rule. Save scheduler state in recovery.

## Requirements And Tests

| Requirement | Evidence required before launch |
|---|---|
| Durable source | full immutable generation, SHA verification, seal independent of rolling pruning |
| Resume not warm start | exact model/optimizer, replay sampling and next update reproduction on original GPU |
| No old experiment rewritten | copied identity migration receipt; original runtime and data untouched |
| One factor | actor rate changes only; critic/temperature and AdamW state unchanged at scheduler event |
| Scheduler edge cases | eighth miss reduces for patience7; abs signed metric; cooldown/floor/activity/NaN; restart exact |
| Stop coordination | historical best never reset; new20-epoch budget and20 after actual reductions, hard cap wins |
| No duplicated evaluation | use metrics already produced once each epoch |
| Automatic chain | persistent single-worker queue, scheduler then equal-budget control, plots/results and failure receipt |

The weight/margin/action/native-account diagnosis remains approved. No speculative
reward/representation/entropy change is bundled into this scheduler contrast.
Numerical execution must first reproduce the source native account; if it fails,
do not launch that worker until the discrepancy is resolved.

## Implementation Verification

Seals captured epochs69/86/43. Original services explicitly stopped/disabled after
sealing; their SIGINT exit is operator suspension, not a new training failure.
Full CPU regression suite:47 passed with the native engine enabled. Seven focused
scheduler/seal/function-trace checks then passed, including ten-epoch integrated
memmap recovery interrupted after the first LR decrease. Ruff/diff checks applied.

An integration test caught torch's `mode_worse=-inf` sentinel breaking strict JSON
recovery. It is now encoded explicitly as a symbolic sentinel and restored to the
native value on load; no metric, tolerance or best score changed.

Full-policy changes/KL/margins are recorded from the SAME logits used for the
single per-epoch native evaluation. Detailed action/position/command/trade traces
are saved at fork+1,+8,+9, before and after the earliest possible LR reduction.
No second per-epoch TRAIN account is run for this instrumentation.

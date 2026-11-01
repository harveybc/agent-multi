# Paired Stopping Contrast Launch

Implementation:847d2a73, published to master before deployment.
Campaign identity:542a947cd6249f861c04e6d03c4d2b9f70b249ebe2cf039290a0e574a47c5638.
Design:[183](183_PAIRED_STOPPING_CONTRAST.md).

Started2026-11-01T04:52:23UTC (2026-10-31 23:52Bogota).
RTX5090 handles seed7; RTX4090 Laptop handles seed19. Both use fresh financial
weights and replay, not the previous financial checkpoints or synthetic
weights. Parent source/data/config and qualification proofs authenticate.
Six frozen stopping rules per seed share one sequential learning trajectory.

The previous three-signal campaign is closed and remains unchanged. Seed43
failed synthetic qualification; it is not replaced or counted as a financial
success. This contrast explicitly applies to the two qualified seeds only.

## Execution

Two enabled persistent workers and one CPU collector. Worker cap6GiB;
resource admission4GiB available RAM and3GiB free GPU memory. Maximum two
process attempts, contract failures terminal, completed receipts authenticated
before adoption. An interrupted fit restarts fresh, without claiming replay
recovery. Initial process state: both active with zero restarts, roughly
3.8/4.0GiB service memory, CPU collector reports no communication issues.

Every periodic checkpoint is retained. Each rule's chosen checkpoint freezes
at its own stop; later optimization cannot overwrite its decision. Distinct
chosen checkpoints receive full TRAIN/INNER_A accounts, emissions, side-specific
metrics and close counterfactuals. Identical choices reuse those accounts.
Collector produces RESULTS.md, AGGREGATE.json, AUDIT_REQUEST.md and CLOSE.json.

## Verification and Timing

35 tests passed plus scoped Ruff and whitespace checks. CPU end-to-end smoke
passed with128 financial transitions,29 updates, two monitor checkpoints,
full TRAIN/INNER_A replay, exact weight reload and18 authenticated artifacts.
The monitor's CPU/CUDA RNG guards remain active. Original-control monitor
prefix comparisons will be recorded from the first assessment onward.

Initial ETA from the preceding measured fits: approximately6-12hours for the
paired campaign, depending on all six stopping rules. Maximum1048576 decisions
per seed remains unchanged. This is two training trajectories, not twelve
independent runs. No new financial performance is available at launch.

No2024/TEST access, no replacement data, no changed inputs, no hyperparameter
search and no phase2 selection in this run. The output informs the fixed
evaluator for the next feature-selection comparison.

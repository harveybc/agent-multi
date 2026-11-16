# Online Transient Results

2026-11-16. Scope: current359 continuations, not new training, phase2 closure,
architecture search or a model promotion. Same71 inputs/28groups, full original
TRAIN2020-2023 and validation2024. No TEST read. Selected299 remains unchanged.

## Completed Evidence

Twenty authenticated checkpoint replays: epochs0/1/5/10/20/24 and last copied
epoch38 for actor_lr01/seed7,48 for control/seed19,24 for all_lr01/seed43.
Each uses256 fixed TRAIN probes and256 fixed validation probes. Twelve diagnostic
actor-head/encoder exchanges,16 frozen greedy/sampled native accounts, and a
separate same-checkpoint device/attention-dispatch witness completed.

### Original Producer Accounts

| Arm | Copied epoch | TRAIN return % | Validation return % |
|---|---:|---:|---:|
| Initial selected299, all arms |0|16.058224|18.369946|
| Actor LR /10 |1|18.245971|9.737358|
| Actor LR /10 |38|74.638161|-2.486646|
| Control |1|-40.121283|-10.354562|
| Control |48|890.749030|-17.310113|
| All LR /10 |1|-60.142598|-19.062384|
| All LR /10 |24|19.668488|-7.416707|

Different seeds currently occupy different treatments: this is not a paired
ranking of learning rates. The three queues still perform the other treatments
sequentially. These are continuations of an already trained actor, not claims
that a random network learned the market at epoch0.

## Findings

### Actor And Critics Initially Disagree

On the fixed TRAIN probe, the initial greedy actor differs from argmax(min twin Q)
on230/256 rows (89.84%); validation202/256 (78.91%). Initial actor-to-soft-Q KL is
24.97877 on TRAIN. After one epoch it falls to1.78708,0.84227,4.53172 for actor-LR,
control,all-LR respectively. Agreement with the critic improves while the native
validation account degrades. Critics are not a ground-truth oracle: this is an
objective/decision mismatch witness, not proof that their preferred action is right.

At the last copied epochs, TRAIN disagreement is40.63%,29.30%,7.42% respectively.
All-LR has the smallest disagreement and lower critic loss, yet negative validation.
Therefore neither lower TD loss nor closer actor/critic agreement certifies useful
convergence. Replay distributions evolve, so critic losses across epochs are not
errors measured on an identical fixed target set.

### Early Representation Drift Matters

Restore only initial actor encoder or only initial head, leaving the other part
at epoch1. These are CPU diagnostic hybrids, not trained/promoted candidates:

| Arm | Epoch1 validation % | Restore initial head % | Restore initial encoder % |
|---|---:|---:|---:|
| Actor LR /10 |9.737358|9.875613|13.746687|
| Control |-10.354562|-7.866971|2.093384|
| All LR /10 |-19.062384|-20.535405|3.787596|

Restoring the encoder recovers part of the early loss in all three, but none
recovers the original18.369946%. At late epochs the effects are mixed: restoring
the encoder in actor-LR38 gives-4.161%, worse than its current-2.487%; all-LR24
improves from-7.417% to+2.509%. Module co-adaptation matters. These interventions
do not identify one defective layer or justify permanently freezing the encoder.

### Sampling Versus Greedy Is A Large Account Difference

Frozen weights, same bars/costs/business, independent common uniform streams
101/202/303. Native position evolution is simulated for each account, not reused.

| Frozen policy | Greedy TRAIN % | Sampled TRAIN range % | Greedy VAL % | Sampled VAL range % |
|---|---:|---:|---:|---:|
| selected299 |16.058|-49.845 .. -43.825|18.370|-21.962 .. -10.470|
| actor-LR38 |74.638|-36.205 .. -12.981|-2.487|-29.372 .. -13.884|
| control48, CPU |894.791|281.296 .. 356.483|-17.493|-34.025 .. -24.144|
| all-LR24 |19.668|-31.924 .. -5.697|-7.417|-6.802 .. -5.013|

Initial greedy trades:745 TRAIN/207 VAL. Sampled:2275..2339 TRAIN/590..605 VAL.
Training samples categorical actions; monitoring uses greedy actions. This is
normal SAC machinery, not automatically an implementation bug. However, it
creates a materially different trading trajectory for this inherited actor.
This finite witness is not replay of the historical changing-policy collector,
nor proof that greedy collection would learn better. Three draws are descriptive,
not an uncertainty calibration or a robustness guarantee.

### Numeric Replay Is A Separate Issue

The original strict CPU parity check stopped at control48;19/20 copied checkpoints
match native producer returns/trade counts, one does not. The mismatch is retained.
Original checkpoint SHA256:
`fd9d721a611f9424425fa2c0b4d844595db74c021982b2d3d1d4829af8aa2a0d`.

| Execution | TRAIN % / orders | VAL % / orders |
|---|---|---|
| Recorded producer GPU |890.749030 /2774|-17.310113 /448|
| Original GPU, producer FP32 settings |890.749030 /2774|-17.310113 /448|
| CPU default |894.790663 /2775|-17.492998 /450|
| CPU, attention fastpath disabled |894.790663 /2775|-17.492998 /450|
| Original GPU, attention fastpath disabled |894.790663 /2775|-17.492998 /450|

CPU/default GPU differ at7 action rows; max logit difference0.016330242. With
attention fastpath disabled on both, max logit difference0.00031661987 and0 action
differences; complete native accounts agree. Thus this witness isolates execution
dispatch sensitivity, not corrupted weights or a random evaluation. Original GPU
replay is exact. It does NOT explain away the validation loss in the producer.

Two initial GPU diagnostic attempts are not authoritative: the first omitted the
producer's cuDNN TF32 setting; the second began before the copied script finished
updating. Both outputs are preserved. Only the subsequent completed-copy FP32 run
and explicit no-fastpath runs above are used. The probe now rejects source-file
mutation during execution. Do not patch or relabel running359 results.

### No Proven Gradient Explosion

Epoch-mean gradient-norm ranges: actor-LR0.04335..0.10558,
control0.05461..0.92941,all-LR0.05356..0.42098. No nonfinite updates were reported.
These are MEANS, not per-update maxima; they cannot exclude short spikes.
Control TRAIN mean absolute common Q rises0.20856 ->2.02998 while action-gap
returns close to its initial value0.08020 ->0.07824. Increasing Q level alone is
not evidence of better action ranking or an exploding optimizer.

## Decision And Next Work

1. Keep359's three automated queues unchanged through their declared minimum100
   epochs, maximum2000. Patience20 cannot terminate before100. Compare late versus
   initial checkpoints and then paired treatments within each seed. No requirement
   that more epochs must recover validation is assumed.
2. Preserve299. Training reliability remains open; do not close phase2 or launch
   broad DEAP on these observations. No new features, state feedback or scaling.
3. Before the next training intervention, predeclare a bounded TRAIN-only test of
   objective alignment: frozen-policy sampled versus greedy value targets, the
   existing actor's action ranking, and actual native outcomes on held TRAIN blocks.
   If entropy is intervened on, change that factor alone with identical initial
   weights/optimizer/data/budget and multiple seeds. Do not simultaneously switch
   collection, reward, architecture and learning rate or optimize on these VAL rows.
4. Require execution settings and action parity in future cold-replay contracts.
   The dispatch witness must accompany a numerical portability fix; do not silently
   make old GPU and new CPU results the same claimed experiment.

Related G3 control/90:10 synthetic contrast has completed40 epochs each. Both
retain epoch0; last TRAIN/VAL:462.707207/-21.171065% versus514.532937/-22.077415%.
This augmentation did not improve the selected model. It is not a general verdict
on synthetic data or evidence that the inputs contain no useful information.

## Evidence Index

Private artifact root: `/var/tmp/long-sac360-20261116/`. Snapshot copies and weights
remain there. Plot: `analysis-r2/TRANSIENTS.png`; full accounts/hybrids:
`analysis-r2/REPORT.md`, per-seed JSON and `RESULT.json`. No training state modified.

| Relative artifact | SHA256 |
|---|---|
|analysis-r2/RESULT.json|f00994eeabcc314b33bfc675e19bd96ed952d2def558568cf6ad1c8a0534e9fc|
|collection/RESULT.json|fef34126e0e301ec455e97aa538cd4e338da0411dd82f5c745b1b2f5acbb183f|
|parity360-gpu-fp32/RESULT.json|12eb166fbafd4ff90451234c50db2706c7e5eb2b3bfadbb1cec744f14494eec0|
|parity360-gpu-no-fastpath/RESULT.json|999dffc2b43ead22025d5a141a4ffd1ec570268018282398b4d0efc6c42e63fe|
|parity-cpu-no-fastpath/RESULT.json|d1a170ebe0e18eca9e7fad9b912f41d69bb3dd11aa7506629a1ccbcae7c8b21d|

Commands are exposed by `tools/analyze_online_transients.py`,
`tools/diagnose_sac_collection_policy.py`, `tools/probe_checkpoint_device_parity.py`.
The analysis-run helper digest predates only the later nonfinite-alpha rejection
test; no measured finite-alpha calculation changed. Tests verify actor-only swaps,
common-Q invariance, invalid probe rejection and categorical sampling boundaries.

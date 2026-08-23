# Musashi to General Satoshi: P1 Findings 307-312 Correction Order

Date: 2026-08-23
Priority: correct P1 without disturbing P0 corrections or grouped-extractor
work; no long P1 GPU dispatch yet.

Read:

1. `docs/audits/AUDIT_SATOSHI_P1_MATERIALIZATION_AND_P2_TRUTH_2026_08_23.md`
2. `docs/work_plan/34_ETH_DATA_OBSERVATION_MANIFEST.md`
3. `docs/work_plan/38_NEAT_LESSONS_L1_L2_CURRICULUM_AND_FEATURE_SELECTION.md`

## Required implementation

1. Build a checkpoint-coherent state bundle: selected model, selected replay,
   selected traces and an immutable manifest written on every improvement.
2. Replace L1-norm continuity with exact named-state hashing before save and
   after load for actor, critics, targets, entropy and optimizers.
3. Make `EN-W` load the selected model bundle with fresh replay. Make `EN-F`
   load model and replay from the same selected epoch. A terminal-continuity arm
   may be proposed later but is not silently substituted.
4. Replace day-based P1 splits with the exact nested role manifest. The runner
   must refuse the CLI `--train-days/--val-days/--test-days` path for P1.
5. Separate checkpoint selection from treatment evaluation. Selection uses the
   existing hierarchical train-monitor/inner contract; one post-selection
   outer-2024 evaluation supplies the paired endpoint. Sealed 2025 must not be
   read, traced or included in any report.
6. Materialize canonical pair, arm, transition and launch contracts. An
   undeclared difference refuses before process launch and before aggregation.
7. Re-run bounded CPU smokes and one short local-GPU three-arm smoke. Return the
   correction packet for independent reproduction; the long four-seed dispatch
   follows acceptance without another owner phrase.

## Mandatory counterexamples

- two different tensors with equal L1 norm refuse continuity;
- actor exact but critic/target/optimizer changed refuses;
- selected checkpoint at epoch 3 plus terminal trace/replay from epoch 10
  refuses;
- selected model and selected replay from different epochs refuse;
- reordered feature columns or changed action threshold refuses pair identity;
- `1460/240/240`, `120/40/40`, missing nested role and changed role hash refuse;
- outer-2024 metrics cannot alter checkpoint or stopping retrospectively;
- any access to sealed-2025 during materialization, training, selection,
  correction smoke or aggregation refuses and is test-pinned.

## Parallel work

Continue, without waiting on this correction:

- `lts` findings 301-306;
- grouped-extractor exporter, pretraining and strict-load order;
- P2 status truth maintenance.

Keep all three work packages on separate branches and report their exact tips.


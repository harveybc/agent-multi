# 383: no qualifying pre-fold upstream bundle found

Question from [382](382_SHORT_FACTORIAL_RESULTS.md): can the source-label
population contrast be replayed for annual 2021 and 2022 folds with the
same 71 channels and all upstream choices genuinely preceding scoring?
No, not with the evidenced bundles below. Do not start a GPU replica.

Read-only CPU census of accessible RECIPE.json files under /var/tmp and
~/.local/share/agent-multi on three campaign hosts. Match ordered column
names by SHA256 of their compact UTF-8 JSON list, not count or filename.
Bundle count means the five neighboring runtime/receipt files are present;
it does not certify their weights. Each matching recipe failed on dates alone.

| Execution site | Recipe files | Exact order | With bundle files | With TRAIN and scaler ending before or at either fold start |
|---|---:|---:|---:|---:|
| Coordinator |283|81|77|0|
| Worker A |622|105|101|0|
| Worker B |181|47|47|0|
| Total copies |1086|233|225|0|

Every match declared the same exclusive end for TRAIN and scaler:
2024-01-01. There were 11 missing/inaccessible path reports per host;
counts exclude unreadable subtrees, remote archives and checkpoints
without a matching recipe. These are not unique-model counts or proof
that no other artifact exists anywhere. A bare weight file is not evidence
of which observations were used to produce it.

* Active 299 bundle: verified model SHA256
  69b5254489d224d4b20ac358956e4dd3b9a67a2f214a9aff98045c5def25a536.
  Its fixed scaler was projected from a 2020-23 TRAIN fit, not refit.
  Its manifest binds the export to selection identity
  b67b6fd1d05762e279a90cb57ddc4b2f313a0bbecc29ad681e44443a04765802.
  The [299 decision record](299_SELECTED72_CUMULATIVE_SELECTION.md) explicitly
  selected columns on 2024 validation return/week. Resetting just the closer,
  actor or scaler cannot retroactively make this feature list pre-fold.
* [372](372_SUBSET_PRODUCER_REBUILD.md) and [374](374_MATCHED_HISTORY_TRAINING.md):
  6/12/24/48 months are trailing histories ending in 2024, not expanding
  prefixes ending before 2021 or 2022. The 6-month scaler starts July 2023.
  The activity producer's offline cross-fitting is not a forward-only receipt.
* Inventory command: python3 docs/work_plan/generalization/fold_inventory383.py
  followed by the ordered-list digest
  b3fd815379412b504815bdf28fbf6a9134b8266db0bc2b305879bfd22f8fadd6
  and the two search roots. Output contains local paths; keep it private.

Decision: retain 382 as conditional evidence about short heads. For a strict
historical replay, reconstruct each fold's upstream using only preceding
bars and resolved labels: feature discovery, fitted producers, scalers,
risk profile, actor/encoder, checkpoint selection and reference long closer.
Verify fit/selection intervals and suffix-perturbation invariance before
spending GPU time. Fold-specific lists may differ; pair both training
populations on the exact same representation *within* each fold.
Alternatively, to keep today's 71 inputs and exact checkpoint, freeze both
closer variants and compare prospectively on newly arriving data, not by
relabelling 2021/22 as unseen. No release, floating pass or live change.

# Architecture Gene Preflight

2026-11-14. Phase2 closes in351. This is the beginning of the NEXT optimization
phase, not more feature selection or live-trading work.

## Executed

The existing old `simple_sac_search.make_model` constructs its earlier simple
architecture without the selected attention/explicit critic regularization.
Reusing it unchanged would silently change the control. It has not been launched.
The new contract starts from selected299's actual MODEL_CONFIG and keeps its
attention, positional encoding, categorical action semantics and fixed inputs.

CPU constructor/decoder audit:18 genes,88 requested boundary/interior/choice
cases,54 distinct phenotypes,34 explicitly recorded phenotype aliases. All54
constructors consume their requested values and all54 checkpoint reloads repeat
the probe logits exactly.176 append-only request/result events are preserved.
The lower/upper endpoints of a categorical bin can intentionally decode to the
same value; these are labeled cache hits, not additional evaluated models.

No epoch, optimizer step or financial evaluation is part of this audit.
The checkpoints are freshly constructed witnesses, NOT newly trained champions.
The selected trained299 bundle remains untouched.

## Initial Verified Domains

These are configurable search domains, not claimed optimum values. Full JSON:
[gene contract](../../examples/config/phase2_architecture_gene_contract.json).

| Gene | Constructor domain |
|---|---|
| Shared group width |4,8,16,32|
| Shared group residual blocks |1,2,3|
| Core width |8,16,32,64|
| Core residual blocks |1,2,3,4|
| Critic head width |4,8,16,32|
| Actor head width, independent |4,8,16,32|
| Attention trailing patch |1,3,5,7 observations|
| Attention embedding |4,8,16,32|
| Attention heads |1,2,4; divisibility checked|
| Attention feed-forward width |8,16,32,64|
| Attention initial residual gate |0..0.5, linear|
| Learning rate |1e-6..1e-3, log10|
| Discount |0.95..0.9995, linear|
| Target interpolation tau |1e-4..0.02, log10|
| AdamW weight decay |0,1e-6,1e-5,1e-4,1e-3,1e-2|
| Entropy fraction |0.3..0.9, linear; constructor only|
| Initial alpha |0.001..1, log10; NOT the trained checkpoint alpha|
| Online-critic core dropout |0..0.3, linear|

Each residual TCN block contains two causal convolutions, so block count is not
the number of individual convolutions. The attention adapter includes fixed
sinusoidal positions; no positional-encoding omission is allowed.

## Candidate Record

`app/architecture_gene_contract.py` implements strict normalized-genome decoding,
exact endpoints, categorical-bin witnesses, log-scale interpolation and a locked,
fsynced hash-linked JSONL event ledger. Invalid domains are rejected, not clipped.
Tests cover NaN, bad type/domain, tampering, duplicate/failure retention and
baseline round trips. Each construction event records raw genes, decoded config,
actual module/optimizer values, seed, phenotype identity, parent list, artifacts,
component parameter counts/norms and checkpoint hash. Trading metrics are absent,
not invented zeros. Search-time fields and metric producers must be wired through
the training runner before this becomes the DEAP ledger.

Run identity:`d4905aeb17cf56fefa83c95c45632db8c378c0f5423509816ef6ec8b686b11b4`.
Ledger tail:`ddfbd2823c41d81c6eb1a86d30154ade091c36ac5e43ec8ac9f306a901b3e7ee`.
[All receipts](generalization/architecture_genes_20261114/RESULT.json),
[candidate events](generalization/architecture_genes_20261114/CANDIDATES.jsonl).
Weight files remain in the durable local run root `architecture-gene352-20261114-r1`,
not in git or OLAP blobs.29 combined focused tests and scoped Ruff pass.

## Exact Continuation

1. Extend the existing configurable temporal builder for per-group widths,
   block counts/types, kernels/dilations/activations, variable-width fusion and
   head depth, preserving baseline output behavior. Verify causal-prefix,
   tensor-shape, positional, module-mode and checkpoint invariants first.
2. Connect the training genes, including batch32, actual replay sampler, update
   cadence, optimizer settings and stopping. Check the actual training path;
   an entropy_fraction field in a constructor is not proof its update is applied.
3. Adapt the DEAP runner to the frozen selected recipe, current architecture,
   paired seeds and explicit training/checkpoint/fitness contract. Persist raw
   genes AND effective configs, all metrics/curves/failures and weight diagnostics
   with the candidate ledger; bind its records to the existing governed metrics
   schema after inspecting that schema. Do not reuse the old combined-monitor
   campaign without reconciling it with the current selection rule.
4. Only after those executable gates launch component-wise search followed by
   interaction search. Preserve20 population and the previously requested2000
   generation ceiling as search-budget defaults; stopping semantics must be
   versioned, distinct from epochs and from the historical pre-phase2 campaign.

The JSON explicitly lists unsupported dimensions as NOT ENABLED. This does not
claim exhaustive architecture parameterization yet. DEAP is not running; no
decision or credential from the owner is needed for these implementation tasks.

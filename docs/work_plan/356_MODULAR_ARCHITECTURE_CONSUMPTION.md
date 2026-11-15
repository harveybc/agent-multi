# Modular Architecture Consumption

Continuation of352 while355 G3 remains running on its pinned runtime. No live
work, STL, input selection change, new trading rule or DEAP launch in this step.

## Requirements And Tests Frozen Before Implementation

| ID | Requirement | Behavioral acceptance |
|---|---|---|
| A1 | Preserve baseline | Old constructor checkpoint and fixed logits remain exact; unchanged parameter keys/shapes/order |
| A2 | Per-group temporal modules | Independent widths/counts/kernels/dilations/activations; fused input channels equal sum of group widths |
| A3 | Causal alternatives | TCN/pointwise configured explicitly; future perturbation leaves sequence prefix unchanged with attention enabled |
| A4 | Independent heads | Actor/critic depth, width and activation consumed; exactly three categorical logits and two Q surfaces |
| A5 | Strict decoding | Reject unknown fields, bad types, mismatched group/dilation counts and contradictory pointwise kernels |
| A6 | Training consumption | Finite SAC update changes parameters, entropy target follows configured fraction, batch32 works; exact checkpoint reload after update |
| A7 | Traceability | Actual resolved stages and receptive field recorded; no constructor witness reported as trading performance |

Architecture: retain existing causal convolution/residual TCN classes, concatenate
group channels into the current fusion projection, retain causal patch attention
and positional encoding, then configurable temporal core and separate head depths.
No new input preprocessing or agent state. Optional fields default to the exact
old path. Group overrides follow the pinned recipe's resolved group order.

Unit design: constructor edge cases and explicit-default parity; integration:
group concat through attention/core/head plus TorchRL twin expansion and update;
system: reload preserved selected299 bundle/probe on CPU, build variants from its
real71-column recipe, verify finite outputs and checkpoint reload. No full training
or new fitness claim. The immutable G3 runtime is not modified.

Remaining after this bounded step: expand the versioned genome and audit every
requested boundary, wire replay/update/stopping genes through the actual runner,
then paired component search. Unsupported dimensions stay disabled, never silently
ignored. G3's result is evaluated separately before admitting synthetic training.

## Implemented And Verified

`DoubleTCNExtractor` now accepts `group_configs` in the pinned resolved-group
order and `core_config`. Each stage supports `width`, `blocks`, `kind` (TCN or
pointwise), `kernel`, a positive integer `dilations` list of exactly one entry
per block, and `activation` (silu/relu/gelu/tanh). Fusion concatenates the actual
sum of group widths; its activation is configurable. Unknown fields and invalid
pointwise kernels are rejected. Defaults preserve the previous constructor.

`ThreeSignalSAC` supports head depth/activation and independent actor head
width/depth/activation. Output semantics stay categorical NOP/BUY/SELL, with
opposite direction closing externally. Critic-only dropout mode remains intact.
There is no input clipping, new normalization, feedback or risk head.

Each residual block has TWO causal convolutions. The recorded per-token nominal
receptive field without attention is `1 + 2*(kernel-1)*sum(dilations)` per stage;
composed stages add their fields minus one. Attention accesses the full causal
prefix; the final flattened head sees all13 observed positions regardless of
individual token fields. Stride2 denotes sampled rows, not guaranteed elapsed
clock hours across market closures. The metadata does not conflate these notions.

The CPU system audit completed32 cases on the real71-input/28-group recipe:
baseline, each of28 individually changed groups, two core alternatives, and
independent heads. Each performs one batch32 fixture update, checks finite loss
and actor/critic changes, tests temporal prefix invariance, and reloads both
actor logits and twin-Q outputs exactly. No trading return was measured in this
audit; a fixture update is not a trained candidate or a fitness evaluation.

**Baseline comparison:** old and new code at the same checkpoint on the same
CPU give byte-identical logits. The older saved32-row probe differs from BOTH
implementations by at most1.1920928955078125e-6, with zero changed argmax actions.
The first strict comparison to the exported probe consequently failed; the
same-host old-code witness isolates this from the new implementation. Do not
claim cross-environment exported logits are byte-identical. The audit retains
both results separately and does not modify the preserved bundle.

85 focused tests pass including the native engine recovery/replay test;9 GPU
parameterizations are skipped because verification was CPU-only. Scoped Ruff
passes. An initial native test invocation pointed at an older checkout lacking
`AbsoluteBarEnv`; rerunning against the parent experiment's actual engine fixture
resolved those seven import failures without production-code changes.

Full external receipt: `modular-architecture356-20261115-r1/RESULT.json`, SHA256
`a7a98a9dbf36c24931812b27fa659d088c92938f24473de1c8d945cd7d2775ff`.
The32 updated fixture checkpoints remain beside that receipt, not in Git.

## Exact Next Work

Generate the next gene-contract version against the pinned group order. Stage
overrides supersede shared fallback dimensions: mark shadowed fallback genes
inactive rather than pretending mutations to them change modules. Extend nested
gene paths to list entries, test all boundaries/choices and persist requested,
resolved and actual module values. The v1 gene contract is historical and must
not be edited to retroactively claim these new cases were part of352.

Then verify batch32/64/128/256, replay sampling/capacity, warmup, update cadence,
learning rates and stopping through the actual runner. The present update tests
prove batch32 and dynamic entropy consumption, not every training-loop gene.
No DEAP fitness campaign has been started. G3 remains pinned to its original
runtime; this constructor refactor does not alter its ongoing comparison.

## V2 Genome Continuation Completed

The next part above was also executed in this session. `app/modular_gene_contract.py`
generates a recipe-identity/group-order-pinned contract with192 genes. Nested list
paths now reach each group individually. The four shared fallback dimensions are
excluded from the search when per-stage overrides replace them. Inactive kernel
and dilation genes have an explicit active mask and requested/effective receipt.
Changing a pointwise dilation to an arbitrary canonical1 initially broke exact
numerical parity despite algebraic equivalence. The decoder now freezes that
inactive value to the baseline value instead; regression restores exact logits.

Domains added per group: width4/8/16/32, blocks1/2/3, TCN/pointwise mode,
kernel1/3/5, geometric dilation base1/2/3, activation SiLU/ReLU/GELU/Tanh.
Core widths8/16/32/64 and blocks1/2/3/4; other stage domains as above. Actor and
critic head depths1/2/3 and their activations are independent. A stage may also
accept an explicit dilation list, but arbitrary lists are not genome dimensions.
Mode selects a kernel constraint, not a new unrelated network implementation.
These are configurable domains, not empirically optimized values.

1,171 boundary/interior/choice requests map to419 constructed configuration
phenotypes after effective-config deduplication. Every constructed candidate
is checked against actual module dimensions, both convolutions' kernels and
dilations, activations, head depths and optimizer values, with finite logits.
32 representative cases additionally perform a batch32 SAC fixture update and
exact actor/twin-Q reload. All192 genes pass these consumption witnesses; this
does not claim419 fitted trading models or test every joint gene combination.

Final regression:134 passed,9 GPU-only cases skipped; scoped Ruff passes.
V2 contract: [JSON](../../examples/config/phase2_architecture_gene_contract_v2_selected299.json),
SHA256 `3ca0d937181e285b6c05656e69f6a4abe10a400ecf3cbbcdaaaadb2cadb015b8`.
Final external result root: `modular-architecture356-20261115-r3`, receipt SHA256
`d9f75fac8906e3f246d5979d6c67f4559b5c555d897559cbfd18d5730d8df076`.
Earlier r1/r2 evidence remains, not overwritten. The exact NEXT action is now
the training-loop gene/ledger integration, followed by paired component DEAP
after G3 adjudication. No live work or repeated feature-selection cycle.

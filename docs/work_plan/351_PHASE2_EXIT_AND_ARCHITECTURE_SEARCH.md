# Phase 2 Exit And Architecture Search

2026-11-14. Owner correction supersedes the recent-data/retraining next step
in349/350. Musashi owns offline phase2 completion and subsequent architecture
and training optimization. Satoshi owns live trading. No live deployment,
periodic-retraining campaign or provider work is part of this transition.

## Current Evidence

- Selected299:71 physical inputs,28 groups,49 absent phase2 return candidates.
-299 completed deletion then cumulative return over the original120-feature
  phase2 universe. Phase1 rejections and paused STL columns are not reopened.
-350 completed and independently verified two bounded actor-head trials.
  Neither qualified. Retain299; close that critic/actor micro-refinement line.
- Frozen checkpoint reproduction is verified. This is not a claim of identical
  outcomes from independent fresh training or globally optimal features.

## Finite Phase2 Closure

1. Authenticate299 recipe, scaler, input order, feature ledger, donor weights,
   exact support, account engine and complete native baseline metrics. Reuse
   verified evidence instead of retraining a control.
2. Build a final-context coverage table: every retained coherent block removed
   from the FINAL71 set; every absent phase2 block reintroduced to that SAME
   set. Historical299 early deletions were measured on an evolving subset,
   so an ordered pass alone is not a final-neighborhood certificate.
3. Reuse prior evaluations only when full evaluation identity matches, including
   selected subset, source/donor weights, tensor order, rows, scaler, engine,
   strategy, objective and execution mode. Evaluate only missing neighbors.
4. Preserve299's selection rule: eligible validation return/week; more than
   two orders/week in both TRAIN and validation; publish both separately.
   Removal is retained on non-worsening score, addition only on strict gain.
   Preserve existing numerical comparison semantics; do not invent a new
   epsilon or silently change objective. Keep coherent sin/cos and probability
   blocks atomic. A failed evaluation is not a tie.
5. If no eligible removal ties/improves and no addition improves, close as
   LOCAL_FROZEN_NEIGHBORHOOD_STABLE. This is conditional on frozen weights and
   donor-assisted returns, not a global or fully retrained subset optimum.
6. If a move qualifies, run one ordered deletion/cumulative-return cycle and
   verify/export its physically compacted model. Allow at most TWO corrective
   cycles, with a final-context coverage check after each. If exhausted while
   moves remain, close as BUDGET_CLOSED_NOT_LOCAL_OPTIMUM and retain the best
   eligible model under the existing rule. Never claim convergence from a cap.
7. Verify cold reload, masked/physical action and account parity, finite tensors,
   feature/group counts and full cumulative return inventory. Publish the exit
   receipt and freeze inputs for the next phase. Do not require another full
   refit merely to close a frozen-model selection procedure.

This is a proposed executable closure protocol, not a claim that the final
coverage table or gate has already run. Implementation must verify it against
the existing selector before dispatch. Identity mismatch leads to targeted
repair, not a restart of unrelated experiments.

## Next Phase: Configurable Search

Inventory the actual consumer and plugins before publishing numerical ranges.
Each parameter must have a type, default, bounds or enumerated options, scale,
conditional activation, decoder, effective destination and a witness test.
Unsupported options require implementation/tests before entering the genome.

| Component | Parameters to inventory and expose |
|---|---|
| Each profile-group extractor | Layer type, count, widths, kernels, dilations, residual/projection paths, activation, normalization location/type, dropout |
| Fusion | Projection width, fusion operator, attention enablement, heads, head dimensions, feed-forward width, positional encoding |
| Temporal core | Layer types/count, widths, kernels, dilations, receptive field, residual paths, activation, normalization, dropout |
| Actor head | Depth, widths, activation, regularization; categorical output semantics remain fixed |
| Twin critic heads | Separate depth/width choices, activations, regularization, target update settings |
| SAC training | Learning rates by component, optimizer settings, batch size including32, replay capacity/sampling, warmup, update frequency/ratio, discount, target interpolation, entropy settings, clipping |
| Stopping | Evaluation cadence, patience, minimum delta, checkpoint criterion and initial monitoring policy |
| Strategy | Only already-authorized configurable coefficients; separate search block with fixed business invariants |

Temporal window/stride, data, selected inputs, fixed TRAIN scaling and business
contract remain fixed in the first architecture comparison. Do not silently
reduce data or windows to accelerate a candidate. Preserve causal execution,
required positional encoding and valid channel/head divisibility. Log receptive
field in bars/hours and coverage of the unchanged window.

Use staged component searches followed by a joint interaction search: extractors,
fusion/core, actor/critics, training, then compatible combinations. Every stage
uses the current pinned baseline and paired seeds/support. Do not infer that a
change beneficial alone remains beneficial in combination. Search limits,
fitness and stopping must be frozen before dispatch; distinguish DEAP generations
from model epochs and selection fitness from training loss. This document does
not launch DEAP or silently reactivate the old pre-selection campaign.

## Mandatory Candidate Ledger

Persist one append-only evaluation record per candidate/seed/fold, and an
aggregate decision record. Store unsuccessful candidates, timeouts, OOMs,
rejected genomes, duplicates and cache hits as well as champions.

- Campaign/schema/code/data/split/recipe identities; candidate ID; generation;
  parents, crossover, mutation, RNG state/seed and search-space version.
- Raw genotype, active-gene mask, decoded requested phenotype, repaired genes
  with reason, fully merged effective config and ACTUAL constructed model spec.
- Effective parameter values read from modules/optimizer; parameter counts per
  component, tensor dimensions, receptive fields, memory and runtime.
- TRAIN/validation objective components separately: return, net profit, Sharpe,
  drawdown, trades/week, BUY/SELL counts/wins, close reasons and early-close
  counterfactual diagnostics; fitness and constraints, not just a scalar rank.
- Epoch curves, stopping reason, patience history, selected epoch, actual steps,
  evaluation identity, hardware/software determinism settings and failure cause.
- Checkpoint/tensor hashes, scaler/input order, optimizer state and artifact paths;
  cached result provenance and duplicate phenotype identity.
- Weight diagnostics per extractor, fusion, core and head: norms/drift, gradient
  statistics and saturation/dead-unit checks where meaningful; fixed-input
  activation, logit/action changes and account impact. Cross-topology tensors
  are not directly comparable without an explicit correspondence.

Keep bulky tensors/curves outside git; store references and hashes with durable
metrics. Map the ledger to the existing governed OLAP schema after inspecting
it, without guessing columns or writing to unrelated tables. Successful export
and reimport must preserve gene/metric linkage for every evaluated candidate.

## Decoder And Analysis Gates

Before expensive training, test lower/upper bounds and interior points for each
active gene through JSON merge, constructor and checkpoint reload. Demonstrate
actual model/optimizer change. For integer/categorical genes test each boundary;
for log-scaled genes test orders of magnitude. Declare intentional aliases and
inactive genes explicitly. Reject accidental fixed parameters, silent clipping,
unreachable options and stale-cache identities. Include these regressions in CI.

Gene-space analysis means measured parameter-effect and interaction analysis,
not inference of performance by reading a genome. Separate conditional genes,
seed variation, topology differences and unequal compute; account for adaptive
search sampling. Use paired interventions to test suggested causes, not only
correlations among selected winners. Combine with weight/function diagnostics
to identify what a parameter changed. Preserve observations for later surrogate
or meta-optimization, which is not automatically enabled by this plan.

## Immediate Order

Authenticate final-neighbor coverage -> evaluate only missing cases -> bounded
correction if needed -> phase2 exit receipt -> executable parameter inventory,
decoder/ledger tests -> launch staged DEAP. No live-trading work, endless critic
microvariants, unchanged full floating loop, or STL resumption in this sequence.

## Closure Implementation

`app/phase2_closure.py` checks all one-block neighbors of a FIXED final subset,
then conditionally invokes the existing ordered deletion/cumulative-return
procedure, for at most two corrections. `tools/run_phase2_closure.py` authenticates
the original expanded actor and donor tensors rather than rebuilding a new
restoration context from a different source. It reuses sealed299 caches only
under the original evaluation identity. Its runtime retains the299 inference
implementation; a later actor implementation is not silently substituted.

| Requirement | Behavioral check |
|---|---|
| Final-context completeness | Counterexample where an earlier harmful removal becomes neutral after another removal |
| Cumulative return | Historical absence can return; coherent paired channels remain atomic |
| Exact existing decision rule | Deletion ties qualify; insertion ties do not; inactive/NaN candidates never qualify |
| Bounded termination | Budget exhaustion is explicitly different from local stability |
| Replay | Cached restart repeats all decisions without new inference |
| Source identity | Source/donor/runtime/support hashes and exact expanded tensors before cache use |
| Deployment-independent evidence | Physical export only if changed; full CPU TRAIN/validation replay; no live action |

Initial coverage census:111 final-context coherent-block neighbors,107 present
in the299 cache and4 missing. This census is not the completed exit certificate.
18 focused CPU tests pass, scoped Ruff passes. No fresh training is required
for this frozen-neighborhood check.

## Completed Exit

Status: **LOCAL_FROZEN_NEIGHBORHOOD_STABLE**, identity
`18ee3fd442c0f6ffe7a50b54a9e6cbcca74db61b9123b343e7bf27df404957cc`.
The111 neighbors comprise67 retained coherent-block deletions and44 cumulative
returns covering all49 absent phase2 features.107 neighbors were reused from
authenticated299 evidence;4 were newly evaluated, plus the reused baseline.
No move qualifies under the existing rule; no corrective cycle or new physical
export is needed. The retained model is exactly299,71 inputs and28 groups.

| Metric | TRAIN | Validation |
|---|---:|---:|
| Return |16.058224%|18.369946%|
| Sharpe |0.781969|2.690712|
| Orders |745|207|
| Orders/week |3.564593|3.833333|

Producer evaluation took10.98s (service including imports/preflight17.86s).
The first automatic verifier invocation encountered a Torch interop-thread
reinitialization error AFTER the producer sealed its results. Fresh-process CPU
verification completed in7.03s service time, reproducing both accounts and
independently enumerating every final neighbor. No selection evaluations were
repeated. The orchestrator now invokes verification in a fresh CPU process.

29 focused tests pass, including altered cache source/device/column-order/result
rejection and the bounded-selector/decoder/ledger tests. Scoped Ruff passes.
The original runtime remains pinned; only the orchestrator's future process
handoff was repaired. This closes phase2 for the current frozen-weight method,
not all possible joint feature/architecture combinations. A later materially
different architecture may justify a separately identified selection study,
but does not automatically reopen this one.

Evidence: [result](generalization/phase2_closure_20261114/RESULT.json),
[CPU verification](generalization/phase2_closure_20261114/VERIFIED.json).
Next: [352 construction/lineage gates](352_ARCHITECTURE_GENE_PREFLIGHT.md),
then configurable component builders and training-path consumption before DEAP.
No live, STL, periodic-refit or critic microvariant was launched.

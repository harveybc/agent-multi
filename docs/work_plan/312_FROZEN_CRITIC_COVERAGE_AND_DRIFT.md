# Frozen critic coverage and policy drift

Owner approval2026-11-12: execute Musashi's remaining diagnostic while Retsu
qualifies COT/live sources. This follows310 and311; no new training or model
promotion. All measurements explicitly use the historical mixed-COT store.
They diagnose the310 failure, not corrected-data trading performance.

## Questions and frozen protocol

1. How accurately do calibrated300 critics rank actions outside their113 FIT
   anchors? Measure on a new deterministic stratified panel, excluding all151
   old FIT/MONITOR timestamps, before seeing continuation returns.
2. How much do native returns change when continuation follows each310 fresh
   head instead of the preserved299 actor? Start from exactly the same account
   snapshot and first action, so this isolates continuation-policy drift.
3. Does a candidate prefer actions the fixed critic overvalues? Record native
   regret of actor choices, Q-ranking regret and action-gap errors, separating
   flat/executable, flat/restricted, long and short states and calendar years.

Choose at most12 flat/executable,4 long,4 short and2 restricted rows per year
2020-2023:88 new anchors maximum. Seed312 fixed. Report unavailable strata,
never synthesize positions or choose anchors by future return. Reuse the
original300 intervals2020-2023 and2023-2024, not new truncated horizons.
Both intervals are TRAIN diagnostics;2023 is NOT a new independent validation
set because310 actor heads trained over all four years. No2024 inference.

At each source-policy snapshot, intervene NOP/BUY/SELL and finish the full
declared interval under preserved299 or each of four310 heads. Freeze first
action, account, engine, costs and inputs. Native reward returns use gamma0.99
and actual decision duration, matching300's plain-reward calibration targets.
Do not add entropy to only one side. Cash returns are reported separately;
they are not critic target units. These matched-state continuations are NOT
an additive explanation of the whole changed-policy account loss, nor a
measurement on the candidates' own state visitation distribution.

Also score existing151 labels using the same saved critics. New-point nearest
FIT distance is RMS over the original fixed-scaled input window, descriptive
only, not a learned distance or noise estimate. No threshold is optimized.
Summaries include each twin and minimum, ties, regret, errors and action counts.
Do not pool oversampled strata as an unbiased market-wide success rate.

## Tests and acceptance

| Requirement | Evidence |
| --- | --- |
| Frozen, disjoint panel | Deterministic sampling, duplicate/non-TRAIN rejection, exclude151 |
| Same-state intervention | Snapshot hashes, first effects identical across policies, no parent mutation |
| Native correctness | Source first-action suffix reproduces full interval cash; repeated first witness |
| Matching units | Plain discounted reward, separate dollars; tie-aware regret and rankings |
| Actual frozen critics | Authenticate300/310 receipts; candidate critic tensors exactly unchanged |
| Resumption | Identity/config pin, atomic per-anchor result and sealed digest; no silent skipped errors |
| Unchanged research scope | Zero optimizer updates; original checkpoint unchanged; no VALIDATION/TEST inference |

CPU sequential queue on the secondary worker; cached batched network inference
only once. Per-anchor receipts, progress with measured remaining time and a
final machine-readable report require no intermediate approvals. Existing
STL GPU queues remain independent. No automatic training or phase2 selection
from this historical-source diagnostic.

# Local Window And Component Optimization

## Current Follow-up: Missing Neighbours And Shared Seeds

Owner continuation2026-10-21: fresh13 seed7 is closed and reproduces the
reference policy tensors, business traces and metrics exactly. No mismatch;
TRAIN-14.547669%, VALIDATION-7.278253%, validation DD10.731872%,203 trades.
The following new sealed study completes the omitted local comparison without
rewriting any previous experiment: first12/14 seed7, then13/12/14 seed17,
then13/12/14 seed29. Eight new fits, all mandatory irrespective of promotion.
The original13 seed7 is reused only after its exact fresh reproduction.

Implementation uses the same native runner and common suffix35 evaluator,
new schema temporal_local_window_sweep.v1. One worker, one heavy GPU process,
preferred GPU sequentially; strict baseline-config parity admits only window
and declared seed changes. Preserve all native metrics, compare each neighbour
with13 for all three seeds, report paired daily-growth differences, DD and
TRAIN/VALIDATION gap; do not select a convenient seed or claim a global optimum.
Seed7 promotion remains the existing conservative rule; multi-seed evidence
is reported separately and cannot silently rewrite that decision.

TRAIN2020-2023, VALIDATION2024, TEST closed,15 selected inputs,stride2,
100000 transitions,99000 updates,final100k and four diagnostics; no refitting,
no expensive ES, no model/DOIN change. Window12 spans22 observed intervals,
13 spans24 and14 spans26. Warmups23/25/27 differ; heads/initialization can
differ with shape, as already documented. Common-time support is authenticated;
pre-suffix positions and account histories are not reset. Native costs retained.

Automation reuses bounded two-attempt recovery, durable task ledger, GPU/RAM/
disk admission, per-shape synthetic GPU/reload witness, governed outbox/readback,
independent closed-artifact collector and automatic successor dispatch. Closure
requires all expected common reports read back. Integrity errors fail closed;
resource waits resume by timer. The next input-subset study remains planned,
not an implicit experiment started after this finite batch.

Traceability: new PRE3 expected failures; POST local finite-queue replay verifies
all8 fits even when13 wins, six unconditional confirmations, no duplicate fits,
publication wait then close, strict maximum, predecessor identity and tampering.
Actual12/14 CPU gradient/reload witnesses use actor1030/1094 and critics1574/1702.
Deployment verified: runtimed0f05548,spec2d3cd84c,first contract75a887f6.
Window12 seed7 TRAIN started23:52:34UTC;4000/100000 in80.6848s verified.
GPU witness16 updates/exact reload,actor1030/critics1574. Worker26 tests pass;
local63 targeted tests and2 native-engine integrations pass. The initial local
integration command named a nonexistent gym checkout and failed before loading
the environment; rerunning with the pinned checkout passed both. No code or
scientific setting changed to obtain that pass. GPU readiness renewed before
dispatch; host~72GiB disk free, independent archive~673GiB free. Baseline
common reports2/2 readback and first independent copy verified. Worker,
publisher and collector timers enabled; user lingering enabled on the worker.
Provisional full-batch duration~9-11h from the completed4207s control, not an
extrapolation of the initial random-action1000 steps. No new final metric yet.

## Previous Cuts

Latest result2026-10-21: fine window queue ALL_TRIALS_CLOSED4/4, no failures.
VALIDATION returns9=-31.7603%,11=-34.4700%,13=-7.2783%,15=-26.5253%,17=-19.7006%.
Retain13 under this recipe/seed. No refinement or confirmation was activated.
Native publication current; common reports10/10 readback,5 independent closed
copies including control. No fit remains active; next is the planned subset
study, whose driver/configs are not yet deployed. Launch notes below are history.

Owner-triggered comparability audit follow-up: five archived runs verified,
including real weights and metrics. Fresh13 seed7 reproduction is now running
at the sweep runtime, contractf2437f30, with automatic closure comparison.
The original conditional refinement skipped12/14 when13 remained incumbent:
that is incomplete local refinement, not evidence that13 is optimal. Next
window design must include those missing neighbours and shared-seed checks;
do not rewrite the closed sweep. Its fit execution and the scope of its
conclusion are separate questions. Planned subset work is retained afterwards.

Owner authority: 2026-10-21. Small improvements matter; optimize components
experimentally rather than assuming that greater capacity helps every component.
Weekly adaptation remains last. TEST stays closed. DOIN is not modified.

## Research Posture And Authority

Owner clarification2026-10-21: use evidence without optimistic or pessimistic
framing. A finite failed contrast only limits that recipe, data, budget and
evaluation protocol; it does not exhaust SAC, RL, inputs or trading strategies.
Do not recommend abandoning the research because a limited grid failed. The
owner decides whether to end or redirect the program. Close finite experiment
batches as specified, retain all outcomes, and use observations to update the
next hypotheses and controlled tests. Distinguish observation, interpretation
and proposed experiment. Continue improving the static comparative foundation
before measuring periodic adaptation; do not require profit>0 at this stage.

Bayesian/uncertainty follow-up: record the owner's successful predictor-thesis
experience as motivation, not a measured RL outcome. The separate collaborator
lane plans that contrast; do not duplicate its runs or silently import its
implementation into SAC. Later, assess a compatible Bayesian/uncertainty arm
against its deterministic reference, with the same inputs/business/budget and
recorded inference sampling cost. Separate the layer effect from the effect of
using uncertainty for decisions/sizing; preserve mandatory SL/TP/volume limits.
Measure uncertainty calibration and trading metrics under a declared protocol.
This is planned research, not a modification to the active window queue.

The static stage is a comparative foundation for subsequent periodic adaptation,
not a profitability admission test. Select better net performance/risk and less
harmful TRAIN/VALIDATION divergence even while returns are negative. Do not
require profit>0 to proceed to controlled adaptation. Record every trial and
all outcomes, independent of sign. Adaptation benefit is an experimental
hypothesis to measure, not a result assumed before its comparison.

## Ordered Program

Owner addendum: interleave [incremental input subsets](57_INCREMENTAL_FEATURE_SUBSETS.md)
between steps1 and2, before expanding extractor/component capacity. It includes
the provisional15-feature priority table, cyclic coding/transform checks and
bounded interaction rescue. No active request, worker or input identity changes.

1. Fine window search: stride2, observations9/11/15/17 versus the closed13
   control, seed7. If one improves the declared validation/risk/gap rule, run
   its two integer neighbours. If none improves13, do not manufacture a winner.
   The second stage can retain its first-stage incumbent. Confirm the finalist
   versus13 on shared seeds17/29 only if it differs from13. At most10 new fits,
   two fresh-root attempts each. Sequential preferred GPU, no manual boundaries.
2. Extractors: compare per-feature Conv1D and per-feature Dense, not flat Dense
   versus a whole temporal architecture. Keep the selected window, raw inputs,
   dates, business, costs, seeds and training budget fixed; report matched or
   bracketed parameter counts, order-sensitive structure and exact transforms.
   Existing Dense results with different heads/monitor/capacity are NOT a
   matched family comparison and cannot automatically replace Conv1D.
3. Separate component refinements: per-feature width/depth, fusion/bottleneck,
   core width/depth/reduction, actor head and critic heads separately. Reuse the
   prior capacity ladder as evidence, not as proof that all settings are ruled
   out. Begin at the current small recipe. Freeze each tested search range and
   budget before dispatch; these later grids are planned, not implemented here.
4. Interaction check: cross shortlisted window and component settings before
   declaring a final recipe. Coordinate-wise winners are conditional on their
   companions, not jointly optimal. Recheck a previous component when another
   changes materially; use bounded shortlists rather than exhaustive products.
5. Learning/action/reward diagnostics proceed alongside on CPU; cost/accounting
   defects take precedence over optimization. No cost removal to create profit.
6. Adaptation cadence LAST: no refit control, quarterly, monthly, weekly; shorter
   history/cadence experiments only under separately frozen causal protocols.
   Predict before incorporating the period just predicted. Do not reuse the
   old contaminated post-update evaluation as prospective performance.

All stages use TRAIN2020-2023, VALIDATION2024, mandatory SL/TP and volume,
one entry/day,12h close/reopen envelope and identical assumed costs. The active
window stage keeps selected15; the planned subset study derives separately
manifested subsets of those inputs without changing the running identity.
Static SAC uses100000 transitions with diagnostics25k/50k/75k/100k, final100k,
no expensive early stopping, no checkpoint selection. Keep weights, effective
configs, data/code/selection digests, seeds and native OLAP readback/copies.
Positive development improvement is not a guarantee of trading profitability;
always report no-trade, TRAIN and VALIDATION profit, DD, trades, win rate and PF.

## Fine Window Support And Selection

Request v12 admits integer windows8..18 only, stride2 and the fixed minimal
encoder. Stage1 is9/11/15/17; stage2 is winner-1/winner+1, skipped if13 remains.
Native training and evaluation warmup is unchanged. Preserve full-period native
metrics. The additional selection score uses a fixed common trace suffix after
35 observed context bars (maximum window18, stride2). The actual timestamp hash,
row count and dates must equal the reference in both splits; missing interior
rows are rejected, not interpolated. Growth and DD are recomputed using equity
immediately before that suffix as the anchor. No reset, new fit, or fee refund.
Positions/account histories can differ before the common suffix; state that
limitation explicitly. This is common-time mark-to-market comparison, not an
identical-start-account counterfactual. Activity is the native full-period count.

Choose higher common VALIDATION mean daily log growth (delta>1e-8), non-worse
common DD and non-worse positive TRAIN-minus-VALIDATION gap, with activity,
using the existing promotion predicate. This is a conservative trade-off, not
an assertion of a uniquely optimal size. Rank eligible candidates by growth,
then DD. Repeat all three seed-pair outcomes transparently. Do not pick seeds.

## Test Design And Traceability

| Requirement | Tests / evidence |
| --- | --- |
| Old experiments unchanged | v11 still7/13/25; existing context/capacity regressions |
| Finite automatic fine/refine/confirm | recipes, conditional stage, max10, queue replay |
| No parameter drift | v12 strict keys/types/windows/encoder and common-config parity |
| Equal score timestamps | suffix timestamp hashes, count/date checks, missing-row rejection |
| Correct score anchor | known equities, warmup costs retained in native metric, shared suffix DD |
| Closed inputs immutable | source trace hashes checked; no writes inside prior campaigns |
| Native temporal learning | actual-window gradient/reload and prefix/current/future probes |
| Publication and recovery | existing queue/timers/warehouse readback/independent collector |

Evidence: PRE6 expected failures; POST170 focused checks; native real-engine
monitor integration4 passed,8 deselected, including fine windows8/18 and old
windows7/25. No scientific new fit is claimed by these synthetic CPU checks.
Deployment: runtime6916e49a, spec9e13d797, first contracta7931e4e. Native
window9/stride2 training verified1000/100000 in12.7458s on the preferred GPU.
Synthetic GPU witness16 updates and exact reload, actor934/critics1382.
Worker CPU checks48 passed,11 optional prefix probes skipped (those ran locally).
Worker/publication/independent collection timers enabled; baseline common2/2
reports read back and baseline physical copy verified. Later windows11/15/17,
conditional neighbours and shared-seed confirmations require no human boundary.

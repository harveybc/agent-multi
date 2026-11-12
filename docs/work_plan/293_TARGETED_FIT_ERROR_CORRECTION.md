# Targeted FIT Error Correction

Owner-approved2026-11-11, following292. Preserve the qualified69-input/28-group
source actor, all data/support, current-policy continuations and26 protected
source-correct FIT anchors. Reuse292 controls without retraining them.

## Prespecified Objective

From FIT2020-2022 only, select source mistakes whose measured continuation
returns have a unique best action (top gap >1e-8). Fix the target action and
advantage `w = return(best)-return(source_action)` before updates. Exclude
ties, source-correct rows and all monitor/validation rows from the corrective
objective. All targets come from the agent's own native trading simulator,
with frozen-policy continuations; no ZigZag expert, future observation or new
input. This is an experimental advantage-weighted offline actor improvement
objective, not the unchanged SAC policy-gradient loss.

Minimize `-sum(batch_multiplicity*w*log pi(best))/sum(batch_multiplicity*w)`.
Weights are normalized, not clipped. Each original128-row FIT batch is filtered
to its fixed mistake rows, preserving duplicate multiplicity. An empty error
batch refuses rather than claiming a zero-loss learning step. The saved critics,
representations and alpha stay frozen. Entropy is not added to this corrective
loss; the old native SAC objective remains a separately reported diagnostic.

Use the same projected-guard AdamW mechanism,26-anchor preservation floor,
trust budgets, minimum movement and rollback as292. Two schedules19/43,
eight attempted updates each, fixed endpoint8, points0/1/8. The seeds now vary
sampling only for the corrective objective, not independent data draws. No
relaxed guard, bigger window, missing inputs or longer schedule on failure.

## Requirements And Tests

| Requirement | Test / evidence |
|---|---|
| FIT-only fixed mistakes | Correct rows/ties/monitor excluded; perturbing monitor returns cannot affect targets or loss |
| Correct target weighting | Positive exact advantage; weighted CE checked manually; common scale invariance; repeated batch entries preserved |
| Requested action direction | Analytic logit derivative equals autograd; preferred action margin points upward under independent-logit descent |
| Real translation through shared weights | Per-step sampled requested/actual preferred margins, sign agreement, probability change; full FIT/monitor error counts |
| Protect acquired behavior | Exact292 guard identity, source-mode panel equality, nonlinear margin check and frozen tensors |
| Reliable execution | Matched source/Q/schedule hashes, finite queue, canonical checkpoint reload, rejected optimizer rollback |
| Actual trading | TRAIN/VALIDATION full accounts, direction/win/activity/DD/close diagnostics, symmetric chronological attribution |

Report decreases in correction loss separately from corrected greedy decisions
and monetary gain. The native best action is conditional on the source-policy
continuation, not a globally optimal trading policy. No validation-based target,
step budget, coefficient choice or checkpoint selection; no TEST/live promotion.

## Completed Results

Runtime `183255f5`; identity
`9dc979acb7bab817a9a9b0628077690775f64629b52501fcc75c1372305193d1`.
Two arms completed automatically on the primary GPU in28.48seconds, followed
by1.12seconds of full account attribution. Worker and collector exited0;
114 worker files and15 report files passed collection hashes.33 focused tests
and scoped Ruff passed before execution.

There are63 fixed FIT mistakes (total native advantage22.833581),26 protected
FIT decisions and23 diagnostic-only MONITOR mistakes. Targets are conditional
native continuation returns, not hindsight features supplied to the model.

| Arm | Accepted / attempted | Corrected FIT errors | TRAIN return % | VAL return % | TRAIN / VAL orders |
|---|---:|---:|---:|---:|---:|
| Frozen export |0 /0|0|13.913055|16.336139|739 /207|
| Targeted projected19 |0 /8|0 /63|13.913055|16.336139|739 /207|
| Targeted projected43 |0 /8|0 /63|13.913055|16.336139|739 /207|

All16 linear projections solved successfully with nonzero proposals, but all
16 nonlinear backtracking searches rejected. Rejected steps restore both
weights and optimizer moments. No probabilities, greedy actions or trades
changed at the saved endpoints. This is **not** an improvement in learning,
profit or generalization; the protected reference remains the current model.

Unchanged account details, including trading costs:

| Metric | TRAIN2020-2023 | VALIDATION2024 |
|---|---:|---:|
| Profit USD |1391.305545|1633.613925|
| Sharpe |0.685588|2.400526|
| Maximum drawdown % |9.548436|3.257322|
| BUY / SELL orders |224 /515|23 /184|
| BUY / SELL wins |152 /304|14 /108|
| Total win rate % |61.7050|58.9372|
| Orders/week |3.5359|3.8333|
| Early / TP / SL / forced closes |54 /403 /220 /62|6 /110 /66 /25|
| Helpful / harmful early closes |25 /29|4 /2|

## Rejection Diagnosis

Independently replayed the first proposal for each schedule, including all24
backtracking fractions, without saving any candidate or changing the sealed
run. Both projection results and gradient norms reproduce exactly.

| First full proposal | Schedule19 | Schedule43 |
|---|---:|---:|
| Correction loss before |2.879971396|2.821819637|
| Correction loss after |2.879250116|2.821154884|
| FIT margin RMS movement |0.001758367|0.001831531|
| Worst protected margin change |-9.536743e-7|-3.278255e-7|
| Protected action flips |0|0|
| Failed gate |protected margin only|protected margin only|

Both full proposals satisfy all trust ceilings and improve the intended loss.
The linear projection lands on active guard boundaries (minimum predicted
margin change approximately0), but the actual float32 network loses tiny
protected margins. No useful first-step fraction passes: fractions down to
2^-10 fail protection; smaller fractions fall below the1e-6 movement floor,
with protection passing only once movement is effectively zero. All48 replayed
trials preserve the26 protected greedy actions, but decision preservation alone
is weaker than the fixed confidence-margin contract.

This identifies the immediate obstruction in the first steps as the strict
nonlinear margin gate, not a reversed loss or exhausted trust budget. It does
not prove that every rejection is pure roundoff, that no feasible corrective
direction exists, or that accepting these proposals would improve trading.
The discrepancy includes finite-precision evaluation and nonlinear departure
from a first-order boundary; those contributions are not separated here.

## Integrity And Next Step

Independent audit authenticates262 files and all four saved checkpoints;
source-correct membership, fixed error labels/weights, monitor exclusion,
source/final margins, untouched frozen tensors and zero accepted optimizer
steps reproduce exactly. Four full-period account deltas reconcile with0 USD
error. No live promotion, new floating pass or access to TEST.

Next: design a strictly interior projected proposal or nonlinear feasibility
repair on FIT, retaining the existing1e-7 acceptance tolerance and26 protected
actions. First test nonzero feasible movement and the requested/actual error
direction on this same panel, then use a separately identified fixed-budget
contrast. Do not relax the gate after seeing validation, inflate training epochs,
repeat the unchanged floating pass or discard the source weights.

Evidence: [manifest](generalization/targeted_fit_20261111/MANIFEST.json),
[summary](generalization/targeted_fit_20261111/SUMMARY.json),
[audit](generalization/targeted_fit_20261111/AUDIT.json),
[rejection traces](generalization/targeted_fit_20261111/REJECTION.json).

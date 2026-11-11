# All-State Critic And Actor Refinement Contrast

2026-11-11. Owner authorizes the successor to280. Preserve original69 and the
qualified one-update candidate. No repeated selection, live promotion, new
feature, state feedback, synthetic data, normalization or execution rule.

## Hypotheses And Design

274 fitted the actor on45 flat executable anchors only. Its longer updates hurt
validation, and calibrated opening ranking did not transfer. This contrast starts
from qualified control19, not its original parent, and includes holding states.

Record the actual greedy policy trajectory separately in2020-2022 FIT and2023
MONITOR. Sample8 flat,4 long and4 short states per calendar year, deterministically
with seed281, before calculating outcomes. All available states are used when a
stratum is smaller; report that count. At each of at most64 anchors execute all
three categorical actions in identical native state and then the frozen greedy
policy to the end of its own period. Never cross2023 into FIT labels or2024 into
MONITOR labels. Preserve thirteen observations at stride2 and complete native
TRAIN2020-2023 / VALIDATION2024 trading evaluations. The small anchor panel is a
bounded mechanism diagnostic, not a reduced training dataset or full refit.

Factorial: retained calibrated critic versus fresh online critic heads fitted
for4096 updates to these same-policy greedy discounted returns; actor entropy
coefficient retained versus zero. Both latter branches use the categorical SAC
actor objective, with alpha zero explicitly an objective ablation, not standard
maximum-entropy SAC. Critic encoders stay fixed. Actor heads alone then update
at fixed points0,1,8, endpoint8; seeds19/43, batch128 and existing learning rate
and weight decay. Schedules are identical within seed. No monitor/validation
selection, no automatic winning checkpoint. Critic initialization plus greedy
calibration is one composite intervention, not a claim isolating either alone.

## Traceability And Tests Before Implementation

| ID | Requirement | Test and execution evidence |
|---|---|---|
|R1|Period-local real counterfactuals|Reject late FIT terminal, monitor context before2023, wrong role/state, nonfinite labels|
|R2|Position semantics without network state inputs|HOLD/NOP equality for existing long/short; log flat restricted states separately|
|R3|Outcome-independent panel|Deterministic stratified timestamp sampling and unchanged choice after unrelated fields change|
|R4|Unchanged baseline and frozen modules|Native account parity, actor/encoder/alpha tensor comparisons and source rehash|
|R5|Actual actor update and cold export|Finite nonzero head changes; all saved points cold-reload exact logits and accounts|
|R6|Mechanism versus money|Q ranking/regret by state and split; actor probability-weighted target change; complete trading metrics and early-close counterfactuals|
|R7|Autonomous bounded run|Exclusive lock, identity-bound cache, per-stage receipts, persistent sequential queue and automatic collection|

Use existing tested native continuation, initialization, account, snapshot and
frozen-state APIs. No broker calls. State is used only to construct/audit native
counterfactual labels, never appended to model observations. Inference preserves
source trainability flags to avoid changing attention kernel dispatch.

## Reading The Results

Report reward-return and USD rankings separately: discounting can disagree with
total profit. Include monitored regret, equivalent-action errors, changed signals,
orders by direction, win rate, costs, drawdown and helpful/harmful early closes.
For every seed19 endpoint, automatically attribute TRAIN account differences by
chronological native replay; the contribution sum must equal actual profit change.
VALIDATION is descriptive for the fixed endpoints, not used to fit or choose them.
No pass implies global feature optimality or a live-model release. A useful
refinement requires a separate qualification before another changed-policy pass.

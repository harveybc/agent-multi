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

## Launch Verification

32 focused CPU/native-fixture tests pass; CLI import and authentication against
the real qualified bundle also pass. First launch stopped before any training:
the consumer expected an embedded identity while qualification279 uses the
digest of its configuration. Corrected with a regression; failed run preserved.
The corrected runtime is3170f91c, deployed to a new immutable directory and new
output identity, not edited over the first attempt.

Readiness verified the RTX5090 and available memory. Native baseline accounts
match the qualified actor. The64-state real counterfactual panel has started;
the finite service continues into critic calibration, the eight factorial arms,
cold replay and chronological attribution. A separate finite collector transfers
and verifies all terminal file hashes. Neither service requires a chat session.
Experiment identity:
`e0697e9f9fca6b3263bbd7fb85cfcc1bcf3a35026905253f4ade28d99213a8d4`.

## Completed Results

GPU worker completed in271.39seconds (274.08seconds including startup). Eight
arms, two4096-update critic fits, fixed actor points0/1/8, complete native account
evaluation and cold reload completed. Collector authenticated471 files and
exited0.33 focused tests pass including the independent-report additions. No
source checkpoint, data, feature set, account rules or live service changed.

The panel contains64 states and192 native continuations: FIT48 / MONITOR16.
Each year contributes8 flat,4 long,4 short. FIT has21 flat both-open-allowed
and3 flat restricted states; MONITOR has6 and2 respectively. The holding
equivalence gates pass. This is a small diagnostic panel, not proof of global
critic generalization. Full trading evaluation was not subsampled.

The table shows the predeclared eight-additional-update endpoints. The reference
already contains its separately qualified one saved update. No arm was stopped
or selected by the table. Seeds share one pretrained representation.

| Critic | Actor entropy | Seed | TRAIN return % | VAL return % | VAL Sharpe | VAL orders | VAL wins % |
|---|---|---:|---:|---:|---:|---:|---:|
| Frozen reference | unchanged |19|11.565540|14.591090|2.165596|206|58.2524|
| Retained | native |19|12.126289|12.128543|1.805259|209|56.9378|
| Retained | zero |19|12.167249|12.892204|1.908319|209|57.4163|
| Greedy calibrated | native |19|11.091066|13.465330|2.016616|208|57.6923|
| Greedy calibrated | zero |19|10.661944|14.018250|2.083461|207|57.9710|
| Retained | native |43|12.126289|12.128543|1.805259|209|56.9378|
| Retained | zero |43|11.345287|12.892204|1.908319|209|57.4163|
| Greedy calibrated | native |43|11.545821|14.693262|2.179247|206|58.2524|
| Greedy calibrated | zero |43|11.545821|14.693262|2.179247|206|58.2524|

All endpoints remain active. Full per-side orders/wins, costs, drawdown and
close types are in the evidence summary. Validation DD remains approximately
4.0446% throughout. The seed43 calibrated endpoints retain24 buys /182 sells,
with15 /105 winners,3.8148 orders/week and6 early closes (4 helpful,2 harmful)
versus the reference7 (4 helpful,3 harmful). They are saved research candidates,
not a live promotion. Their TRAIN change is-0.019719pp, VAL+0.102172pp. The other
calibration seed does not reproduce the improvement; do not report a robust gain.

## Symmetric Attribution And Constant Controls

After the seed43 improvement appeared, a separate, explicitly post-run report
was executed symmetrically for all8 endpoints in both periods, not just that
candidate. It performs no fit or selection. Runtimeb38c7e3a,52.68seconds,51 files;
all16 chronological contribution sums reproduce the account differences with
zero USD reconciliation error. Original four seed19 TRAIN analyses were reused
only after authentication; remaining twelve were executed independently.

- Retained/native: four NOP-to-SELL interventions account for the entire
  VAL loss of246.254685USD in each seed. Retained/zero has three harmful extra
  short-entry interventions, losing169.888535USD. These are intervention counts,
  not necessarily the difference in final order count after downstream effects.
- Calibrated seed19/native has two harmful extra short-entry interventions,
  losing112.575975USD; zero has one, losing57.283985USD. Entropy removal reduces
  the damage here but does not alone repair it.
- Calibrated seed43/native and zero each improve VAL by10.217220USD. One changed
  decision accounts for the entire improvement: with a short open, the old BUY
  symbol closed it early; the new NOP lets native forced-flat settle it within
  the same decision interval. This does not demonstrate better learned timing
  over many early-exit opportunities or authorize changing forced-flat rules.
- Their TRAIN difference is-1.971890USD: five effective opening changes, three
  helpful and two harmful. Their validation gain is not simultaneous dominance
  over the reference in both periods.

Critic monitor pairwise accuracy rises32.35% ->67.65% in both recalibrated seeds,
but the constant-symbol check prevents overstating this as pattern learning.
Seed19's minimum twinQ chooses BUY at all16 monitor anchors: regret0.088350,
exactly the constant-BUY baseline. Seed43 chooses NOP6 / BUY10 / SELL0, with
regret0.126776, worse than that same constant control. Here BUY is a categorical
signal: it can open long, hold long or close short depending on external state;
it is not a claim that the executed account only buys.

No greedy actor decision changed at the16 monitor anchors, despite small changes
in their probability-weighted return. Changes elsewhere in the full accounts
still caused gains/losses. Aggregate panel accuracy and expected return are
therefore insufficient gates for the next actor refinement.

## Disposition And Exact Next Step

The experiment and report are complete, no worker remains running. Preserve the
qualified69 reference and all candidate checkpoints, including both seed43
calibrated endpoints. Do not promote a seed selected after viewing validation,
repeat the unchanged floating pass, or resume joint TD training from these
diagnostic snapshots with unqualified target-critic/optimizer state.

Next design: TRAIN-only decision-boundary calibration concentrating on where
NOP becomes an executable SELL/BUY, with holding controls and constant-symbol
baselines. Audit the actual actor gradient at those boundaries, not just Q RMSE
or average panel ranking; validate on disjoint internal periods. The data do not
justify blaming missing input signal or discarding the representation. A new
candidate must pass qualification before another changed-policy floating pass.

Evidence: [manifest](generalization/all_state_refinement_20261111/MANIFEST.json),
[full summary](generalization/all_state_refinement_20261111/SUMMARY.json),
[return curves](generalization/all_state_refinement_20261111/RETURNS.png).
Report identity:
`5a4b3c0d65597e24e60bdd43ab7a2b243b76ea7501ed4004ee7e52b7346f16f7`.

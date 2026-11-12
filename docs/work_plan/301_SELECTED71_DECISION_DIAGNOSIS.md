# Selected71 Decision Diagnosis

2026-11-12. Owner requests diagnosis after300 critic calibration, before any
actor refinement. Preserve the71-input actor and both calibrated critic seeds.

| Requirement | Verification |
|---|---|
| Correct source | Authenticate300 calibration and299 export, original parent and full support |
| No actor update | Calibration-only diagnostic, zero optimizer steps; saved source hashes and in-memory actor parameters unchanged |
| Temporal scope |151 inherited TRAIN anchors,113 FIT and38 MONITOR; no validation observations or posthoc events |
| Executable decisions | Separate flat/entry-allowed, flat/blocked, open-long, open-short; label command preemption separately |
| Native comparisons | Same-state full-policy continuations for each first action; not independently additive trade P&L |
| NOP/HOLD/ties | Count all tied optimal actions, including equivalent hold symbols; no forced unique oracle label |
| Critic versus actor | Compare min-twin critic rankings and actor argmax against native reward and net account outcomes |
| Update interference | Fixed first128-sample batch from eight-step covering schedule, seeds19/43; local versus shared-head and first-AdamW projections, entropy/native-target controls |
| No fictitious finite effect | Projections are local derivatives, not executed policy updates or predicted dollar profit |
| Auditable output | Hash-sealed diagnostics, independent rankings and action strata; finite CPU execution, no live change |

Reuse the immutable57497834 runtime's tested calibration-only diagnosis.
The small derivative computation is CPU-only with two threads. Supplemental
stratification is read-only and must authenticate the same source arrays.
Actor reference remains TRAIN16.058224%,VALIDATION18.369946%,745/207 orders.
The panel is small and inherited, not exhaustive current-policy coverage.

The monitor stopping-budget question from300 remains a separately declared
contrast, not permission to silently replace the fixed4096 critic endpoints.
Any later corrective actor update must rebuild current FIT protection/error
sets and satisfy the existing interior feasibility and nonlinear trust gates.

## Completed And Verified

The fixed diagnostic completed in2.98s CPU work using immutable57497834,
with no optimizer step.26 focused tests pass, including explicit counterexamples
for tied-HOLD guard and target omissions. Scoped Ruff passes. The collector
authenticated3 diagnostic and20 calibration files, independently reproduced32
ranking/regret checks and1440 probability-derivative checks. Maximum analytic
versus autograd error is3.87e-8. Source weights and receipts remain unchanged.

The shared loader reads the historical store, but only151 TRAIN rows enter
inference, scoring and Jacobians here. No2024 observation or posthoc validation
event enters this diagnostic. FIT has113 rows;2023 MONITOR has38. The latter
is diagnostic, not an actor-fitting target.

## Concrete Decisions

Each count below is agreement with the best **same-state, first-action native
continuation**, allowing all tied optima. This is not trade win rate or a
whole-market estimate. The continuation uses the frozen actor thereafter and
ends at the role's temporal boundary. Closing comparisons therefore include
subsequent account opportunities, not just the current position's P&L.

| State | FIT rows | Actor optimal | Critic19 optimal | Critic43 optimal | MONITOR rows | Actor /critic19 /critic43 optimal |
|---|---:|---:|---:|---:|---:|---|
| Flat, entry allowed |82|26|52|35|28|8 /10 /6|
| Flat, entry blocked |6|6|6|6|2|2 /2 /2|
| Open long |12|9|9|10|4|2 /2 /2|
| Open short |13|7|9|10|4|2 /2 /1|

Blocked entries make the action alternatives equivalent in these8 cases; their
perfect agreement is not evidence of a learned abstention policy. For the82
executable FIT openings, actor chooses NOP46 times, BUY16, SELL20.40 of those
NOP choices miss a better entry continuation;2 selected entries occur where
NOP is strictly best. Critic19 chooses27 NOP/25 BUY/30 SELL;critic43 chooses
41/38/3. These are signal recommendations at fixed snapshots, not executed new
trading accounts. Critic43 remains directionally imbalanced in this panel.

Opening mean reward regret is actor0.271788,critic19 0.095422,critic43 0.187617
on FIT. On MONITOR,critic19 regret0.155034 beats all constant-symbol controls
(best0.222411);critic43 regret0.303542 does not. The better sampled ranking
does not justify replacing the policy wholesale.

### Early Close

All33 sampled open positions admit an actual agent close in the first-action
counterfactual. We distinguish this from stops/forced closes preempting an
agent command; tests explicitly cover that distinction.

| FIT state | Actor helpful /harmful closes | Actor missed beneficial closes | Critic19 helpful /harmful | Critic43 helpful /harmful |
|---|---|---:|---|---|
| Long,12 positions |0 /1|2|0 /1|0 /0|
| Short,13 positions |0 /0|6|2 /0|6 /3|

The actor misses8 useful FIT close opportunities and issues one harmful close
under the continuation objective. In MONITOR it issues no close on the8
sampled positions and misses4 beneficial closes. Neither critic resolves
closing consistently: on those8 MONITOR positions,critic19 has1 helpful/1
harmful close and misses3,critic43 has1 helpful/2 harmful and misses3.
No claim is made that these small sampled counts describe all real closures.

Reward-optimal and final-net-account-optimal action sets differ in7/82 FIT
and3/28 MONITOR opening probes. They agree in the sampled closing probes.
Discounted reward and terminal cash are distinct objectives; do not silently
call reward regret dollar loss or substitute a new reward during this audit.

## Shared-Head Interference

On the82 executable FIT openings, the following counts lower the actor's total
probability mass on native-optimal symbols in a local derivative calculation:

| Projection/control | Seed19 | Seed43 |
|---|---:|---:|
| Critic objective, independent logit direction |28|34|
| Critic objective, shared first-AdamW direction |50|60|
| Same shared direction without entropy |51|62|
| Native-FIT targets, independent logit direction |3|3|
| Native-FIT targets, shared first-AdamW direction |63|57|

The shared critic direction lowers optimal mass on12/19 currently correct
opening decisions, respectively. Replacing Q targets with native FIT returns
does not by itself remove shared-parameter interference. Neither does removing
entropy. These projections use fresh moments and the first fixed128-sample
batch, not the constrained interior update and not an executed finite step.
Counts are not aggregate derivative magnitudes or predicted trading losses.

## Tied-Action Coverage Finding

Two existing refinement helpers require a **unique native-optimal symbol**:
`protected_actions` in `app/fit_decision_preservation.py` and `error_targets`
in `app/targeted_fit_errors.py`. That requirement omits an important business
case: NOP and the same-direction signal are equivalent HOLD actions while an
opposite signal closes the position.

Measured on the current FIT panel:

- The legacy guard protects26 correct flat decisions, but none of the16
  correct HOLD decisions (9 long,7 short). Six all-equivalent blocked cases
  are also excluded, but do not require a distinction between actions.
- There are65 actual FIT errors; the unique-optimum target builder includes63.
  It omits one flat error and one harmful long close because multiple better
  actions tie. The harmful close should have a HOLD correction target set.
- A two-row synthetic witness keeps the unique-optimum guarded row unchanged
  and flips the unguarded HOLD row to a bad close: the old preservation gate
  still passes. That bad close also receives no old correction target.

This is a reproduced coverage omission, not proof that every historical loss
was caused by it. No real actor was altered for the witness. Historical
returns remain the measured outcomes of their recorded runs; they do not
certify protection of all correct HOLD decisions.

## Required Next Step

Before another refinement, implement a separately identified **optimal-action-
set** correction and guard, retaining the historical strict profile for replay:

1. Protect correct decisions against suboptimal symbols, including HOLD ties;
   allow switching within equivalent optimal symbols and skip all-equivalent
   rows where there is no bad alternative.
2. Include all positive-regret FIT errors. Optimize the probability mass of
   their complete optimal action set rather than dropping tied optima or
   arbitrarily treating one equivalent HOLD symbol as the only valid label.
3. Test both directions, harmful/beneficial closes, blocked entries, tied entry
   alternatives, no future/monitor fitting, infeasible proposals and rollback.
   The nonlinear accepted action must remain in the protected optimal set;
   probability-mass improvement alone is not sufficient.
4. Only after those checks, compare the fixed-budget guarded interior update
   against the preserved71-input actor. Keep native full-account parity and
   activity checks; no raw AdamW retry or new floating pass before a qualified
   changed actor exists.

Do not use this audit to add agent-state inputs, change scaling, move dates,
reset the actor, select a new critic endpoint retrospectively or promote live.
The separate stopping-budget opportunity from300 remains documented.

Identity:`bd6cbb3a07edfab4aef3dec59eca96d5b378c077012b7a82e1aff82d5dd1224f`.
Private root:`/var/tmp/sac71-decision-diagnosis-20261112-r1`.
Evidence:[summary](generalization/selected71_diagnosis_20261112/SUMMARY.json),
[state decisions](generalization/selected71_diagnosis_20261112/STATE_DECISIONS.json),
[guard and target coverage](generalization/selected71_diagnosis_20261112/GUARD_COVERAGE.json),
[integrity](generalization/selected71_diagnosis_20261112/INTEGRITY.json).

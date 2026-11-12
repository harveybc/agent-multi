# 304: Selected71 boundary reachability

Resume the exact next action of303 after delivering the live-input handoff to
Retsu. This is a finite CPU diagnostic, not another training sweep or feature
selection. Preserve export299 and the303 endpoints.

Before execution: authenticate301 Jacobians and303 source-value-floor receipt,
check row/target/source identities, and use only FIT rows to construct proposals.
For every source-incorrect FIT row and each tied native-optimal action, measure
the winner gap, a first-order head-distance lower bound, and the exact minimum
FIT margin-RMS change needed to cross the current winner in logit space.
The latter is only a one-step necessary condition, not a multi-step impossibility.

Build finite projected direction witnesses with all three action comparisons,
then add303's conservative source-margin constraints and the nonincreasing
first-order fixed-error objective. Keep the existing budget numbers. Compare
unguarded and guarded linear witnesses. Solver failure means unresolved by
this solver, never a proof of nonlinear infeasibility. A linear witness is
not a tested neural-network update, new checkpoint or simulated profit.

Tests first: three-way crossing, nonzero direction, zero sensitivity, strict
numerical tie buffer, and a coupled protection constraint. No MONITOR/VAL/TEST
target enters proposals. No actor weight is changed or exported. Hash the
inputs and outputs. The follow-up must test any useful witness in the actual
network with nonlinear rollback and native account replay before adoption.

Predeclared follow-up: test every guarded witness that meets the linear
RMS/drift budgets on the actual CPU network. No line search or refitting.
Check the unchanged nonlinear guard, loss, KL and drift budgets; restore all
weights even on exception. A passing sampled check still requires native
chronological account replay before adoption. This probe exports no model.

## Results

The diagnostic completed in2.08s CPU with no optimizer step.33 focused tests
pass. An initial unsealed attempt failed only while serializing a NumPy count;
the integer conversion was corrected and the complete run was repeated in a
new root. No source artifacts were changed or reused as completed results.

| Measurement | Result |
| --- | ---: |
| FIT rows / source errors |113 /65|
| Tied-optimum target cases |67|
| Trainable head parameters |1699|
| Winner gap min /median /max |0.005573 /0.802830 /17.372850|
| Error rows whose necessary one-step RMS exceeds0.002 |62 /65|
| Unguarded /guarded local witnesses found |67 /67|
| Guarded witnesses within linear RMS and head-drift budgets |1|

This does NOT prove62 errors cannot be corrected over multiple steps. The
bound describes a single step from the preserved source. Neither solver
success nor a local witness proves nonlinear feasibility or optimality.
The results also refute the claim that the margin constraints freeze every
useful direction: one useful guarded candidate exists within the declared
limits. The previous loss-descent path simply did not find that crossing.

### Actual-network probe

The sole predeclared eligible witness was checked in the CPU neural network,
without line search. FIT panel index63, native row9304, changes NOP ->SELL.
All113 FIT source-value floors pass;105 conservative margin constraints pass.
One of65 errors becomes native-optimal. Zero source-value regressions.

| Nonlinear check | Measured | Limit |
| --- | ---: | ---: |
| FIT margin RMS |0.001524574|0.002|
| Mean KL |1.5641e-7|1e-4|
| Max KL |3.8466e-6|1e-3|
| Relative head drift |0.000864903|0.01|

Fixed-error loss decreases2.655644490780 ->2.655644423889. This small decrease
is not the main result; an actual useful sampled decision changes while the
native floors remain intact. Probability mass on the optimal set increases
on23 errors and decreases on40; do not describe this as uniform improvement.
MONITOR corrects0/24 sampled errors. No2024 inference or test data was used.

All1027 state tensors are checked byte-identical after restoring the temporary
step. No candidate checkpoint is exported, no source model is replaced, and
no live authority changes. No chronological account was evaluated in304, so
TRAIN/VAL returns remain the preserved299 values, not newly measured returns.

Diagnostic identity:
`11d2060235f1d845d7d417328b9003166dd2f755fc76f18971c325958fbae02f`.
Nonlinear probe identity:
`365110bd3123bf56c1ea45062ab95e6eb42e26ff3699e9cd3ddd62407d1ad5d1`.
Evidence: [summary](generalization/selected71_boundary_20261112/SUMMARY.json),
[local witnesses](generalization/selected71_boundary_20261112/ROWS.json),
[nonlinear probe](generalization/selected71_boundary_20261112/NONLINEAR.json).

## Exact Next Action

Evaluate this fixed FIT-derived direction through the native full chronological
TRAIN/VAL accounts, with unchanged strategy, costs, activity rules and source
control. Verify cold reload and attribution of every changed order. Do not
optimize the direction using validation or promote it from the sampled panel.
Only after an account-qualified refinement should another cumulative phase2
outward/return pass run. Retsu's source qualification remains a separate lane.

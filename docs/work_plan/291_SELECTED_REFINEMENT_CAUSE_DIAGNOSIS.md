# Selected Refinement Cause Diagnosis

Owner-approved2026-11-11; follows290. Preserve frozen69-input export and all
refinement checkpoints. Diagnose, do not fit or promote another policy.

## Frozen Questions And Tests

| Question | Observable / acceptance gate |
|---|---|
| Does the critic rank executable actions correctly? | Native own-policy continuation returns on the151 authenticated TRAIN anchors; split FIT/monitor and state; report optimal action, NOP-relative gaps and regret |
| Does a correct local preference survive the shared head? | Analytic categorical gradient versus autograd; same-row versus cross-row Jacobian projections; include duplicate batch multiplicity |
| Is AdamW different from raw gradient pressure? | Decompose first-step adaptive preconditioning into self/cross contributions using the same diagonal preconditioner; decay separate; compare saved checkpoint1 |
| Does the first-step diagnosis describe the actual endpoint? | Project saved checkpoint1/8 parameter deltas through baseline Jacobian; compare actual logits/probabilities; disclose nonlinearity residual |
| Could target error explain transfer? | Diagnostic native-FIT return replacement and zero-entropy controls, no optimizer run, no learned candidate |
| What happened on the known missed validation entries? | Read-only posthoc Q/logits/Jacobian probes at the two previously attributed timestamps; never in gradient batch or targets |
| Are sources preserved? | Verify sealed source/export receipts and source hashes before and after; no TEST, live changes or trained checkpoint |

CPU execution is sufficient. Use saved production Q and recorded first batch;
verify CPU Q against it within1e-5. First-step replay tolerance: max parameter
error3e-6, logits2e-5 and identical probe actions. This is a cross-device check,
not a claim of bitwise GPU replay. Exact projection identities use double
precision and analytic/autograd agreement is tested separately.

Unit controls: independent rows produce no cross influence; shared rows can
reverse an otherwise correct action; applying the same AdamW preconditioner to
the self and cross terms must reconstruct the full adaptive step; positive
native actions need not be the optimal action. Count row/action pairs separately
from trades. Local logits and Jacobians are infinitesimal diagnostic controls,
not achievable account gains or independent trainable models.

Only FIT rows contribute gradients. MONITOR2023 and two posthoc2024 probes are
observations only. The native-FIT diagnostic is an oracle on historical TRAIN,
not a deployable input. First-step decomposition alone cannot prove every
intermediate update has the same cause; endpoint residuals bound this limitation.

## Completed Evidence

Diagnostic identity:
`98ede2e1175ac62698eca86c4636232552d87f6d633569226522e0a04252f662`.
CPU runtime2.94seconds.24 focused tests and scoped lint pass. Independent NumPy
audit recomputes986 identities/counts, with maximum additive discrepancy
2.22e-16. The two first-step in-memory replays match production checkpoints
within7.45e-9 in parameters and1.19e-7 in logits, with identical probe actions.
All source receipts and weights remain unchanged. No training candidate,
account resimulation, feature change or live promotion was produced.

The initial preflight incorrectly expected an explicit identity key in the
existing report CONFIG; that schema derives identity by digest. It stopped
before Jacobian analysis. The corrected reader verifies that digest and the
source receipt hash. Preserve that failed preflight separately. An initial
successful diagnostic was subsequently extended with per-layer attribution;
the final result above has its own identity and output directory.

## Two Measured Causes

There are82 executable flat-state FIT anchors (81 unique native optima, one
tie),28 executable monitor anchors, and41 anchors in other states. Native
optimum means the best of three current-policy continuations, not a globally
optimal market policy. All action arrays are ordered NOP, BUY, SELL.

| Critic seed | Optimal FIT decisions | Optimal monitor decisions | FIT missed positive direction-vs-NOP gaps |
|---|---:|---:|---:|
|19|45/82|9/28|33/79|
|43|36/82|6/28|46/79|

The last column counts action comparisons, not trades. The native-optimal
flat FIT action counts are9 NOP,35 BUY,38 SELL. Critic19 chooses34/22/26;
critic43 chooses40/38/4. Thus both overprefer NOP on this panel, and43 severely
underselects SELL. Lower value-fit loss is not enough to establish correct
action ranking.

Even removing critic error on FIT does not remove shared-head interference.
The following counts show a decrease in probability of the native-optimal
action, at the declared1e-10 threshold, among82 executable FIT anchors:

| Target/control | Seed | Independent local logits | Own-row head gradient | Full-batch head gradient | First AdamW linear projection |
|---|---:|---:|---:|---:|---:|
|Calibrated critic|19|28|36|52|47|
|Calibrated critic|43|35|43|62|59|
|Native FIT returns|19|2|12|70|63|
|Native FIT returns|43|2|12|68|60|

The independent native-return direction increases77/82 optimal probabilities,
reduces2, and has3 negligible changes. With shared parameters, gradients from
other observations can outweigh that useful local pressure. This is a real
tradeoff of the aggregate objective, not a reversed optimizer sign. The native
control changes FIT targets only; monitor rows and validation probes contribute
zero gradient. No oracle information was supplied as a network input.

Zero-entropy control does not remove this pattern: critic/AdamW lowers51/82
and59/82 FIT optimal probabilities, versus47/82 and59/82 with native entropy.
Neither disabling entropy nor calibrating Q alone is justified as a fix here.
The controls are derivative diagnostics, not new trained-model trading results.

## Why The Two Validation Entries Disappeared

The actor, not the critic, chooses the inference action. The validation rows
never entered an update. Consequently their Q values are diagnostic only:
the immediate execution cause is the actor's changed SELL-minus-NOP logit
margin, transferred from FIT updates through shared parameters.

| Previously attributed entry, UTC | Seed | Initial SELL-NOP | After8 updates | Action effect |
|---|---:|---:|---:|---|
|2024-06-04 17:00|19|+0.00132406|-0.00116456|SELL becomes NOP|
|2024-06-04 17:00|43|+0.00132406|-0.00261226|SELL becomes NOP|
|2024-11-14 17:00|19|+0.00797641|+0.00220382|SELL remains SELL|
|2024-11-14 17:00|43|+0.00797641|-0.00134033|SELL becomes NOP|

The first head layer's weight matrix contributes71.35%,79.83%,83.50%,88.36%
of the respective summed linear margin changes. All four tensors contribute
in the same downward direction on these probes. These are weight-space
attributions, not percentages of monetary loss and not an ablation proving
that freezing a layer improves generalization. Endpoint8 first-order residual
over all153 probes is at most6.92e-5; at the two missed-entry probes it is
below1.71e-6. The actual endpoint decisions, not the linear approximation,
determine the action-switch conclusion.

The full-account effects are already reconciled in290: -61.101210USD for19,
-107.407005USD for43. No new trading gain is claimed. The preserved research
baseline remains TRAIN13.913055%, VALIDATION16.336139%,739/207 orders.

## Exact Next Contrast

Do not repeat283/284's generic last-layer/projection comparison as if it had
never been run. The new evidence concerns the selected composition and its
fresh calibrated critics, and confirms both causes on that source.

Next design a FIT-only decision-preservation constraint, instead of accepting
updates solely because mean loss and aggregate drift improve. Protect source
actions that are strictly best among their measured FIT continuation returns;
require their margins versus alternatives not to deteriorate beyond a declared
numeric tolerance. First test feasibility and nonzero movement in weight space;
an infeasible or zero step must be recorded, never labeled successful learning.
Compare against the saved290 update without refitting critics or changing data.
Only after that control is qualified should a small bounded candidate undergo
full native accounts and activity checks. Such a constraint protects the sampled
FIT decisions only; it cannot guarantee validation profit or unseen decisions.
The two validation timestamps above must not become constraints or targets.

This next contrast is designed, not launched by this diagnostic. Preserve the
frozen export and all previous checkpoints. Phase2's next floating pass should
use an actually changed, qualified actor; do not repeat the unchanged pass.

Evidence: [summary](generalization/selected_cause_20261111/SUMMARY.json),
[independent audit](generalization/selected_cause_20261111/AUDIT.json),
[probability-pressure plot](generalization/selected_cause_20261111/PRESSURE.png),
[hash manifest](generalization/selected_cause_20261111/MANIFEST.json).

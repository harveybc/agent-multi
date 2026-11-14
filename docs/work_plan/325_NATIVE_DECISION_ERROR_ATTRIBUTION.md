# Frozen Native Decision Error Attribution

Owner authorizes the next diagnostic after324. No training, loss changes,
threshold tuning, model promotion, new account simulation or feature selection.
Reuse the four322 base controls and eight324 frozen endpoints. Preserve299,
71 inputs,49 return candidates and paused STL; DEAP remains after phase2.

## Questions And Fixed Scope

1. Which exact winner switches explain the change in native regret? Reconcile
   per-row deltas with the reported aggregate, split into helpful/harmful/
   neutral and by native position/legality. These are isolated same-state
   action consequences, not additive contributions to a complete new account.
2. Does the improvement in value/gap RMSE occur on the same rows as worsening
   decisions? Decompose MSE into common-level and centered-action components.
   Record small-margin crossings, target-optimal sets and action confusions.
3. Are errors concentrated at small true action gaps? Define low/mid/high
   gap bins from the original FIT-only positive distinct-optimum gaps, then
   apply unchanged to all scopes. Report ties separately; never treat a switch
   between equally good actions as harmful. Numerical tie tolerance is1e-8.
4. How much does twin-min aggregation contribute? Report both individual
   critics, their mean and minimum, action disagreement and cases where the
   minimum winner is neither twin's winner. These are explanatory views,
   not competing deployed policies or a replacement chosen on audit results.
5. Link functional changes to the hidden and output layers with the exact
   decomposition: deltaQ=(h_new-h_old)W_old^T + h_new(W_new-W_old)^T
   +(b_new-b_old). This is one declared algebraic path, not unique causal
   credit assignment. No weight transplant or parameter adjustment here.

Scopes: original FIT, expanded FIT, original stop and original audit, separately.
Audit strata are descriptive; any subsequent loss design must use FIT only.
Replay original stored metrics on their native GPU/batch geometry before
analysis; authenticate all seals and source identity. No2024/TEST inference.

## Tests And Evidence

| Requirement | Evidence |
| --- | --- |
| Exact regret accounting | Synthetic helpful/harmful/tied switch tests and ledger reconciliation |
| Value error is not decision quality | Lower-RMSE/worse-action witness |
| Common shift invariance | Identical actions/regret after shared value-level shift |
| No audit-derived bins | FIT-only bin construction; unchanged when held targets change |
| Native ties/empty strata | Explicit tied-optimal sets, zero-regret equivalent switches, null empty bins |
| Layer attribution | Exact linear/SiLU decomposition and finite-input tests |
| Native replay | Twelve stored control/endpoint replays, no fitting and source-file parity |

Emit machine-readable summaries, per-row ledgers, functional layer diagnostics
and a compact report. Retain the source actor regardless of these diagnostic
results; select at most the next FIT-only contrast after inspecting evidence.

## Completed Results

Runtime `1cfde346`, identity
`c9f949404158210905ed28c925d2571871b61786878ee0670c4b7af9cc1ebaee`.
Completed in2.30s, optimizer steps0.17 focused tests pass. All12 native metric
replays match exactly. Independent CPU verification checked3804 row decisions,
32 scope summaries, FIT-only bin thresholds and the saved output arrays.
Maximum regret-accounting residual is6.94e-17. The declared per-twin layer
decomposition reconstructs output differences within8.74e-7 in native float32.
Source299 and all71 inputs are unchanged; no model or critic is promoted.

### Winner Changes At Equal Updates

| Fold/seed | Helpful | Harmful | Neutral switches | Audit regret delta |
| --- | ---: | ---: | ---: | ---: |
| A/19 |14|8|4|-0.029195|
| A/43 |15|13|3|+0.006555|
| B/19 |14|19|4|+0.019110|
| B/43 |11|21|5|+0.053946|

These are238 paired row evaluations over119 distinct audit anchors, not238
independent trades. Total131 winner switches:54 helpful,61 harmful,16 neutral.
47/61 harmful switches occur with both openings allowed; seven each occur in
long and short positions. Harmful changes include both directions and NOP,
not one universal BUY-first or never-SELL pattern.24/61 harmful switches have
lower row-level value MSE. Their new predicted margins range0.004289-0.707971;
none is within the1e-6 numeric near-tie diagnostic. Eighteen occur in the low
true-gap bin,34 in the middle and9 in the high bin. Tiny gaps alone do not
explain the failure. Gap thresholds were frozen using original FIT labels.

### Coverage Trades Off Previously Fitted Decisions

Equal-weight mean regret across four fold/seed pairs:

| Scope | Original control |4x FIT,8192 updates |4x FIT,32768 updates |
| --- | ---: | ---: | ---: |
| Original FIT |0.008043|0.099032|0.031218|
| Expanded FIT |0.232719|0.123486|0.035423|
| Stop |0.272516|0.265748|0.315583|
| Audit |0.236451|0.249055|0.253084|

At8192 updates the original FIT regret worsens in all four pairs while the
expanded FIT regret improves in all four. Both fits started from the same
initialization: this is a coverage/exposure tradeoff, not proof of catastrophic
forgetting during a warm start. More exposure reduces but does not remove the
original-FIT disadvantage on average, and does not solve audit performance.

83.53% of the mean audit MSE reduction at8192 comes from the common-value
component; centered-action MSE also improves. Removing a common level from
the already aggregated three-action vector cannot change its argmax. Lower
centered/gap error still does not ensure correctly signed decisive differences.
These exact accounting statements identify an objective/decision mismatch,
not a demonstrated fix or a causal attribution to one optimizer setting.

### Twin Aggregation And Internal Layers

Replacing the minimum with the mean is not a universal repair: at8192, the
mean's audit regret worsens versus its own control in all four pairs. Both
individual critics worsen on average as well. Five paired rows choose a
minimum winner preferred by neither individual critic. At32768 the mean's
average delta is slightly favorable (-0.002190), but this is a separate
descriptive view, not authorization to select that rule on inspected audit.

The declared output-change decomposition at8192 has mean per-arm audit RMS
0.928295 from the hidden-layer term and0.167143 from the output-layer term.
This hidden layer is the8-unit layer INSIDE THE CRITIC HEAD, not a TCN or a
feature extractor. The encoders never changed. These terms can cancel and
depend on the chosen algebraic path; their magnitudes do not prove that
freezing a particular layer will improve trading.

## Next Bounded Contrast

Use saved weights for a paired component-recombination diagnostic before
another training run: control hidden layer + expanded output layer, and
expanded hidden layer + control output layer, versus the two existing complete
heads. Change both twins' matching components together, keep identities and
same rows, and inspect original/expanded FIT decision preservation first.
Stop/audit remain descriptive; do not promote recombined weights from them.
This tests whether the apparent hidden-layer drift is functionally reversible
or reflects coadaptation. If justified by FIT, follow with a matched warm-start
output-only versus full-head calibration, using the same starting weights.
Neither successor is running yet. Preserve299 and do not repeat unchanged
floating selection, resume STL, or start DEAP.

[Full endpoint tables and machine-readable summaries](generalization/native_decision_error_20261113/RESULTS.md).
Detailed per-row ledger and raw output arrays remain in the sealed external
325 collection; the independent receipt authenticates their reconstruction.

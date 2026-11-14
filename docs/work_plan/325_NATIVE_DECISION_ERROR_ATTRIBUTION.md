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

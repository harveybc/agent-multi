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

# 267. Selected69 refinement decision audit results

Completed2026-11-10. Design: [266](266_REFINEMENT_DECISION_AUDIT.md).
All three workers finished:18 full accounts,779 changed-signal occurrences,
87 worker receipt files independently verified. Automatic collection and
aggregation completed.15 focused native tests passed. No optimizer updates,
source checkpoint changes, feature changes or model promotion.

The output filesystem gate initially refused one worker for insufficient
reserved space. Its empty output directory was relocated to a filesystem with
sufficient capacity; the original reserve was retained and the gate passed.
No datasets or previous results were deleted. Worker elapsed times were
196.87,91.09 and280.98 seconds for SmoothL1,M1 andM0.5 respectively.

## Account results

Means across three actor seeds sharing the representation and calibrated
critic for each objective. These are not independent market replications.
Returns are percentages, not Sharpe. The source is the unchanged69-input model.

| Model | TRAIN return % | VALIDATION return % | VALIDATION Sharpe |
|---|---:|---:|---:|
| Preserved source |10.7921|14.5911|2.1656|
| Bayesian, SmoothL1 critic |10.2503|12.4337|1.7723|
| Bayesian, mean squared + variance |11.4976|11.7956|1.7561|
| Bayesian, mean squared +0.5 variance |11.1007|12.3687|1.8306|

Each account's chronological decision contributions reconcile to its entire
net-profit difference with error0.0 USD. Individual isolated effects are NOT
added to claim an account effect. All neutral changes are retained.

| Period | Changed signals | Helpful | Harmful | Neutral |
|---|---:|---:|---:|---:|
| TRAIN |499|72|60|367|
| VALIDATION |280|31|58|191|

Counts repeat events across objectives/seeds. The60 harmful TRAIN occurrences
represent15 distinct timestamp/old-action/new-action tuples;58 harmful
VALIDATION occurrences represent12. They are not60/58 independent trades.

## What explains the regression

1. All58 harmful validation occurrences are added entries. TRAIN has50 added
   entries,7 changed entry directions and3 suppressed entries. All60 harmful
   TRAIN changes occur while flat. No added early close explains this
   particular regression; this does not establish global close quality.
2. The critic's action ordering conflicts with all three same-state stochastic
   soft-return continuations in20/60 harmful TRAIN occurrences and36/58 in
   validation. The TRAIN subset accounts for56.53% of gross harmful
   contributions, but not a majority of harmful occurrences. Three draws
   are diagnostic, not an exact expectation or significance test.
3. In21/60 harmful TRAIN changes, the actor changes against the raw Q ranking.
   The entropy-adjusted local score favors39/60 harmful changes. Shared
   parameter updates, entropy and KL prevent interpreting every argmax change
   as a greedy Q decision. These categories overlap.
4. For47/60 harmful TRAIN changes, the stochastic calibration trajectory held
   a position while the audited greedy trajectory was flat at the same market
   observation. This is an observed state-distribution mismatch, not proof
   that position ambiguity causes every error. Matching direction alone also
   does not match the complete account state. No state input was added.
5. Weight-space decomposition reconstructs every changed pairwise log-odds
   with maximum residual7.00e-7. Mean absolute contributions: weights0.0296623,
   biases0.0106423, posterior averaging2.58e-7. These are algebraic boundary
   contributions, not percentages of profit or a global causal importance.
   The immediate decision drift is overwhelmingly in the posterior means,
   not Monte Carlo probability averaging.

## Concrete TRAIN witnesses

Even within the matching flat-position subset, SmoothL1 favors the new action
on2022-03-14T15:00Z (Q gap+0.00245), while all three soft continuation gaps
are negative (-0.5250,-0.5300,-0.5227). On2023-07-11T16:00Z the Q gap is
+0.03724 versus(-0.1073,-0.7269,-0.0280). Conversely2021-02-19T09:00Z has
negative Q gap(-0.00438) and negative gaps on all three continuations, yet the
actor still changes and harms the account. Therefore one explanation alone
does not cover these observed failures.

## Next bounded step

The frozen TRAIN-only decision rule returns MIXED_MECHANISMS. Do not replace
it with the stronger-looking validation diagnosis. Preserve the69 champion.
Before another actor update, construct a paired TRAIN-only opening diagnostic:

- Reproduce the witnesses and include unchanged/helpful opening controls;
  inspect full account-state identity, not just direction.
- Compare Q ordering under the actual flat decision state with its stochastic
  calibration state, keeping continuation policy/entropy semantics matched.
- Separate critic misranking from changes against Q with the saved actor
  gradients and weight/bias updates; no new learning or validation-fitted gate.
- Require a demonstrated repair of these witnesses AND the broader control
  panel before another refinement, then evaluate complete accounts/activity.

This differs from repeating the old70-input paired-advantage or generic
LR/KL experiments. Reports239 and245 were reviewed: reduced drift and better
gap magnitude had not consistently repaired decisions. Do not repeat floating
selection on the unchanged69-input evaluator merely because compute is free.

## Evidence

Compact authenticated [summary](generalization/decision_audit_20261110/SUMMARY.json)
and [collection receipt](generalization/decision_audit_20261110/COLLECTED.json).
Full per-event boundary terms, states and continuations remain in the local
collection `/var/tmp/decision69-audit-20261110/`; they are not all copied into
this compact public packet. The existing source model retains its validated
return14.5911%,206 orders,58.2524% wins and maximum drawdown4.0447%.

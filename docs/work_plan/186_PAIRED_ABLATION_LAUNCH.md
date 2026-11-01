# Paired ablation launch

Started 2026-11-01 10:39:40 UTC. Native runtime `45ea82dd`; feature_selector
plugin `00ec628`, wheel 0.2.2. Campaign identity:
`815ddbeed8242eb24e234c154f9e9dee658874b0a751163b31120be7632d0757`.

## Verified before launch

- Feature-selector tests: 47 pass, four legacy tests skipped; installed-wheel
  demo and identical-config resume pass.
- Native tests: 33 pass; scoped Ruff passes. Same-support structural pruning,
  actual actor/critic dimensions, frozen stopping, prefix-budget enforcement,
  smoke rejection and report pairing are covered.
- Final CPU smoke: 128 decisions, 29 updates, exact reload, full TRAIN and
  INNER_A accounts, 19 authenticated artifacts. Maximum cash discrepancy
  2.30e-11 USD. Engineering evidence, not a candidate score.
- Both assigned GPU hosts pass the existing launch/readiness probe, including
  framework visibility and disk reservation.

## Unattended sequence

One seed stays on its original control GPU throughout all subsets. Each
worker replays its authenticated full-model prefix, then retrains the single
return deletion, cyclic-pair deletion and whole-wavelet-branch deletion.
The primary worker waits for the existing stopping contrast to finish; the
secondary has already replayed its control and started `return_single__19`.
Observed at 10:40 UTC: 4864 decisions and 193 gradient updates, no restart.

Two persistent sequential queues and one persistent CPU collector are enabled.
Workers have 6 GiB memory limits and resource admission, at most two fresh
attempts per unit, and retained failures. The collector verifies artifact
digests and replays selection decisions from the histories before reporting.
It writes `AGGREGATE.json`, `RESULTS.md` and final `CLOSE.json` automatically.
No approval is required between these units. Mid-fit resumption is not claimed.

Estimated closure: 18-22 hours after launch, based on the previous measured
per-host learning rates and the primary predecessor. This is an estimate,
not a measured completion time for the new subsets.

## First reference replay

Seed19's full-control replay selected step327680 for TRAIN-best and step163840
for protected combined scoring. Their INNER_A returns are -16.7046% and
-3.8697%, with Sharpes -3.6956 and -0.8860 and 371/318 trades respectively.
These repeat existing control evidence; no ablation has completed. Do not
infer which evaluator is best for feature selection from this reference alone.

The final comparison uses paired monitor deltas, changes in signals on the
same timestamps, seed agreement and complete trading metrics. It does not
automatically promote a feature list or start the full floating search.

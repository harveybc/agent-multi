# Partial Critic Results

## Verified Completion

Design250 completed2026-11-10T08:07:18.423841Z (03:07 Bogota) in191.26 seconds.
Runtime `0a487b30`. Two new arms,4096 updates each, and66 component-restoration
interventions. Four archived controls reused with authenticated settings,
initial metrics, native labels and identical per-seed minibatch schedules.

Worker and collector exited0. All17 new collection files verified, identity
`af75f09efae8f60f3195687a3d1c88d4c8a4a41c9152117052302a628fcdd2f0`.
Actor full-TRAIN logits, frozen parameters and original69/70 checkpoint hashes
remain exact. No strategy change, actor update, model promotion or new trading
return claim. The queue is complete.

Evidence: `/var/tmp/sac70-partial-critic-20261110`.
Public [comparison curves](generalization/partial_critic_20261110/CURVES.csv),
[plot](generalization/partial_critic_20261110/FIT_MONITOR.png), and
[restoration effects](generalization/partial_critic_20261110/RESTORATIONS.csv).
SOURCE_DONE authenticates the full source collection, not just this subset.

## Matched Fixed Endpoint

All metrics below use the minimum of the twins. Pair accuracy is correct
ordering of informative action returns, not trade win rate. Fit1075 states;
monitor351 states. No checkpoint selected using monitor performance.

| Seed | Trainable scope | Fit pair accuracy | Monitor pair accuracy | Monitor gap RMSE | Monitor mean regret |
|---|---|---:|---:|---:|---:|
|19|Heads (archived)|54.66%|54.70%|0.378132|0.192022|
|19|Last core + heads (new)|55.13%|52.48%|0.379088|0.193103|
|19|Full critic representation (archived)|66.51%|51.04%|0.394906|0.185488|
|43|Heads (archived)|54.92%|54.44%|0.378243|0.191440|
|43|Last core + heads (new)|55.26%|51.57%|0.379403|0.193726|
|43|Full critic representation (archived)|67.02%|51.31%|0.394975|0.188401|

Gap errors and regrets use discounted-reward units. Last-core fitting does
not beat heads-only on aggregate monitor ordering, gap RMSE or mean regret
in either seed. It reduces the gap-error deterioration of full unfreezing,
but loses most of the latter's fitting improvement. Full unfreezing still has
slightly lower aggregate regret despite worse ordering: preserve that mixed
result instead of substituting a single preferred metric.

Open-long monitor accuracy for the new scope is46.03/46.83%, between full
unfreezing (38.89/40.48%) and heads-only (51.59/50.79%). Its open-long gap RMSE
is0.294009/0.294025, close to heads-only0.292673/0.292617.

## What Weight Interventions Showed

The33 components are28 individual feature branches, fusion, attention, two
core blocks and head. Each intervention restores only one component from the
initial calibrated critic into the full-unfreezing endpoint; parameters return
exactly to the endpoint before the next intervention. Endpoint metrics replayed
within the declared1e-6 tolerance before interventions.

Largest relative parameter drift in both seeds: head about38%, fusion about35%,
last core/branch20 about27%. Parameter drift is not importance or causal harm.

Restoring fusion gives the largest reduction of open-long monitor gap RMSE:

| Seed | Full endpoint | Fusion restored | Reduction | Open-long ordering before/after |
|---|---:|---:|---:|---:|
|19|0.357490|0.295069|17.46%|38.89 /50.00%|
|43|0.359159|0.295551|17.71%|40.48 /50.79%|

But this is NOT a global repair. Aggregate monitor ordering falls from
51.04/51.31% to46.34/46.21%, and mean regret rises from0.185488/0.188401 to
0.226287/0.227798. Its improved open-long behavior is offset by other decisions.
Restoration effects are conditional on the other trained components; they do
not establish independent additive causes or justify adopting a hybrid model.

## Disposition And Next Diagnosis

Do not promote last-core or restored-component checkpoints. Keep all source
champions and these diagnostic weights. No new feature pass is justified by
this contrast alone.

Before another unfreezing sweep, quantify action-value compatibility at the
same market observation under different feasible native account states. The
market-only model emits one Q vector, while NOP/BUY/SELL have different effects
for flat, long and short accounts. Replaying valid alternative histories can
measure the resulting target differences without adding state observations or
assuming that such differences explain all errors. This is the next proposed
diagnosis, not a launched experiment or an established root cause.

As in248/249, monitor2023 was held out of these gradients but was used in source
pretraining. Quarterly labels overlap within roles. No2024 trading evaluation
or TEST scoring occurred. Results concern critic calibration, not learned actor
trading profitability or an independent final generalization estimate.

# Action-Advantage Calibration Results

## Verified Completion

Design242 completed2026-11-09 at19:39 UTC (14:39 Bogota),456.79 seconds.
Runtime75a78cc4; four sequential arms,4096 critic and256 anchored actor
updates each. Both services exited0 and the automatic collector authenticated
125 files. Identity:
`629d04144e2cdbbea0942f5f04aab43eddf5ebb8887b5001225204855aef6fe1`.
Both absolute controls reproduce the archived calibrated parameters exactly
and their32/256-update native accounts match the preceding KL contrast.
Original69/70 checkpoint hashes remain unchanged; no model promoted.

Full results, trading CSV and return PNG:
`/var/tmp/sac70-advantage-calibration-20261109`.
Compact evidence accompanies this document in
`generalization/advantage_calibration_20261109/`; its SOURCE_DONE receipt
authenticates the full source collection, not just the public subset.

## Fixed256-Update Trading Endpoint

Returns are net account percentages. VALIDATION metrics occupy all columns
except TRAIN. The parent is unchanged, not trained again.

| Policy | TRAIN % | VALIDATION % | Sharpe | DD % | Trades | Win % | Orders/week |
|---|---:|---:|---:|---:|---:|---:|---:|
| Parent70 |11.5145|13.8085|2.0971|3.6249|206|57.77|3.8148|
| r19 absolute |12.1507|14.0574|2.0895|3.6249|207|57.97|3.8333|
| r19 advantage |12.4067|14.0574|2.0895|3.6249|207|57.97|3.8333|
| r43 absolute |12.2751|12.8394|1.9039|4.0446|205|57.56|3.7963|
| r43 advantage |13.2474|12.9580|1.9444|4.0447|203|57.64|3.7593|

New minus control VALIDATION:0.0000pp and+0.11854745pp. Mean paired effect
is+0.059273725pp, descriptive across two replicas, not a significance claim.
TRAIN improves by0.25602565pp and0.97235910pp. The new loss does not beat
the unchanged parent consistently. At32 actor updates the new loss is worse
in both replicas (13.2858 vs14.0574;13.0648 vs13.5773). Do not choose a
different endpoint retrospectively to strengthen the claim.

## Critic Ordering

Metrics below use held-out TRAIN anchors,95 per replica, and greedy parent
continuations. Anchors are excluded from fitting, but their future trajectories
overlap with fit anchors: these are not independent market holdouts.

| Replica | Arm | Pairwise accuracy % | Mean regret, reward units | Return RMSE |
|---|---|---:|---:|---:|
| r19 | absolute |54.0816|0.167289|0.570760|
| r19 | advantage |52.0408|0.167056|0.570985|
| r43 | absolute |55.2239|0.172088|0.571132|
| r43 | advantage |56.7164|0.159435|0.571726|

Fit-anchor ordering improves in both replicas (61.6883 to64.1234 percent;
54.0134 to59.6990). Monitor ordering does not. Regret improves slightly in
r19 and more in r43, while absolute-return RMSE increases slightly in both.
This confirms the loss changes its intended emphasis; it does not establish
that the earlier harmful ordering cases were corrected.

The r43 monitor critic prefers BUY on82/95 anchors versus87/95 for absolute.
This is critic argmax, not executed trades: the anchored actor still executes
both directions, with179 SELL and24 BUY trades. Keep these layers distinct.

## Direction And Closing Checks

| Policy | Buy trades / wins | Sell trades / wins | Early closes / harmful |
|---|---:|---:|---:|
| Parent70 |23 /15|183 /104|6 /2|
| r19 absolute |23 /15|184 /105|6 /2|
| r19 advantage |23 /15|184 /105|6 /2|
| r43 absolute |24 /14|181 /104|7 /3|
| r43 advantage |24 /14|179 /103|7 /3|

Early-close harmfulness compares the fixed position with passive protection,
not the entire subsequent account. The new loss has not reduced this count.
All endpoint accounts satisfy the activity rule; none is a zero-order result.

## Next Diagnostic

Retain the new loss as a tested candidate, not a universal replacement.
Compare same-state action changes between each new endpoint and its absolute
control, using stored weights and exact native continuations. Separate useful
critic corrections, harmful corrections and shared-actor movement against Q.
This is the next bounded diagnostic before another training change or feature
removal pass. No new training or diagnostic is launched by this report.

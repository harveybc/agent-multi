# Paired Advantage Attribution Results

## Execution

Completed 2026-11-09T21:58:01.344692Z (16:58 Bogota), 40.00 seconds.
Runtime `50081347`; 12 focused tests passed; GPU readiness gate passed.
Both worker and automatic collector exited0. Eleven terminal files verified.
Identity: `89bc8a1f444bf3b050b81e0220b94df3ac7d9bb4b0cba1de9294c6832a46e3fe`.

Replayed both fixed256 endpoints, their calibrated critics and original parent
signals exactly. No optimization occurred. All original69/70 and four endpoint
weights remain unchanged. Forty-three changed decisions required258 native
suffix simulations. Four account differences reconcile with error0.0 USD.

Full collection: `/var/tmp/sac70-paired-advantage-attribution-20261109`.
Compact public evidence: [events](generalization/paired_advantage_attribution_20261109/EVENTS.csv)
and [result](generalization/paired_advantage_attribution_20261109/RESULT.json).
SOURCE_DONE authenticates the full collection; the public subset is not a
complete copy of that collection.

## Account Attribution

New advantage endpoint minus same-replica absolute control. Contributions use
chronological interventions and full account continuation, including later
opportunities and sizing. They are not isolated trade profits or Shapley values.

| Replica | Period | Changed decisions | Helpful | Harmful | Neutral | Delta USD |
|---|---|---:|---:|---:|---:|---:|
| r19 | TRAIN |13|1|1|11|+25.602565|
| r19 | VALIDATION |3|0|0|3|0.000000|
| r43 | TRAIN |18|3|2|13|+97.235910|
| r43 | VALIDATION |9|1|1|7|+11.854745|

Only9 of43 changes affect account cash; all nine are entry decisions from a
flat account. No economically effective close-command change explains the
new-minus-control result. Position ambiguity while already holding an order
is therefore not the explanation for these nine events.

## The Two Effective Validation Events

| UTC | Action change | Cash contribution USD | Absolute Q gap | Advantage Q gap | Matched-policy realized gap |
|---|---|---:|---:|---:|---:|
|2024-02-15 15:00|SELL to NOP|-44.610795|+0.029120|+0.046715|-0.400276|
|2024-08-13 13:00|SELL to NOP|+56.465540|-0.032366|-0.011574|+0.550976|

Q gaps and realized gaps are new-minus-old in discounted reward units, not
dollars. The realized diagnostic uses the unchanged parent70 continuation
after forcing the first action in the SAME control-account state. It is kept
separate from the absolute-control continuation used for account attribution.
These states can differ from the original calibration anchors.

Both critics have the wrong action ordering in both validation events.
On February15 the new critic reinforces the wrong preference for NOP and
the actor removes a useful SELL entry. On August13 the new critic still
prefers SELL, albeit less strongly; the actor changes to NOP anyway and the
account improves. This is an observed mismatch between greedy actor choice
and Q preference, not proof of an optimizer bug: updates are shared across
states and include entropy and KL terms.

## What Changed And What Did Not

- Across all9 effective events, the sign of the compared Q gap never changes
  between calibrations. No previously wrong sign becomes correct in this set.
- Gap magnitude error improves in5 events and worsens in4. Better magnitude
  calibration does not guarantee better discrete decisions.
- Both calibrations conflict with matched-policy realized ordering in4 of9
  effective events. Two actor changes go against the new critic: one harmful
  TRAIN entry and the useful validation abstention above.
- The same net validation improvement of0.11854745pp is explained exactly,
  but it is concentrated in two events, not broad evidence of ranking repair.
- These are descriptive event counts, not independent samples or a global
  calibration error rate. The unchanged sources remain available; no promotion.

## Next Focus

Keep the alternative loss as evidence, not a mandatory default. Diagnose the
TRAIN-only critic fitting curve, coverage and error by action/position before
increasing training or adding another regularizer. The current label panel
contains284 fit anchors per replica, with95 monitor anchors and overlapping
continuations, so it cannot establish broad action-ranking reliability.
Separate poor fit, inadequate coverage and actor transfer before another
feature-removal cycle. This report launches no new training.

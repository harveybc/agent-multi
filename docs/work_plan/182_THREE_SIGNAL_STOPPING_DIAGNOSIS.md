# Three-Signal Stopping Diagnosis and Phase-2 Handoff

Snapshot2026-11-01T03:18UTC (2026-10-31 evening in Bogota).
Read-only analysis of completed seeds7/19 from plan180. Seed43 is still
running under the unchanged campaign. No new training or selection launched.

## Completed Financial Accounts

| Seed | Stop decisions | Complete TRAIN episodes | Best checkpoint | Stop |
|---|---:|---:|---:|---|
| 7 | 688128 | 32 | 360448 | patience10 |
| 19 | 655360 | 30 | 327680 | patience10 |

| Seed | Period | Return % | Sharpe | Max DD % | Trades | Buy / sell | Win % |
|---|---|---:|---:|---:|---:|---:|---:|
| 7 | TRAIN2020-2023H1 | 1177.960 | 5.827 | 6.276 | 3857 | 1903 /1954 | 64.791 |
| 7 | INNER_A2023H2 | -16.843 | -3.535 | 17.432 | 441 | 208 /233 | 44.671 |
| 19 | TRAIN2020-2023H1 | 1741.510 | 7.067 | 9.833 | 3545 | 1860 /1685 | 68.152 |
| 19 | INNER_A2023H2 | -16.705 | -3.696 | 17.921 | 371 | 190 /181 | 46.631 |

These are complete account paths with costs, not averages of monitor blocks.
TRAIN and INNER_A have different durations; do not divide these total returns
to estimate a generalization ratio. Matched eight-week blocks below provide
the comparable-duration learning-curve diagnosis. Internal validation was used
repeatedly for selection and is development evidence, not untouched validation.

## Retrospective Patience Replay

Replayed the original chronological monitor scores with unchanged minimum
131072 decisions and min_delta1e-5. Only hypothetical patience changes. Each
candidate's recorded eligibility and combined TRAIN/internal score remain
unchanged. Evaluation is deterministic and isolated from the training RNG.

| Seed | Patience | Hypothetical stop | Selected checkpoint | Mean TRAIN return % | Mean internal return % |
|---|---:|---:|---:|---:|---:|
| 7 | 3 | 196608 | 98304 | -2.1883 | -0.0499 |
| 7 | 5 | 524288 | 360448 | 17.7407 | -4.6800 |
| 7 | 10 actual | 688128 | 360448 | 17.7407 | -4.6800 |
| 19 | 3 | 262144 | 163840 | 5.4162 | -1.8799 |
| 19 | 5 | 491520 | 327680 | 13.6626 | -5.2594 |
| 19 | 10 actual | 655360 | 327680 | 13.6626 | -5.2594 |

Returns are means of three eight-week blocks. Minimum duration controls when
stopping may occur; it does not forbid selecting an earlier checkpoint.
Patience5 would save163840 decisions per seed without changing the selected
weights. Patience3 would select different checkpoints with better internal
scores in these two histories, but this is a post-hoc diagnostic, not a new
financial evaluation or justification to declare3 optimal. Historical early
weights were not all retained, so their full-account outcomes are unknown.

The combined average permits TRAIN gains to compensate for worse internal
validation. Seed7's chosen360448 checkpoint has internal-4.68% versus-0.19%
at262144, while TRAIN rises from7.29% to17.74%. The same issue is visible in
seed19 between294912 and327680. Lowering patience alone does not fix this
checkpoint-ranking behavior.

The curves also are not monotonic: seed19's internal score improves again
to-1.1937% at589824, with170 trades across monitor blocks, after the selected
checkpoint's-5.2594% with355 trades. Exposure/activity therefore must be
reported alongside returns. Neither overtraining nor data/architecture
causes are established or excluded by this snapshot.

## Phase-2 Recommendation

The baseline is a useful candidate evaluator for floating feature selection:
the same learned model executes both directions, holds, closes and produces
paired native-account measurements. Profitability is not an entrance criterion
for testing whether removing inputs improves generalization. This does not
establish that redundant inputs caused the current gap or that selection will
remove it.

Next actions, before an expensive wrapper campaign:

1. Finish seed43 without changing its config; summarize all three trajectories.
2. Compare stopping/checkpoint rules in a bounded experiment. Retain the owner's
   TRAIN/internal average as control; test an explicit internal-performance
   non-degradation condition before accepting a TRAIN-led improvement. Freeze
   tolerances and activity requirements before the comparison. Save every
   evaluated checkpoint or a declared frontier, not only the latest winner.
3. Freeze that evaluator for phase2: identical chronological rows, starting
   weights protocol, seeds, budget, monitor, strategy and costs across subsets.
   Refit the actual modular TCN/discrete-SAC model for each subset. Remove a
   whole branch only when its last input disappears; no substitute MLP.
4. Judge feature changes on paired internal-period performance, uncertainty
   across seeds/blocks and activity, alongside TRAIN and the gap. A small gap
   obtained by degrading both periods is not an improvement. Do not optimize
   on2024 or TEST, change support rows between subsets, or call repeatedly used
   selection folds independent confirmation. Upstream phase1 already saw
   2023H2; retain that disclosed development scope.

This extends plans180/181, not a new authorization to deploy a strategy or
launch phase2. Existing automated work continues unchanged.

## Evidence Identity

Campaign:73892e698ce7a20c4a82d17eef047414ea199212860d38d03d9cdc5b7306e72d.

| Seed | Artifact | SHA256 |
|---|---|---|
| 7 | RESULT.json | 15f1725368053ad87580b2ef0bee5b8a3cc6168328c8d0ae0a67de744499c177 |
| 7 | HISTORY.json | 5b0b3b95ac5e0b64155d669fd2c0a3c75876976f81c5346bf8a5542b9c5b9c2e |
| 19 | RESULT.json | afc834b8b9d2499fcf1b1cbe8bd027eabcbf8abdca4ab926d025374400b4015f |
| 19 | HISTORY.json | 42ed7cdfb0071f00791d4943e12eba26d82f0d7615037556f0b4b543e07d033e |

Method reference: [sequential selection evaluates feature changes with an
estimator's validation score](https://scikit-learn.org/1.6/modules/feature_selection.html).
This supports the wrapper principle, not the financial validity of this
specific evaluator or any claim that sklearn implements our floating RL wrapper.

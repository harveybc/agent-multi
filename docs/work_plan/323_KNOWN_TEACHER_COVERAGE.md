# Known Teacher FIT Coverage Curve

Owner authorizes the next step after322. Weight-space analysis is not exhausted;
the evidence justifies isolating FIT coverage before more head surgery.

## Prespecified Contrast

Use source299,71 inputs,13 observations at stride2, frozen encoders and native
208->8->3 twin heads. Reuse the four authenticated balanced-teacher322 controls
at8192 updates. Do not retrain them. Two folds x two initialization seeds x
two expansions =8 new fits. A:49 ->196 ->784 FIT examples. B:98 ->392 ->1568.

The exact teacher from each original fold is frozen, including its original
FIT-only tertile thresholds. Increasing coverage must not refit those thresholds
or rebalance the task. Original239 cached embeddings and all old stop/audit
examples are retained byte-for-byte; new embeddings use the same frozen model.
Sample additions without labels, round-robin across available calendar quarters
inside the original FIT years, with deterministic within-quarter permutations.
Expanded sets are nested and retain every original FIT example. Exclude every
other original panel row, preserve feature-window year boundaries and TRAIN
membership, and never add a stop/audit row.

Same initial tensors as322, LR3e-5, WD0.01, AdamW, batch128, MSE and clip norm10.
Primary endpoint:8192 updates for every size. Secondary endpoint:32768 for4x
and131072 for16x, preserving nominal presentations per example. The secondary
endpoint also changes optimization time; report it separately, not as a pure
sample-count effect. No early-stopping checkpoint selection in this diagnostic:
fixed endpoints avoid the known patience/LR artifact. Plot FIT and unchanged
stop/audit value RMSE, action-gap RMSE and action regret. Record sample counts,
quarter/action distributions, actual optimizer parameters and cold reload.

Primary evidence of a consistent effect requires lower audit RMSE AND no worse
audit action regret in all four fold/seed pairs versus their corresponding
original controls. Report every cell regardless of that rule. This is known-
function calibration, not actual account performance or an actor promotion.

## Test Design

| Requirement | Test/evidence |
| --- | --- |
| Nested additional examples | Superset, exact sizes, deterministic ordering, balanced quarter sampling |
| No held-row admission | Explicit exclusion and temporal membership checks |
| Teacher unchanged | Frozen state digest; original target/metric replay exact |
| Original controls unchanged | Seals and complete endpoint metric replay, no control training |
| Same initialization | Original322 state digest equality per seed |
| Real additional inputs | Pinned store/support, finite native windows, no synthetic feature substitution |
| Fixed objective | Teacher targets only for this diagnostic; original trading labels untouched |
| Autonomous sequence | Immutable runtime, per-arm checkpoints/resume, exclusive lock, progress and seals |

The source's historical mixed-COT limitation is unchanged. No2024 inference,
TEST access, actor update, encoder update, STL or live deployment. Frozen
encoders have seen full TRAIN, so these are internal representation diagnostics.

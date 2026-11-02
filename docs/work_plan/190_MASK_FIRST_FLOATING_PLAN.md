# Mask-first floating selection before DEAP

Owner revised the sequence on 2026-11-02: stop new DEAP candidates, retain
existing artifacts and let the two already running fits finish independently.
DEAP hyperparameter optimization moves AFTER feature selection. Do not reduce
dates, temporal windows, costs or business controls to obtain a speedup.

1. Use the completed paired pilot's full-input TRAIN-best checkpoints (seeds
   7 and 19, same 524288-decision budget and checkpoint rule), not whichever
   single seed has a positive validation return. This is a frozen sensitivity
   baseline, not a DEAP champion. Freeze hyperparameters, strategy, feature order,
   scaler, groups, calendar, costs, support and seeds. No nested DEAP per subset.
2. Authenticate both pilot receipts and reproduce full-control actions and
   native accounts before masks. Preserve the requested prelaunch notice.
   The obsolete DEAP-closure timer is disabled; selection does not wait for it.
3. Use feature_selector's floating search with a separately identified frozen-
   checkpoint evaluator. Zero removed channels AFTER fixed TRAIN scaling over
   the entire window. Preserve original tensor width, branches, weights and
   input-validity mask; a removed channel does not restore missing dates.
4. Evaluate fresh native accounts on the FULL original TRAIN/INNER_A periods for
   each new mask. Reuse a baseline only when checkpoint, inputs, support, mask,
   strategy, costs and evaluation interval all match. Report paired deltas by
   seed and period, not only their average or a small generalization gap.
5. Recompute deletion candidates relative to the current subset and allow
   reinclusion. Keep coding blocks atomic. Use no expert labels or TEST data.
6. Confirm only a small set of distinct finalists with fresh structural fits:
   full control, best masked-validation subset, and a smaller competitive
   subset if present. Remove empty branches for these fits, with frozen
   hyperparameters and projected TRAIN scaler. Compare mask and refit rankings
   and report disagreement; a mask-only result is not structural confirmation.

## Test design and implementation boundary

| Requirement | Acceptance evidence |
|---|---|
| Champion preparation | Closure/spec/generation identity and both native artifact hashes; tampered weights rejected |
| No eligible champion | Explicit report; no synthetic winner and no phase-2 launch |
| Mask implementation | Full-mask parity; no weight/replay mutation; zero only selected channels after scaling; atomic-block enforcement |
| Paired comparisons | Identical rows/clock/costs; cache misses on any identity change; deterministic replay |
| Floating search | Remove/reinclude trace, evaluated subset cache, explicit finite budget and tolerance |
| Confirmation | Fresh weights/replay, no absent-branch parameters, same per-seed comparison; all signs of deltas retained |

The historical closure-to-briefing helper is retained, but no longer gates masks.
Native frozen-mask tests precede launch: no mutation, exact full-control native
replay, no partial coding blocks, source/checkpoint/support cache invalidation,
resume with no repeated evaluations, and failure on altered receipts.
Rank by paired mean INNER_A return per week; report TRAIN separately and use
absolute TRAIN-minus-INNER_A return/week gap only as a tie-breaker. Require
more than two orders/week in both periods in both seeds. Freeze tolerance 1e-5,
minimum one atomic block, and a 16384-unique-subset safety budget. A budget stop
is explicitly incomplete, never a claim of exhaustive or globally optimal search.
After each completed removal neighborhood, write a durable progress snapshot;
resume replays search decisions from authenticated cached measurements.
No gradient, replay buffer or early stopping exists in the mask evaluator.
Cross-device preflight exposed four changed near-tie actions in 24026 windows
with default cuDNN TF32. Disable cuDNN TF32 in the new evaluator. Authenticate
weights and compare logits to archived outputs (rtol/atol 1e-4), reproduce the
historical native account from its archived emissions exactly, then establish
a NEW control on the pinned inference device. Record every changed decision
and both accounts. Repeated checkpoint loading must reproduce canonical logits
bit for bit. Never reuse a historical account across different action vectors.
This is numerical requalification, not retuning the policy's decision threshold.
Final structural retraining is a separate confirmation stage, not implied by
completion of masks.
Do not silently reuse the generic rl_model receipt schema: it requires fresh
training and is intentionally incompatible with a frozen-weight mask diagnostic.

The proposed owner notice and contract are retained in feature_selector's
`docs/MASK_FIRST_FLOATING.md`. Repeated INNER_A use remains development, not an
untouched validation estimate. Neither negative returns nor TRAIN gains alone
prove that a variable is useful or useless.

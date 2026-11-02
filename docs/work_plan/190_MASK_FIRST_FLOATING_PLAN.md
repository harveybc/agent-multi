# Mask-first floating selection after DEAP

Owner accepted this sequence on 2026-11-02. The running DEAP campaign and
its combined TRAIN/internal-validation patience remain unchanged.

1. Finish the current search; authenticate its selected configuration and both
   native checkpoint receipts. Freeze hyperparameters, strategy, feature order,
   scaler, groups, calendar, costs, support and seeds. No nested DEAP per subset.
2. Prepare the owner briefing automatically when CLOSE.json exists. If there is
   no eligible champion, report that state without substituting a failed seed.
   Preparation never launches phase 2. Preserve the requested prelaunch notice.
3. Use feature_selector's floating search with a separately identified frozen-
   checkpoint evaluator. Zero removed channels AFTER fixed TRAIN scaling over
   the entire window. Preserve original tensor width, branches, weights and
   input-validity mask; a removed channel does not restore missing dates.
4. Evaluate fresh native accounts on identical fixed TRAIN/INNER_A blocks for
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

The closure-to-briefing helper is implemented separately from the frozen running
DEAP code. The native masked evaluator and refit queue must pass their acceptance
tests before the phase-2 prelaunch briefing; this document does not claim those
experiments have run. The maximum subset budget, ranking/tolerance and finalist
identities are to be frozen in that briefing before inspecting masked outcomes.
Do not silently reuse the generic rl_model receipt schema: it requires fresh
training and is intentionally incompatible with a frozen-weight mask diagnostic.

The proposed owner notice and contract are retained in feature_selector's
`docs/MASK_FIRST_FLOATING.md`. Repeated INNER_A use remains development, not an
untouched validation estimate. Neither negative returns nor TRAIN gains alone
prove that a variable is useful or useless.

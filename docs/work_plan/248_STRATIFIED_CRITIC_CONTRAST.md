# Stratified Critic Representation Contrast

## Purpose And Authority

Owner authorized continuation of247. Diagnose whether allowing the online
critic representation to adapt helps beyond the32-example memorization test.
No actor update, feature selection, strategy change or model promotion.

## Frozen Design

- Same70 columns, fixed TRAIN scalers,13 observations and stride2. No state
  feedback, rolling scaling, clipping, reduced windows or new features.
- Native quarterly account episodes within2020-2023, with the same risk,
  cost, calendar and forced-terminal liquidation rules as the source engine.
  Each quarter starts flat at the configured capital. This is an explicit
  diagnostic episode convention, not a replay of the uninterrupted account.
- Fit2020-2022; monitor2023. Every action label stops at its own quarter's
  native terminal settlement, without bootstrap or continuation across roles.
  Monitor observations must have their complete model window inside2023.
- Source policy, feature selection/scalers and initial calibrated critic
  previously used2023. The monitor is held out of THESE gradient updates,
  not an untouched test of the entire research pipeline. VALIDATION2024 and
  TEST are not scored or used for this experiment.
- Reference greedy actor determines the natural state trajectory. Candidate
  anchors are classified before future returns: permitted flat entries,
  restricted flat entries, open long, open short. Flat permissions include
  native first-step order admission, not only a calendar predicate.
- Fixed panel seed19. Per quarter select up to32 anchors in each permitted
  flat/long/short stratum and8 restricted-flat anchors, without target ranking.
  Publish available and selected counts, including shortages. Up to1664
  anchors:1248 fit and416 monitor. Sparse strata are never synthetically filled.
- Evaluate all three forced first actions from exactly the same native state,
  then follow the frozen parent to the episode end. Labels are discounted
  reward returns, not hindsight-optimal actions or entropy-augmented returns.
- Heads-only versus online critic representation+heads, seeds19 and43,
 4096 updates each, batch128, same covering schedule and pairwise-plus-mean
  Huber objective. Native learning rate/weight decay; fresh optimizer state.
  Actor, temperature and target networks remain frozen. Inference-mode
  diagnostic critic forwards match246, with trainable attention fastpath
  bypass scoped only to the critic. No new dropout intervention.
- Report0/64/256/1024/2048/4096 updates, per operational state and per role:
  both twins and their minimum, pair ordering, regret, gap/level RMSE.
  Report equivalence-only states separately; do not call them successful
  informative rankings. Same fixed endpoint, no monitor-based selection.

## Tests Before Launch

| Requirement | Test / Evidence |
|---|---|
| Temporal label isolation | Reject overlapping fit-label/monitor boundaries and incomplete monitor windows |
| Fixed sampling, no target-driven choice | Deterministic stratum quotas, uniqueness and shortage accounting |
| Native labels, no reward shortcut | Parent branch reproduces reference terminal cash; hold-action equivalence; independently rerun witness suffixes |
| No state corruption | Every fork checks unchanged parent; native terminal accounting reconciles |
| Matched optimization | Same source weights and hashed minibatch schedule for both scopes |
| Actual encoder updates | Real CUDA gradient canary and changed-tensor checks |
| Actor preservation | Frozen-parameter checks and exact full-TRAIN logits at every arm endpoint |
| Resumability / evidence | Per-quarter and per-arm hashes, configuration pin, exclusive lock, terminal collection |

The quarter-boundary horizon differs from previous full-TRAIN suffix targets.
Comparison is ONLY between the matched arms here, not their numeric Q levels
versus247. Within-role anchors remain correlated; these are diagnostic
curves, not independent-trade confidence intervals.

## Automation And Decision

Run native panel generation on CPU, then the four GPU arms sequentially;
collect authenticated artifacts automatically. Preserve completed quarters and
arms on restart, recompute only an unfinished arm. Store diagnostics as separate
checkpoints; do not overwrite the trading champion.

The next decision uses per-state held-out behavior AND fit behavior. Small-set
fit alone cannot authorize actor refinement. If improvements remain confined
to fit, inspect state-dependent aliasing and temporal drift before another
feature-removal pass. No automatic actor update or production deployment.

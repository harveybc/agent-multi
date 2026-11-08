# Full TRAIN Critic Confirmation

Follow [217](217_FIXED_ACTOR_CRITIC_AUDIT_RESULTS.md) before any actor refinement
or feature-selection cycle. The short replay identified a target-aggregation
effect but did not establish a complete correction.

## Fixed Comparison

Restore epoch40 separately for production twin-minimum and experimental mean
targets. Freeze actor, encoders, temperature and all non-head parameters; retain
the same optimizer state. Reconstruct every TRAIN2020-2023 transition using the
authenticated frozen epoch40 sampling-seed7 actions in the native simulator.
Require exact timestamps, positions, commands and durations, reward agreement,
one terminal transition, the expected24025 decisions, account P&L and trade count.
Do not synthesize transitions from assumed next rows or stitch across periods.

Use1024 updates of batch128 with common shuffled permutations. Every replay row
must occur in the schedule, not just be eligible for sampling. Save the replay,
its tensor digest, native reconstruction receipt and schedule identity. The
13-observation/stride2 windows,70 features and all account rules stay unchanged.

At0/1/64/256/1024 updates measure calibration on all TRAIN decisions, in chunks,
against realized soft returns of seeds7/19/43. Seed7 supplies learning replay;
19/43 are alternate action trajectories on the same dates, not independent
market holdouts or new model fits. They must not alter training or stopping.
No VALIDATION account, TEST, actor update, checkpoint promotion or automatic
feature-selection launch is part of this bounded confirmation.

## Tests and Gates

| Requirement | Evidence |
|---|---|
| Full native trajectory, including terminal | collector regression and row-by-row source receipt comparison |
| No partial-period substitution | expected TRAIN decision count, support and timestamp checks |
| Every row used | covering schedule unit test and persisted coverage count |
| Memory bounded without shortening windows | preallocated CPU replay; chunked Q evaluation |
| Correct sampled-return comparison | existing return-calibration fixtures; all three support arrays equal |
| No policy change | frozen parameter and actor-logit equality at every measurement |
| Restart safety | immutable source/runtime, exclusive lock, identity checks; completed arms reusable |

Publish bias, RMSE and tail errors for both target choices and all three action
seeds, plus compute time. A directionally smaller bias is not sufficient for
promotion if absolute calibration remains poor. A separate realized-return
anchor contrast is the next diagnostic in that case, before joint refinement.

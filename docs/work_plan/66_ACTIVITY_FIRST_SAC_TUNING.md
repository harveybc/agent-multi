# Activity-first SAC calibration

Owner directive2026-10-25: models must exceed two executed trades/week; inactive
policies are not financial comparators. Existing reporting of activity did not
provide a tuning successor. This study adds that missing step, not forced orders.

## Evidence and hypothesis

The completed critic-only study reproduced its control exactly. Both final
policies execute zero trades. Control closes longs while flat; no-dropout closes
shorts while flat. Automatic alpha ends at0.000567 and0.0000108 respectively.
Invalid closes correctly become HOLD. Missing agent state and a continuous SAC
coordinate partitioned into discrete commands remain structural limitations;
neither is proved to be the unique cause. Do not silently change either.

First bounded grid: fixed entropy coefficients0.01,0.1,1.0. These logarithmic
values test progressively stronger exploration than the observed final automatic
coefficient. Use original actor/critic dropout0.3, wd0.01, identical initial
tensors, seed7,100000 decisions,25k checkpoints, same reward units and recipeR3.
Reuse the authenticated auto-alpha control; no redundant control fit.
No trade reward bonus, random evaluation, forced entry, lowered costs, extra
feedback, new features, action remapping, early stop or TEST read.

## Activity before ranking

At final100k: TRAIN and INNER_A must each have executed closed trades divided
by all scored calendar weeks strictly greater than2, plus original validity,
daily entry limit and no ruin. Week denominator includes zero-trade weeks.
Report per-period activity at every checkpoint; do not select checkpoints.
Among eligible final arms rank mean of TRAIN/INNER mean daily log returns, then
smaller absolute generalization gap, lower INNER drawdown, stable arm id.
OBS_2024 is reported separately and never decides the choice. It must satisfy
activity independently to be a valid 2024 comparator. Exactly2 does not pass.
No eligible arm means ACTIVITY_NOT_MET, never winner by zero loss/drawdown.

## Automation and tests

Reuse finite queue, two attempts, recovery timers, checkpoint authentication,
native reports, warehouse readback and independent checksum mirror. Three fresh
arms sequentially on primary GPU. Record training command/invalid-HOLD counts
and fixed alpha, plus frozen TRAIN/INNER policy/Q probes. End after comparison;
an unsuccessful grid leaves a specific next hypothesis, not a claim of success.
Next structural test, if needed: command distribution/action-space and permitted
position routing, not a larger model or arbitrary trades to meet the quota.

| Test | Required evidence |
| --- | --- |
| AT01 | Fixed alpha only; initial policy tensors equal control; saved alpha reloads |
| AT02 | Missing/zero/NaN/exact2/wrong rate invalid; >2 passes; original limits retained |
| AT03 | No selection on 2024 or intermediate checkpoints; no inactive winner |
| AT04 | Authenticated parent and recipe/design immutable; malformed grid refused |
| AT05 | Collector binds effective entropy and activity to native report |
| AT06 | GPU admission and real first progress; sequential finite dispatch |

SB3 primary reference: https://stable-baselines3.readthedocs.io/en/master/modules/sac.html
The implementation uses the installed SAC ent_coef API, not a new optimizer.

## Predeployment evidence

68 focused tests passed on CPU with the pinned native business engine. New
tests first failed because the activity module did not exist. They now cover
strict activity, malformed/missing counters, inactive and incomplete grids,
2024 independence, exact initial tensors, fixed-alpha training/reload, collector
tampering, parent/design identity and finite retry/continuation. Three initial
reload tests accidentally used stochastic prediction; corrected to explicit
deterministic inference, matching the production evaluation contract.

## Verified deployment

Scientific producer7373699f; spec853c1ef85e8673c2ea3e650c754ae9cc5a2ae3ac82cfbc280609301ef3c6cb97.
At2026-10-25T09:09:08Z ACT_ALPHA001 was training on the primary GPU:
2250 decisions,1249 updates. Initial tensor digest matches the authenticated
control. GPU admission passed all three arms,14 synthetic updates each.
Worker recovery and collector timers enabled. First copy/outbox pump succeeded
with0 expected reports,0 pending and0 discrepancies before the first checkpoint.
ETA for the full three-arm batch approximately2h based on prior38min/arm.
No new financial/qualified-activity result exists at this initial check.

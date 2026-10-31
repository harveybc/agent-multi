# Discrete SAC Probe Launch

Runtime: `ab2a27b4`, plan173, campaign identity
`b9f0af7f7be14c5454aef6f19ab6417538ededa1f151a4c8f33ce41ccd4e6aaa`.
Owner explicitly chose discrete SAC and categorical entry plus market-only
CLOSE/HOLD curves. No PPO run is substituted or called SAC.

## Verified Before Financial Fits

- 28 focused tests passed, no skips, and scoped Ruff passed.
- Input/calendar/source digests checked; native preflight for every horizon and
  both internal periods passed.120 columns,31 groups,13 observations,stride2.
- Real integration smoke:128 transitions,eight replay updates,exact reload,
  24,026 sealed emissions,complete internal account replay; cash error4.55e-12USD.
  This smoke is engineering evidence, not model performance.
- Initial synthetic qualification was interrupted before financial training
  when review identified two-point rather than whole-curve coordinate sampling.
  Revision2 covers every interpolation interval; no first-run result is reused.

## Distribution and Automation

| Device | Units | Seeds/horizons |
|---|---:|---|
| RTX5090 | 6 | seed7 all four horizons; seed19 at3/6h |
| RTX4090 | 5 | seed43 at3/6/12h; seed19 at12/24h |
| RTX4070 | 1 | seed43 at24h |

The initial8/3/1 allocation was adjusted before revision2 using observed
qualification throughput: these small convolution batches do not realize the
assumed peak-GPU speed ratios. Each host has one sequential worker,6GiB RAM
limit, resource admission, bounded process retries and a persistent user
service. Qualification evidence is authenticated and reused per seed on a host;
real fits always start with fresh weights, optimizer and replay.

Stages: synthetic SAC learning -> mirrored learned-policy native positive
controls ->32768 real TRAIN decisions with uniform replay -> exact checkpoint
reload -> seal all market-only actor/twin-critic emissions -> native TRAIN and
INNER_A account replay -> authenticated result receipt. A scientific failure
is recorded and never converted into a pass by relaxing the tests.

A separate persistent collector mirrors the three queues, verifies receipts,
and publishes `AGGREGATE.json`, `RESULTS.md` and final `CLOSE.json` in the
private collection root. It requires all12 terminal outcomes, including any
failed qualification/process, before closure. The final JSON retains complete
costs, activity, direction counts/win rates, drawdown and native counters.
Failed attempts remain distinguishable; unfinished fits restart rather than
claiming exact mid-training continuation. No intermediate approval is required.

At06:17UTC all three workers and the collector were active, zero service
restarts. Worker memory was approximately2.1-2.9GiB, below each6GiB limit.
Qualification had reached1280/2048 updates on the5090,1408/2048 on the4090,
and448/2048 on the4070. These are engineering updates, not trading results.
The first full real fit will supply a meaningful campaign timing estimate;
qualification timing alone is not presented as a measured financial ETA.

## Scope

Weights and scaler fit2020 through2023H1. INNER_A is2023H2; upstream selection
and frozen risk references used that semester, so this is development diagnosis,
not untouched validation.2024 andTEST remain unread. No weekly refit, synthetic
path-noise calibration, floating selection or combined-horizon promotion occurs
in this queue. It supplies real learned-output evidence for the next decision.

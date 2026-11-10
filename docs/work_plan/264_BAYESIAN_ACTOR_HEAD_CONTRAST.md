# 264. Paired Bayesian last-layer SAC probe

Owner approved 2026-11-10. This is a separate bounded refinement experiment,
after cumulative phase2 completed 69 -> 69 -> 69 (263). No new mask pass,
feature changes, live promotion, MMD, or risk-controller changes are authorized
by this experiment. The selected69 source remains immutable.

## Scope and architecture

Compare a deterministic last actor Linear(8,3) with a diagonal Gaussian
variational last layer. Preserve the actual TorchRL functional actor weights,
69 inputs, 28 learned groups, fixed TRAIN scaler, 13 observations / stride2,
TCNs, attention, first head layer, action codec and native account engine.
The prior is centered at the inherited weights, with standard deviation 0.1;
posterior means start at those weights, posterior standard deviation 0.001.
This is a local transfer prior, not a claim of an independent-data Bayesian
posterior or calibrated confidence. It uses only prior TRAIN-derived weights.

Both arms optimize the exact discrete SAC actor expectation
`mean_s sum_a pi(a|s) * (alpha * log pi(a|s) - min(Q1,Q2)(s,a))`.
Bayesian arm averages four reparameterized weight draws and adds
`KL(q(weights)||prior) / N_TRAIN_transitions` (coefficient 1).
This is KL-regularized variational SAC, not supervised categorical ELBO.
Both arms use AdamW lr=3e-5, wd=0.01 and gradient clipping at 10.
No loss of prediction standard deviation is introduced.

Fresh critic heads and empty optimizer memory follow F-CRITIC-HEAD-RESET.
Calibrate once on full native TRAIN stochastic replay from the unchanged69
actor, with duration-aware realized soft returns, 4096 updates, batch128.
Both arms receive the identical frozen calibrated Q table and alpha.
Report calibration by action and the constant-return control. Own-policy
calibration does NOT establish alternate-action ranking; no policy promotion
or long refinement is authorized automatically by this bounded probe.

Seeds 7/19/43 define paired replay schedules. Each arm takes 256 updates;
every TRAIN transition occurs in its schedule. Assess full TRAIN and
VALIDATION accounts at updates 0/8/32/128/256. No feature-window shortening,
TRAIN subsampling, evaluation on TEST, or early stopping is used here.
The primary comparison is the predeclared endpoint 256 across seeds;
intermediate checkpoints diagnose drift, not a winner chosen after inspection.

Inference averages probabilities over 32 fixed common antithetic weight draws,
then uses the unchanged native greedy codec. Also report mean-weight inference
at endpoint and repeat posterior inference with independent draws to expose
Monte Carlo action sensitivity. Log posterior probability standard deviations,
predictive entropy, expected conditional entropy and their difference.
These are uncertainty diagnostics, not measured calibration to trading truth.

## Test-led traceability (design gate before implementation)

| Requirement | Test/evidence |
|---|---|
| B01 Mean weights preserve original output | Last-layer zero-noise parity; real full-support greedy account parity |
| B02 Genuine posterior, not fixed fake noise | Positive scales, stochastic draw differences, finite gradients for means/scales, nonnegative analytic KL |
| B03 Correct actor objective | Exact categorical formula and gradients vs direct reference; Q detached |
| B04 Only intended parameters change | Frozen source/encoder hashes; matched initial means; separate optimizer |
| B05 Reproducible inference | Fixed draws, batch-partition invariance, cold reload exact |
| B06 Valid temporal/account support | Input/scaler/source hashes, TRAIN-only replay, identical support, native accounting |
| B07 Fair and auditable execution | Paired schedules/Q, three seeds, endpoint table, per-side trades/early closes, no automatic adoption |
| B08 Sequential unattended work | Lock, pinned config, progress, completed-arm checksum verification, bounded restart, final report/plots |

Normal path: verify inputs -> reference accounts -> TRAIN replay -> common
critic calibration -> cached frozen embeddings/Q -> six sequential arms ->
endpoint report. Invalid identity, nonfinite tensors, source drift or failed
reload stops with explicit error. An interrupted arm restarts from its fixed
seed; already sealed arms are verified and skipped. No unfinished arm is
reported as complete. No hidden extension beyond 256 updates.

## Allocation and next step

Use the secondary RTX4090 for this contrast. Preserve the primary RTX5090 for
the separately qualified refinement line; this document does not claim that
another primary job is running. Emit config, checkpoints, native metrics,
uncertainty arrays, plots, progress and final hashes outside git. Publish only
compact results after collection. Then decide whether matched full SAC
refinement is justified; do not alter the selected69 or the live feed.

## Prelaunch evidence

15 CPU tests passed (last-layer and existing critic lifecycle). Full real-data
CPU preflight: 30,234 valid windows, zero action disagreements after hidden
embedding factorization, maximum logit difference 9.54e-7; TRAIN/VALIDATION
accounts exactly reproduce the selected69 source. These are wiring results,
not Bayesian trading results. No new training ran in this preflight.

## Launch receipt

Runtime `687a8ef3`, 2026-11-10 18:40:54 UTC (13:40:54 Bogota).
RTX4090 Laptop gate PASS; native GPU reference passed; 64 common critic
updates observed, actor unchanged, zero service restarts. Sequential six-arm
worker and hash-verifying collector active. Worker has a 6 GiB memory ceiling,
bounded failure restarts and completed-arm checksum verification. The collector
copies final weights, arrays, plots and tables to the coordinator automatically.
No Bayesian endpoint trading result existed at this launch receipt.

# Categorical output and reward execution sequence

Owner launch authorization: 2026-10-25. Implementation in
`tools/recipe_hybrid_queue.py`, collector `tools/recipe_hybrid_collect.py`.
This supersedes the tentative RLlib backend in plan 67: use the already
installed SB3 PPO optimizer with a tested custom conditional distribution.
No new RL training algorithm is written. The old SAC producers are unchanged.

## Effective contract

- Actor: existing recipe temporal encoder, compact head, five categorical
  logits with legal-command masking. Conditional direction-specific Beta
  distributions produce TP, SL and budget fraction for opening commands only.
- Command identity is an integer sampled from Categorical. A four-slot Box is
  only SB3's numeric rollout storage codec. Fractional command IDs are rejected;
  there are no intervals, rounding decisions or continuous command samples.
- NOP while flat and HOLD while positioned are counted separately. The action
  mask routes inventory and calendar restrictions; it is not fed to the temporal
  encoder. The critic uses the same encoder and a position-routed value head.
- Effective encoder dropout is **zero during both rollout and optimization**.
  The source recipe still records its original dropout, but the new algorithm
  override is explicit in DESIGN. This is necessary here for identical PPO
  likelihood evaluation. AdamW decay remains the recipe value. This is a new
  algorithm package, not a claim of a one-variable contrast with old SAC.
- Risk samples map inside the intersection of sigma bounds and executable
  distance rules. Quantity is rounded down at the execution price under the
  requested budget; minimum-size infeasibility never enlarges the order.
- Fixed-risk control uses midpoint feasible SL/TP and full permitted budget.
  Learned-risk arms use conditional risk heads. Matching seeds must have
  identical initial policy tensors before training.
- Elapsed-bar discount and GAE account for skipped feature windows. Missing
  windows do not become fabricated market inputs. Terminal placeholders never
  bootstrap. No nine-variable feedback vector, rolling scale or ZigZag labels.

## Finite sequence

Three seeds per arm: 7, 19, 43, sequentially, one GPU process at a time.
102400 policy decisions per fit, 512 per rollout, 256 per minibatch, five PPO
epochs per rollout, learning rate .0003, gamma .99, GAE .95, entropy .01,
clip .2, value coefficient .5, gradient norm .5. No early stopping or refit.
Weights and training diagnostics saved at 25600/51200/76800/102400; full native
TRAIN/INNER_A/OBS_2024 evaluations only at the fixed final checkpoint.

1. CAT_FIXED: categorical control, existing equity-delta reward.
2. HYBRID_EQUITY: learned SL/TP/volume, equity-delta reward.
3. HYBRID_LOG: log-equity reward.
4. HYBRID_DD025, HYBRID_DD100, HYBRID_DD400: log reward minus incremental MDD,
   respective penalty .25/1/4.
5. HYBRID_DSR: differential Sharpe with causal moments.
6. HYBRID_MIXED: .5*(100*log return) + .5*differential Sharpe.

DSR engineering settings: eta .01, 32-bar warmup, variance floor 1e-6 in
fractional-return squared units. These are predeclared numerical parameters,
not optimized discoveries. MIXED is an initial fixed-coefficient treatment,
not a claim of empirically equalized learning magnitudes. Reward component
scale calibration/refinement remains a later comparison. Nonpositive equity
raises a terminal experiment failure; no invented logarithmic substitute.

All eight ranking variants are computed from the common final reports, with
scales derived from the common eligible TRAIN candidates, frozen before INNER
ranking. Three complete eligible seeds required; observed 2024 never selects.
Calmar based on CAGR is reported separately; a robust Sharpe-uncertainty
estimator and adaptive coefficient refinement are not implemented in this
release and cannot silently dispatch. No DEAP or optimum claim.

## Automation and recovery

New private spec pins inputs, recipe, native engine, source files, software,
units and parameters. Queue lock prevents concurrent dispatch. Resource
admission checks RAM, disk, GPU memory and an idle GPU. Persistent service/timer
resumes after reboot or resource wait; interrupted fits restart as distinct
attempts, maximum two. Completed authenticated results are adopted without
refitting. A failed seed control prevents its dependent fits. All failures stay
visible; completion with failures is not successful full completion.

The collector checksum-copies results to a separate machine, authenticates
checkpoint/native metrics, and uses the existing governed warehouse outbox.
No destructive cleanup, changes to other experiments or public topology files.
An active service with a fresh training heartbeat is required before reporting
the campaign running; deployment itself is not a scientific result.

## Verification scope

Test-first failures observed for the new distribution/reward and fitness modules.
Focused tests cover categorical support, illegal/fractional commands, conditional
likelihood and inactive-risk gradients, analytic reward identities, DSR formula,
elapsed GAE, native order/cost rules, budget and reload parity, unchanged PPO
log probabilities, selection-year mutation, missing seeds, inactive selection,
and recovery without duplicate fit. A synthetic policy test additionally learns
each of NOP/long/short when that action is optimal. Its initial 2048-step probe
selected the correct long action but did not meet confidence .6; the frozen
engineering test uses 8192 steps and all three action cases. These are not
financial measurements or evidence of predictive signal.

Completed before deployment: 39 focused CPU tests passed in 34.93 seconds;
real-shape synthetic GPU admission passed for fixed and learned risk, four
updates each and saved-policy replay. Broker costs remain assumed; no
system-wide perfection or profitability certification is implied by these checks.

## Verified launch

Spec SHA256: `bcd180f16a548dd9e402adecfdb9cb578066c611c5dd1b1340896ff0cf8df9f6`.
First financial fit started 2026-10-25 12:21:44 UTC on the primary 32-GiB GPU.
At 12:25:48 UTC CAT_FIXED_s7 had 80384/102400 decisions and 780 PPO updates.
Worker and checksum collector timers are enabled; the first collector completed
COPIED_AND_PUMPED with zero reports expected before a final evaluation exists.
RAM cap is 6 GiB, swap disabled. Only one heavy GPU process is dispatched.
The other 23 fits are automatic, subject to the bounded recovery and control
dependencies above. No early stopping, periodic refit or TEST read is enabled.
Measured training throughput at that cut is about 331 decisions/second; final
evaluation costs still need measurement before a complete ETA is established.

Follow-up at 12:28:47 UTC: first unit completed successfully at 12:28:08 and
CAT_FIXED_s19 started automatically. First-unit end-to-end wall time was
384.5 seconds (worker measured 379.5); all three native period reports were
mirrored, delivered to the warehouse and read back, zero pending/mismatched.
INNER_A: 65 trades, 2.4074/week, win fraction .44615, net -1266.09 USD,
Sharpe -0.85848, maximum drawdown 30.4268%. This is one seed of the fixed-risk
control, not a three-seed treatment conclusion. Learned-risk timings are still
unmeasured; initial full-batch estimate is roughly three hours, provisional.

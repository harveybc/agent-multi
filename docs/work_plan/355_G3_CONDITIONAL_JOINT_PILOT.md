# G3 Conditional Joint Pilot

Owner launch authorization2026-11-14 follows354. Preserve selected299, all71
features, the fixed TRAIN scaler and original real evaluation. No STL or live
substitution. First materialize and qualify a conditional synthetic TRAIN path.

## Frozen Design Before Results

The experiment conditions on the ORIGINAL released macro/COT trajectory and
calendar. It does not invent macro publications. Sample the seven FX price
innovations JOINTLY conditional on that context and the generated EURUSD past.
This is a conditional augmentation experiment, not an unconditional generator
of the entire economy. Historical macro is a conditioning variable, not an
unrelated column attached after generating prices independently.

Use a TRAIN-only empirical conditional sampler: nearest neighbors of the
current lagged macro/COT context, lagged24-row return and realized volatility,
within the same UTC hour/week-day and gap class. Scale the SAMPLER context
with TRAIN constants only; do not change the agent's input transforms. Choose
among10 neighbors uniformly. Continue a sampled block for at most4 rows only
while source/target clock classes match and source innovations and target
conditioning context are complete; otherwise resample from current context. These are frozen pilot
settings, not claimed optima. One fixed generation seed19; no best-of-seeds
selection. No added Gaussian noise or post-hoc clipping.

One innovation row contains EURUSD close return, open relative to previous
close, upper/lower log wick distances, and all six cross close returns.
Reconstruct positive OHLC and cross prices recursively. Preserve original
cross observation masks. The first256 rows seed context from actual TRAIN;
they are not counted as generated. Recompute all71 derived inputs using354,
including original row-specific OOF activity estimators, frozen factor/GMM/
Kalman state and original publication joins. Keep the original timestamp grid,
so the OOF mapping is unchanged. No generated values become live data.

## Tests And Gates

Before implementation: unit tests cover deterministic prefix generation under
a frozen fit, future exclusion at fitting, joint donor identity, OHLC algebra,
clock matching, source mask preservation, invalid configuration and empty donor
groups. Integration replays the zero-change original producer control from354,
recomputes synthetic features, checks full coverage/order, and executes native
orders/rewards on synthetic bars with unchanged business/cost rules.

Fit/scoring statistics use TRAIN only. For every FX return series, compare
Wasserstein distance (scaled by real TRAIN standard deviation), standard
deviation and absolute99th-percentile tail to the variation between real TRAIN
quarters. For correlations compare the maximum matrix difference to the maximum
between quarters. The pilot gate allows1.25 times the measured quarterly
envelope, with numerical floors only. Preserve the failed candidate if a gate
fails; do not tune thresholds after seeing it. This is a coarse distribution
screen, not proof that a synthetic path is realistic or profitable.

Only after these gates pass: a real-only/90% real+10% synthetic replay contrast
with matched update budget, initial state, seeds, architecture and real
evaluation. All real TRAIN remains in the primary traversal. Synthetic
transitions come from a separate native account; never join their positions,
next observations or rewards to the real account. Report actual sampled mixture.
The generator passing is not a model improvement or an automatically promoted
checkpoint. The training adapter and recovery tests must pass before dispatch.

## Training Contract

Start both arms from the authenticated selected299 checkpoint, including its
optimizer state. Restore the model constructor's online parameter trainability
in BOTH arms, rather than silently retaining old head-only freezes. This is a
matched full-model continuation, not a fresh-from-random experiment. Preserve
the original71 inputs,13 observations/stride2, architecture, fixed scaler and
real account support. Batch128, update every4 real decisions, warmup4096,
seed7. The control runs up to2000 full TRAIN epochs, validation-only patience20,
min_delta1e-8,20-epoch counting warmup; eligible early checkpoints are retained.
The starting checkpoint remains an eligible fallback.

Then the augmentation arm runs EXACTLY the control's completed epochs and
updates, even if its own shadow patience would have stopped sooner. Record
that signal, but do not confound augmentation with a shorter training budget.
Sample cumulatively10% synthetic transitions,12 or13 out of each128 batch.
Keep independent65536-capacity real/synthetic replay stores and separate
native positions/rewards. Both use the same explicit seeded replay sampler.
The synthetic collector has its own action RNG; its inference restores Torch
RNG so collecting extra transitions does not consume real action randomness.
Recovery commits both stores, optimizer, RNG streams and sample counts at each
epoch. Report full TRAIN/VALIDATION accounts and PNG curves each epoch. No
automatic model promotion, live changes or automatic further search.

## Materialization Results

Generation seed19 completed in26.78s:24,639 generated bars plus256 initial
context rows; all71 inputs rebuilt. All seven return distribution gates and
the cross-correlation gate passed the predeclared quarterly envelopes.
24,052 synthetic complete windows (distinct synthetic support, not substituted
into real evaluation). Original real support and original model remain intact.
Generator cold/prefix replay and71 recomputed-feature prefix checks are exact.
Native NOP submits0 orders; BUY/SELL/alternating controls reconcile within
2.55e-11 USD. Those controls validate accounting, not trading profitability.

The real account preflight reproduces selected299 exactly on CPU:
TRAIN+16.05822355000215%,745 trades; VALIDATION+18.369945750000117%,207 trades.
These are the preserved starting results, NOT new augmentation results.

## Execution And Traceability

50 focused tests pass with the native engine fixture enabled, including full
epoch interruption/recovery with synthetic collection and matched-budget
continuation beyond the shadow patience. Memory and memmap replay ring wrapping
and cold sample reproduction are covered. The CPU preflight also passes on the
assigned training host. RTX5090 readiness gate passes.

The real-only control is running. A persistent user service automatically runs
the90/10 arm next, resumes committed epochs after interruption and limits
repeated failure restarts to3 within an hour. Progress, history, full metrics,
plots, both replay buffers and model bundles are stored outside git. After
both arms it writes a matched-budget comparison and exits; it does not promote
either model. Longer historical data remains a separate unadmitted candidate.

| Requirement | Implementation | Evidence |
|---|---|---|
| Joint innovations, TRAIN-only fit | `app/g3_joint_sampler.py` | sampler tests; sealed generation result |
| Recompute71, frozen producers | `tools/prepare_g3_joint.py` | original parity, synthetic prefix, native controls |
| Independent accounts and90/10 | `app/g3_replay.py` | mixture/ring/recovery tests |
| Matched learning budget | `tools/run_g3_contrast.py` | shadow-stop continuation test; terminal budget assertion |
| Original real evaluation | existing native assessor | exact starting TRAIN/VALIDATION accounts |
| Recovery and plots | structural epoch runner | full interrupted/uninterrupted native learning test |

Generation identity: `f6d167c6a8ae7827f9d19ea5016d40094e348aa99efef1b1a5e4e713dd7c5252`.
Training identity: `6eb265d3fc39231a089c2fb571ef5fecc14a8e0967c870680559828404be9069`.

# Completed generalization diagnosis, 2026-11-06

## Scope and execution

G1 of [203](203_GENERALIZATION_DIAGNOSIS_AND_CONTRASTS.md) is complete. All three
regularization runs ended at epoch40. This diagnosis loaded their selected and
final checkpoints on CPU; no new training, source-data changes, TEST reads or
feature elimination occurred. Inputs remain93 columns in30 groups,13 observed
bars at stride2, four TRAIN years and calendar2024 validation. All figures
below are from seed7, not a multi-seed conclusion.

`tools/run_generalization_diagnostics.py` authenticates each pinned producer,
input recipe and recovery manifest, and runs a sequential, resumable queue.
It writes REPORT.json, QUARTERS.csv, INPUT_SHIFT.csv, PNG training diagnostics
and a DONE receipt outside the checkout. A queue lock prevents duplicate
execution; QUEUE.json reports COMPLETE or the concrete failed stage. Reuse
requires matching run, diagnostic-code and report hashes. It never launches
training as a side effect.

## Selected checkpoints

| Arm | Placement | Selected epoch | TRAIN return % | VAL return % | VAL Sharpe | VAL orders | VAL win % | VAL DD % |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| R1 | Critic core dropout0.1 | 7 | 39.0376 | -1.1815 | -0.1157 | 266 | 50.7519 | 7.2645 |
| R3 | Critic post-core BN | 11 | 0 | 0 | undefined | 0 | undefined | 0 |
| R4 | Critic first-core-convolution BN | 14 | -0.5822 | 0.0990 | 1.0014 | 1 | 100 | 0.2029 |

R3 and R4 fail the activity requirement. R4's one winning short is not evidence
of a useful generalizing policy. R1 validation has59 buys and207 sells;52 agent
closes, of which21 help and31 harm versus passive protection of the same fixed
position. That close comparison does not include subsequent account opportunities.

At epoch40, R1 reaches TRAIN+1109.1969% but VAL-22.8351%; R3 has TRAIN-84.4243%
and VAL-22.5629%; R4 has zero trades in both periods. The selected and final
checkpoints must not be conflated. R1 demonstrates deterioration with continued
training; it is not justified to dismiss overtraining as a possible contributor.
It does not establish that early stopping alone can solve generalization.

## Measured findings

### Replay covers the four years

All65,536 stored observations AND next observations match the reconstructed
decision sequence exactly in each run:196,608 transitions checked overall.
Terminal flags match the end of each full TRAIN traversal. The writer has
961,000 transitions,40 traversals of24,025 decisions. Retained transitions
come from epochs38-40 and cover all16 TRAIN quarters; they do not contain only
the last calendar year. Depending on ring position, a historical decision
appears two or three times. This is retained-buffer composition, not a record
of which batches the optimizer actually sampled; that history was not logged.

### The BN failures are distinguishable

Frozen probes use512 equally spaced eligible observations per period and
1,024 transitions from the final retained TRAIN replay. They restore weights,
buffers, mixed module modes and RNG. They do not fit BN on validation.

* R3 epoch40: sampled TRAIN Q values range from-317,071.84 to-82,943.16,
  while sampled immediate reward spans approximately-0.47 to0.48. Mean absolute
  frozen TD residual is857.14. Its final recorded pre-clipping gradient norm is
  13,157.20; the update code does clip gradients, so this is NOT an unclipped
  optimizer-step norm. Critic instability is measured; its ultimate cause is
  not established by these probes alone.
* R4 epoch14: a sampled TRAIN observation produces Q up to2,065,120.63 and
  validation up to394,701.22. The largest-Q TRAIN observation has maximum
  absolute standardized input3.78, so a huge raw input is not required for this
  witnessed amplification. Comparing batch statistics with running statistics
  changes Q by up to1,165,235.75 in this diagnostic. This is a counterfactual
  measurement, NOT a proposal to evaluate validation in training mode.
* R4 epoch40: all six per-action critic outputs have exactly zero standard
  deviation over both512-row samples, and full native evaluation makes zero
  orders. A low mean TD residual0.0322 here is not successful learning.
* R1 epoch40: mean absolute TD residual0.0883; sampled twin-Q disagreement
  grows from0.075 in TRAIN to0.693 in VAL. It is materially more stable than
  these BN arms, but this does not establish robustness or a winning policy.

The independent TD0 computation includes duration discounting, terminal masks
and entropy. It matches TorchRL's native target within rtol2e-5/atol2e-5.
Largest absolute differences are0.0625 on very large R3 targets and0.0410 on
R4 targets; these are not bit-exact replay claims. Selected-checkpoint TD is
evaluated on the FINAL replay buffer, not its unavailable contemporaneous one.

### Performance and input distributions vary by period

R1 selected-checkpoint quarterly validation returns are-0.203%,+2.969%,
-1.915%,-1.958%, with48,47,69,102 closed trades. These are slices of the same
continuous account, including entering equity; no quarter resets positions or
capital. Trades are explicitly attributed by exit date, unlike mark-to-market
equity changes.

Largest absolute mean shifts in TRAIN-standard-deviation units include COT net
share-1.2715,ATR ratio+1.0916,ATR-0.7568,COT net-z156-0.7528 and wavelet log-price
D2 energy-0.7039. These observations motivate stability checks; they do not
prove data corruption or authorize deleting columns. All93 per-column results
are in INPUT_SHIFT.csv.

Reversing the order of a frozen window changes R1 selected greedy actions in
37.3% of the TRAIN probe and20.5% of VAL. Temporal order matters to this policy.
Reversal is an out-of-distribution sensitivity probe, not a feasible market,
causal importance estimate, or augmentation recommended for training.

## Next contrast and its concrete constraints

G2 preserves the original13xstride2 branch and adds a small13xstride6 context
branch. The latter lacks context on160 of24,026 eligible TRAIN windows and8
of6,208 validation windows:48 early-history cases and112/8 nonfinite-context
cases respectively. Terminal eligible windows are not trading decisions, hence
the one-row difference from24,025/6,207 decision counts.

The rows are admitted observed H1 bars, NOT a continuous clock grid. Long-context
elapsed span has median120 hours, range72-153 in TRAIN and72-154 in validation,
because closed sessions occupy no rows. It must not be described as a fixed72h
window. The original branch and every baseline decision remain available.

Implementation contract for the next arm:

1. Keep original features, fixed scaling, business, attention and selected
   training/monitor settings. Do not promote either BN placement. The existing
   attention-only control remains the architecture reference; R1 is a separate
   regularization result, not silently combined into G2.
2. Context extractor: same feature groups, two small causal residual TCN blocks
   per group, width4 versus width8 in the original branch. Align extracted
   context to original slots by latest available past timestamp, not linear
   interpolation or access to a later context slot.
3. Skip the context branch when its inputs are unavailable; use the unchanged
   baseline path. Do not replace missing source observations with synthetic
   values or delete baseline decisions. This requires an explicit availability
   contract and tests before a training run can be claimed as started.
4. Tests: future perturbations cannot change earlier outputs; unavailable-context
   path matches the baseline; optimizer reaches the context parameters; bundle
   reload and interrupted replay preserve both branches; data support identical.
5. Do not change learning rate, reward, activity threshold or add synthetic data
   in the same contrast. Record Q/TD and activity diagnostics so another collapsed
   policy is recognizable without manually examining final returns.

G3 remains separate. The inspected `synthetic-datagen` OHLCV bootstrap producers
cannot alone generate the cross-asset factors, rates, COT publications and event
dependencies of these93 columns. Keeping unrelated real external features next
to generated EURUSD is not an admissible shortcut. A producer coverage map and
joint regeneration tests are required before synthetic transitions enter replay.
Resampling intact historical episodes is a different control, not new synthetic
information. No synthetic training or G2 training was launched in G1.

## Evidence and verification

| Run | Identity | REPORT.json SHA256 |
|---|---|---|
| R1 | be89e2e5b123183dbb751e805ac5b497b15d705b1eaff265b39789beae2e44bf | e193c3fc105985b0ea30cb2f605b154b56ff85e28ba5506e7408e846b41213dc |
| R3 | 02b01c1397d736ef819ff9200e0ccce253f4d60465d5ce21f328982baf884099 | cd56d7e5a8810e2bfb2e1f0ddc49afbdf7c6623151bec0faed68bb4c6b6ba6d9 |
| R4 | 56bc409298502bd1973db1d435a91380684d55b541a883103fcee1b6757cdc57 | 5c89853c2983aaeb9c58e60cbfe7aabea90ece494947c8b001b66e6b7c08fefe |

Focused CPU suite:29 passed,5 CUDA-only skips. Six new regression tests cover
ring wrap/cursor rejection, mismatched windows, quarterly equity attribution,
missing-context preservation, BN/RNG restoration on error and terminal/duration
TD targets. The production replay matching and native TD parity above are
additional real-data acceptance evidence. No full repository suite is claimed.

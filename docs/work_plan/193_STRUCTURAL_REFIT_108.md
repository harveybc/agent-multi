# Structural refit after the ordered two-pass selection

Owner instruction, 2026-11-03: train the selected inputs structurally, not as
zero masks, and preserve weights for the next inference-only selection pass.
This follows192 and precedes any further feature search or DEAP.

## Frozen Experiment

-108 ordered columns,31 retained groups; authenticate the completed two-pass
 result and its receipt before constructing the recipe.
-Fresh discrete SAC weights, optimizer and replay. Preserve the baseline
 double-TCN architecture,13 observations at stride2, batch128, learning rate
0.0003, gamma0.99, tau0.005, entropy fraction0.6, weight decay0.01,
 replay65536, warmup4096 and one update per4 decisions. Seeds7 and19.
-The interrupted DEAP did not establish a champion. These are the fixed
 baseline parameters used by the mask evaluator, not a claimed DEAP optimum.
-TRAIN:2020-01-01 through2023-12-31; VALIDATION:calendar2024; TEST2025
 remains closed. This is a new split identity, not the historical
3.5-year TRAIN / half-year INNER_A comparison.2024 is development validation,
 already exposed in historical work, not a virgin holdout.
-Refit fixed scaling constants using TRAIN only. Preserve declared consumer
 transforms and full-parent observation support; project the tensor to108
 channels. No rolling scaling, input clipping, imputation, state feedback,
 resampling, reduced windows or new risk rules.

## Epoch And Stopping Contract

An epoch is one complete chronological TRAIN account traversal, including
native bars with unavailable observations (automatic holds). Reset the account
at epoch boundaries; retain replay and learned weights. VALIDATION never enters
gradient updates or replay. Evaluate one fixed checkpoint on each entire
partition after every epoch, not just sampled blocks or online rollout rewards.

The maximized monitor is
`0.5 * (TRAIN return_pct_per_week + VALIDATION return_pct_per_week)`.
The first finite score establishes the best. A score greater than the best by
more than1e-8 replaces it and resets the counter to0. Otherwise increment once.
Stop at20 consecutive failures or2000 complete epochs, then restore the best.
Activity eligibility is reported separately, not silently substituted into the
owner's patience formula. Report profit, return, Sharpe, drawdown, trades,
orders/week, wins, directions, costs and close reasons for both partitions.

## Persistence And Next Selection

Each improvement exports an immutable inference bundle: weights, effective
architecture, ordered columns, scaler, group indices, split, business contract,
probe inputs and reference logits, plus hashes. A cold model reconstructed
from the bundle must reproduce logits exactly on the same backend. No training
is required for loading this bundle. Subsequent masks apply to its108 channels;
physical deletion and refitting remain separate operations.

Epoch-boundary recovery additionally retains optimizer, replay/writer state,
random-generator states, global steps, updates, monitor history and patience.
An atomic pointer switches only after all checkpoint files are durable. Keep
two recovery generations; do not retain a full replay copy for every epoch.
A restart can redo only the unfinished epoch from its previous checkpoint.

## Test Design And Traceability

| Requirement | Evidence before launch |
|---|---|
| Physical channel/branch removal | Structural subset tests and real tensor/model preflight |
| Four years plus one, no TEST | Fixed split assertions, timestamp coverage, filtered loaders |
| TRAIN-only constants | Future-perturbation scaler test |
| Exact patience semantics | Tiny improvements, ties, reset and20-failure unit tests |
| No inference retraining | Bundle cold-load equality and tamper refusal |
| Restart preserves learning | Replay sample, optimizer and next-update parity test |
| Full native evaluation | Synthetic smoke of epoch/evaluation/recovery, real support receipt |
| No validation learning | Training factory restricted to TRAIN, replay unchanged by assessment |

Execution is unattended after tests and native preflight. Record measured epoch
times; an ETA to the next20 no-improvement checks is conditional, not a promise
that early stopping will occur then. No subsequent selection auto-launches.

## Preflight Evidence

31 focused tests pass, including native synthetic epoch training and exact
uninterrupted-versus-resumed learning. The real support spans48 TRAIN months
and12 VALIDATION months:24025 and6207 actionable decisions per epoch/account,
respectively, with24894 and6220 native transitions. Feature warm-up and missing
observations remain explicitly unavailable; they are not filled or compressed.
The resulting observation tensor is`[30859,13,108]` across both partitions.
Native mirrored opening/closing and flat controls pass before financial fitting.

The initial full-size108-channel inference bundle reloads exactly on CPU. This
is an engineering check, not a trained result. Every selected trained checkpoint
must independently pass its own reload check on the worker backend.

## Execution And Inference

Runtime84ee87a9 launched both seeds at2026-11-03T06:47UTC in independent
persistent workers. The primary targets the physical RTX5090 and the secondary
the RTX4090. A separate collector copies weights and reports every180seconds,
excluding the large recovery replay. No intermediate approval is required.
Run identity:
`82cfdb8e75fb85fbdb64f4c978fab1ce88793e64f92e8163b53f90aa258a507e`.

`BEST.json` points to the best immutable bundle. `HISTORY.json` contains both
full-account metrics at each completed epoch; `PROGRESS.json` records the live
epoch. `RESULT.json` is written only after stopping, restoring the best and
evaluating its final accounts with close-counterfactual diagnostics.

Inference does not enter the training runner:

```python
import torch
from app.structural_refit import load_bundle

model, manifest = load_bundle("<best-bundle-directory>", device="cpu")
# windows: [batch,13,108], using RECIPE.json's ordered columns and fixed scaler.
with torch.inference_mode():
    logits = model.surfaces(torch.as_tensor(windows, dtype=torch.float32))
```

For native backtests and the next mask pass, reconstruct observations through
the run's`SUPPORT_RECIPE.json` and project to its`RECIPE.json`, as`market_for`
does. This preserves the accepted full-parent missing-observation support while
feeding only108 actual channels. Loading only reduced raw columns must not
silently change the comparison's decision dates. The inference probe verifies
the trained network on normalized windows, not raw-data availability by itself.
For the next floating pass, freeze one numerical backend and first evaluate
the unmasked108-input control on that backend. Do not compare CPU masked
scores against archived GPU baseline scores without the paired replay check;
near-tie categorical decisions can be sensitive to cross-device arithmetic.

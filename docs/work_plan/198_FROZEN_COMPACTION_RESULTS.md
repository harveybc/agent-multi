# Frozen compaction: exact behavior retained

Completed 2026-11-04 15:51 America/Bogota (20:51 UTC).
Plan197 was frozen before implementation and the native diagnostic.
Runtime `93e519ca`;58 focused CPU tests passed. The sequential six-arm GPU
diagnostic took15.17 seconds excluding process startup,20.42 seconds including
startup, with2.3GiB peak host RAM and no swap. No training occurred.

## Result

The compact94-input actor reproduced ALL30,234 valid-window logits bit for bit
on the original GPU backend. Maximum absolute error0; zero changed actions.
Full TRAIN and VALIDATION action logs, trade logs and equity traces were exact.
Packing control and fresh94 reference also replayed their archived accounts.
All source SAC tensors remained unchanged. A cold compact-bundle reload
reproduced the full emission exactly. The collected74 artifacts passed SHA256.

| Arm | TRAIN return % | TRAIN Sharpe | VAL return % | VAL Sharpe | VAL DD % | VAL trades | VAL win % |
|---|---:|---:|---:|---:|---:|---:|---:|
| Original108, unmasked |367.4319|4.6768|-3.4537|-0.4509|8.7799|321|50.7788|
| Original108, selected94 mask |210.1869|3.6164|10.4471|1.4496|4.7276|298|57.7181|
|94-input packing control |210.1869|3.6164|10.4471|1.4496|4.7276|298|57.7181|
| **Physically compact94 actor** |**210.1869**|**3.6164**|**10.4471**|**1.4496**|**4.7276**|**298**|**57.7181**|
| Compact94, constant response zeroed |204.0784|3.5575|9.3760|1.2991|4.5062|301|57.1429|
| Fresh94 fit, checkpoint30 |93.2919|2.9875|-1.2114|-0.2796|4.6090|112|50.8929|

TRAIN is2020-2023; VALIDATION is2024. Returns are cumulative over each full
period, not directly comparable annualized rates. All costs and support are
unchanged. VALIDATION was used in stopping and selection and remains exposed
development data, not independent confirmation. TEST2025 was not read.

## What changed structurally

- Actor input is `[batch,13,94]`, not a zero-padded108-input facade.
- Nine surviving branches have physically narrower first convolutions.
-30 learned branches remain. The wholly removed branch is
  `px.close_loc/px.close_loc_2`, originally a single input.
- Its zero-input response is a frozen buffer of104 values, shape `[1,8,13]`.
  It is generated solely from trained weights and biases, not market data.
- The fusion retains31 slots:30 variable branches plus that deterministic
  response. The response is not zero and is not constant across temporal
  positions: measured RMS0.12428 and maximum within-channel temporal range
  0.19637. Replacing it with one scalar or its mean would be another model.
- Actor parameters decrease33,195 ->32,275 (920 fewer). These are ACTOR-only
  counts, not the previous99,586 total SAC loss parameters including critics.
- Core, head, subsequent branch layers and surviving first-layer coefficients
  retain their learned values. All compact parameters are frozen for inference.

No new market feature, normalization, clipping, interpolation or agent-state
feedback was introduced. The buffer preserves a learned deterministic
contribution that was already present when those inputs were masked.

## Trading detail: preserved compact actor

| Metric | TRAIN | VALIDATION |
|---|---:|---:|
| Net profit USD |21,018.69|1,044.71|
| Return % |210.1869|10.4471|
| Sharpe |3.6164|1.4496|
| Maximum drawdown % |9.8647|4.7276|
| Trades |1,827|298|
| Orders per market-week |8.7416|5.5185|
| Win rate % |64.3678|57.7181|
| BUY trades / wins |928 /585|136 /75|
| SELL trades / wins |899 /591|162 /97|
| BUY net profit USD |9,355.93|205.62|
| SELL net profit USD |11,662.77|839.08|
| Early closes |340|64|
| SL / TP / forced close |424 /1,010 /53|91 /122 /21|
| Commission USD |4,168.96|495.06|
| Spread USD |2,791.34|334.09|
| Slippage USD |2,145.43|290.74|
| Financing USD |374.12|36.48|

Early-close diagnostics are unchanged: VALIDATION31 helpful and33 harmful,
mean regret0.43017 times initial position risk, against passive SL/TP protection
of the same position. This is not a full-account continuation comparison.
TRAIN185 helpful and155 harmful; one additional proposal was preempted by
execution ordering. Compaction does not repair these remaining policy weaknesses.

## What the interventions establish

Removing the empty branch response changes247 of30,234 categorical decisions
and lowers VALIDATION return by1.07109 percentage points. It therefore matters,
but is not a sufficient explanation of the fresh-fit gap: the ablated inherited
actor still returns+9.3760%, versus-1.2114% for fresh training.

The fresh actor disagrees on8,426 valid windows. Its VALIDATION signal counts
are NOP5,848 / BUY163 / SELL196, versus4,926 /738 /543 for the preserved actor.
It is substantially less active. This is a measured behavioral difference,
not proof that entropy, one feature or one branch caused it. Reinitialization,
on-policy data collection into replay, critics and optimization trajectory
all changed; identical raw data and seed do not make those trajectories equal.

Uniform256-window activation samples from each period show exact masked/compact
core summaries. Fresh-model core RMS is0.50696 TRAIN and0.48297 VALIDATION,
versus0.76227 and0.76898 in the inherited model. Features in separately trained
hidden spaces are not aligned; these summaries are descriptive, not a causal
attribution or a reason to rescale activations arbitrarily.

## Artifact and reproducibility

`app/frozen_actor_compaction.py` extracts the effective TorchRL functional actor,
projects surviving weights, preserves constants, and implements authenticated
save/load. `tools/run_frozen_actor_compaction.py` runs all six arms sequentially,
replays historical controls, collects full native accounts and checks reload.
No critics, optimizer or replay buffer are migrated. The bundle is explicitly
`frozen_actor.compact.v1`, `training_supported=false`.

The bundle contains `actor.pt`, `CONFIG.json`, `PROBE.npz`, `MANIFEST.json`
and `RELOAD.json`. CONFIG records ordered94 inputs and projected fixed TRAIN
scaling separately from the actual graph, which retains the source fusion
layout. Preserve the original eligibility support when comparing experiments.

```python
from app.frozen_actor_compaction import load_compact

actor, manifest = load_compact("<compact-bundle>", device="cuda")
# x: correctly ordered/scaled float32 [batch,13,94] on the same device.
logits = actor.surface(x)  # categorical NOP, BUY, SELL; no training needed
```

Independent CPU loading was also checked on the saved256-window probe: max
cross-device logit difference2.38419e-6, zero action changes. This is NOT a
full CPU account qualification or a claim of cross-device bitwise equality.
Exact full-account qualification applies to the tested original GPU backend.

Evidence identity:
`1482144c7a0603c57581bf17676388917685d2118c9722518b91d043b5833d95`.

| Artifact | SHA256 |
|---|---|
| RESULT.json |`2fc356b9d76a236628f97d460a6954ea630639f74d8679236e5cbae957799e1c`|
| COMPLETE.json |`c8796f487f779784c7832c054e1442a8b92047ce22744fc5083e09f16b9961ed`|
| bundle/actor.pt |`acd99f1937b8cc968968936b7dae2c521ebea7c8e8588157014fa469de580060`|
| bundle/MANIFEST.json |`2e61e3d9b67ad334d7fe8c2be4f41b75ed0d09f6cab4574f4759a057f03cb7ac`|

## Next decision

Keep the compact actor as the preserved inference reference for further paired
selection diagnostics. Do not discard the learned function just to obtain a
smaller input tensor: this test demonstrates that those are separate operations.
If subsequent work needs continued SAC training, first define and test critic,
target-network and optimizer migration or an explicitly new training protocol.
No new training/search queue was silently started. All six diagnostic arms and
the artifact export are complete. The requested Ruff executable was absent in
this checkout; no lint-clean claim is made.

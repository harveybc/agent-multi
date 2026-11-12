# Actor head8 versus16

Owner approved2026-11-12, following309. Isolated actor capacity diagnostic,
not a joint SAC retraining and not automatic model promotion. Preserve299.

## Frozen Protocol

- Same71 inputs,28 groups,13 observations stride2, fixed TRAIN normalization,
  data/support, attention, branch/core weights, business controller and costs.
- Actor hidden width8 vs16. Critic head width stays8 in both arms. Explicit
  `actor_head_width` config prevents accidentally doubling the critics.
- Two paired seeds19/43. Each pair uses the same corresponding calibrated300
  critic, frozen temperature and identical TRAIN minibatch order. Both actor
  heads are freshly initialized. Original299 remains a separate reference.
- Cache frozen encoder activations and calibrated min-twin Q values. Fit only
  on all24026 valid TRAIN2020-2023 rows;2024 is not sampled for gradients.
- Objective is the discrete SAC actor objective: expectation of
  `alpha * log_probability - Q`. Critic and alpha are fixed. No oracle-label
  substitution, extra state feedback, new regularizer or zeroed input.
- Fixed20000 AdamW updates per arm, batch128, learning rate3e-5, weight decay
  0.01, gradient norm cap10. This is ~106.55 passes of TRAIN samples, not20000
  market epochs. No per-update account simulation or validation early stopping.
- Loss/probability/Q diagnostics at0,64,256,1024,4096,10000,20000 updates.
  Fixed final endpoint for trading comparison; do not pick a curve checkpoint
  retrospectively. If loss has not stabilized, propose an equal extension for
  both widths, never extend only the losing/winning arm.
- Full chronological TRAIN/VALIDATION native accounts once at the final
  endpoint, plus source reference. Record buy/sell counts, wins, closes,
  drawdown, Sharpe, net return and activity. >2 orders/week required for
  eligibility. Report all seeds, no zero-order winner, no automatic floating.

## Tests And Acceptance

1. Changing actor width never changes frozen encoder/critic/temperature values.
2. Actual shapes and parameter counts differ; JSON survives save/reload.
3. Cached and native actor outputs/greedy actions reproduce; no stale tensors.
4. Cached critic values originate only from the authenticated300 checkpoint.
5. Identical row order/schedule for paired widths, complete repeated TRAIN coverage.
6. Fresh optimizer per arm; source checkpoint and all non-head tensors unchanged.
7. Cold export reproduces full-support logits and account actions; source control
   agrees with sealed299 accounts before trusting the contrast.
8. Each arm is independently sealed and resumed; failure is explicit, not skipped.

This experiment tests whether a wider fresh actor head better fits the frozen
critic's action surface and improves account behavior. Its Q targets describe
the preserved policy: a changed actor may make them stale. A successful result
requires subsequent own-policy critic refresh before longer RL refinement.
Low loss does not establish good trading, and a negative result does not prove
the inputs lack signal or that core capacity is sufficient.

Run on CPU with cached features: the head has only1699/3395 parameters and
GPU transfer/scheduling is unnecessary. Existing GPU STL queues continue.
The finite queue performs preparation, four arms, exports, cold verification,
curves and summary without intermediate approval. No live deployment.

## Execution

Runtime `23dd44b0`, enabled restartable user service
`am-head-capacity-20261112.service`, started2026-11-12 at15:40 Bogota.
Owner-local run root: `/var/tmp/sac71-head-capacity-20261112-r1`.
The queue verified source native TRAIN/VALIDATION parity before training.
The first head8 arm completed its20000 updates, accounts and cold export;
head16 of the same seed started automatically. No GPU job was interrupted.

Focused verification:61 tests pass,5 CUDA-specific cases skipped in the
CPU-only test process. Tests include native engine fixtures, cached versus
native head outputs before/after an update, unchanged critic/encoder state,
actual width/parameter differences and inference bundle reload.

## Completed Results

All four arms completed in143.49seconds total, approximately28seconds each
including native accounts and cold inference. CPU caching avoided repeated
encoder/critic evaluation during80000 total head updates. Service exit0, no
restarts. Identity `8c5fe1c5366d8e3e958724c7fadd52b6be2c473a7f617c35f78822b348850672`.

| Seed | Head | TRAIN return % | VAL return % | VAL Sharpe | VAL orders |
| --- | --- | ---: | ---: | ---: | ---: |
|19|8|-77.249852|-10.723399|-1.295536|362|
|19|16|-82.682300|-13.509080|-1.469622|545|
|43|8|-74.964723|-28.279608|-3.489107|520|
|43|16|-72.537961|-25.750921|-3.120510|501|
|Preserved299|8|16.058224|18.369946|2.690712|207|

All four endpoints meet the >2orders/week activity rule; none meets the
nondecreasing source-performance gate. Preserved299 is not a freshly trained
capacity control. The paired fresh8 and16 arms are the capacity comparison.

At20000 updates, matching the frozen critic's greedy action improves from
74.30% to79.66% for seed19 and78.04% to78.35% for seed43. The minimized SAC
actor objective also improves with16 in both pairs. Trading does not improve
consistently: validation delta16-minus8 is-2.785681pp and+2.528688pp.
These are two paired diagnostics, not evidence of a statistically established
capacity advantage. Loss still improves near the endpoint; neither arm is
declared fully converged.

Cache versus native logits: maximum difference0 for all four arms. Cold full
support logits: exact for all four. Every non-actor-head state tensor remains
exact, including frozen critic/encoder/temperature values. Four bundles,
optimizer states, seven-point loss curves/PNGs and complete native account
ledgers are retained. There is no per-epoch validation curve because this was
a fixed-endpoint diagnostic, not validation-selected joint SAC training.

Disposition: retain299, do not widen the core or start another unchanged
floating pass on the strength of this result. Better imitation of a frozen
critic does not guarantee better account behavior. Next investigate frozen-Q
ranking reliability/coverage and policy drift on TRAIN native continuations
before another actor reset. The existing calibrated critics saw113 FIT anchors;
extrapolating their values to24026 TRAIN windows is a limitation of this
contrast. It is not evidence that the71 inputs lack trading information or
that a16-unit head cannot work with better-calibrated/updated critics.

Evidence: [RESULT.json](generalization/head_capacity_20261112/RESULT.json).
The next cumulative return still includes all49 currently absent phase2
features. Real-feed/COT qualification and STL continue on their separate lanes.

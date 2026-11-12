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

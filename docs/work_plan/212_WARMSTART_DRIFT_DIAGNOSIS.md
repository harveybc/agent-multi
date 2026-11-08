# Warm-Start Drift Diagnosis

## Question and Invariants

The70-column projection already reproduces the masked76-column policy and its
native accounts. The issue is degradation after further learning, not inability
to reproduce the initial policy. Source is the eligible epoch0 of run
`d4b260fffb17e0fcaddb0fe1c833493f30e28e942ffda4fbb36e5c3404a5f6e2`.
The initial validation return is13.80849%,206 orders. After six epochs of
head-only updates validation has35 orders, return-3.31670%, and mean signal
entropy0.23226. Reduced activity is a symptom, not yet a causal explanation.

Keep70 columns,28 groups,13 observations/stride2, all TRAIN2020-2023 and
VALIDATION2024 rows, fixed scaler, native accounting and strategy unchanged.
No TEST access. Never overwrite the source or promote a diagnostic checkpoint.

## Ordered Tests

| ID | Hypothesis | Intervention / observation | Evidence and interpretation |
|---|---|---|---|
| D0 | Projection or frozen weights drifted | Authenticate epoch0 and completed recovery snapshot; compare tensors and full accounts | Retained representation must remain exact; standalone replay must reproduce the initial account |
| D1 | Actor follows inappropriate inherited Q values | Same initial weights, TRAIN transitions and minibatch schedule; actor-only vs critic-only vs joint | Track Q action advantages, twin disagreement, TD residual, actor KL, entropy and NOP share after0/1/16/64/256 updates |
| D2 | Frozen low temperature contributes to activity collapse | Joint fixed-alpha vs joint adaptive-alpha; same initial alpha and all other settings | Measure alpha, entropy, action shares and account activity; no alpha jump imposed |
| D3 | Step size destroys a good policy | Joint learning rate3e-5 vs3e-6, otherwise paired | Parameter movement, policy KL and full accounts; this tests step size, not optimizer-history continuity |
| D4 | Short operator result fails over a complete trajectory | After D1-D3, choose a mechanism-specific full-epoch contrast with predeclared settings and seeds7/19/43 | Dynamic replay and account evaluation; short fixed-replay probes alone do not establish a durable remedy |
| D5 | Frozen representation or critic recalibration prevents adaptation | Only if D1-D4 support it: critic warmup with actor held fixed, then controlled release; separately critic-core unfreeze | Preserve initial actor; isolate critic calibration from representation freedom |
| D6 | Restart semantics or reward objective mismatch | If preceding contrasts do not explain drift: restored head optimizer moments vs fresh, replay age/composition, reward decomposition and position-dependent action semantics | Check parameter-name mappings and reward/discount units; do not silently add agent-state inputs or change rewards |

The six initial arms are `no_update`, `actor_only`, `critic_only`, `joint`,
`joint_adaptive_alpha`, `joint_lr_div10`. All start from the same saved weights
and fresh optimizer state, matching the current refinement. Their common4096
transitions are collected on TRAIN with the unchanged stochastic epoch0 policy.
The first4096 transitions are an update diagnostic, not a replacement training
dataset. Every trading evaluation still covers the full TRAIN/VALIDATION periods.
Each arm receives the same recorded128-row minibatches; indices and batch content
are hashed. Dropout RNG is reset identically for each arm. No-update is evaluated
twice and must retain identical state and accounts. Representation weights must
remain exact in every arm. The standalone zero-update replay must match the
previously saved account, including order counts.

Probe summaries use512 evenly spaced valid windows per period for Q/distribution
diagnostics only, never for training or replacing full account evaluation.
TD probes compare the independently computed soft TD0 target with TorchRL.
All metrics include activity, return, Sharpe, win rate, drawdown, direction,
close reasons and costs through the native account evaluator. A falling TD loss
without maintained trading utility is not a successful repair.

## Test-First Acceptance

| Requirement | Regression / runtime gate |
|---|---|
| One intervention per arm | named parameter trainability and LR assertions |
| No-update is a true control | all state tensors and repeated accounts exact |
| Frozen parameters stay fixed | byte equality after updates |
| Actor-only cannot change online critics | one-update counterexample |
| Critic-only cannot change actor | one-update counterexample |
| Train-only replay | collect from TRAIN environment only; no validation observations in batches |
| Common minibatches | saved schedule and tensor digest shared across arms |
| No false source attribution | source runtime/file hashes and recovery snapshot identity checked |
| No silent partial completion | atomic per-point results, terminal manifest only after all arms pass |

## Allocation and Automation

Primary RTX5090 continues the authorized head-only refinement unchanged, retaining
eligible epoch0. The RTX4070 runs D0-D3 sequentially under a separate supervised
service; this does not alter the live worker. The RTX4090 is currently occupied
by the R0 seed19 independent replication, not by this diagnosis. Do not claim it
is idle or interrupt that control. Assign D4 paired full-epoch arms to available
workers after the mechanism evidence is reviewed; D4 is not launched blindly.

Initial diagnostic ETA is measured from its first completed arm. Budget is six
arms,256 updates per learning arm, one shared replay collection, full accounts
at every diagnostic checkpoint, and no automatic hyperparameter sweep. Preserve
all failures and results. A hypothesis is supported only by its paired measured
effect; the first stage is single-seed engineering evidence, not replicated proof.

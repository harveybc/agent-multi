# Critic Blocked Stopping Contrast

Owner authorizes launch after317. Preserve actor299, its71 inputs and frozen
encoders; no state feedback, actor updates, STL, feature changes or promotion.

## Contract Before Execution

The old targets for2020/2021 run until2022 ends. Splitting those rows by year
alone would leak stopping-period rewards into the fitting labels. Reconstruct
the exact source-policy account states, then fork each action with native
terminal settlement at its own calendar-year end. Preserve the source state,
calendar, costs, risk rules, features and continuation policy. This changes
the diagnostic target explicitly; compare stopping versus fixed4096 on these
SAME new targets, never against317 as if only stopping had changed.

Use all239 authenticated anchors. The original146 FIT anchors are the only
ones eligible for fitting or stopping. The existing55 held anchors never
choose an update or regularization; report them descriptively after choices.
Require feature-window start within its year, and label terminal strictly
before the next block's earliest feature-window start. Record every exclusion.

| Fold | Fit (original FIT only) | Stop (original FIT only) | Subsequent audit |
| --- | --- | --- | --- |
| A | 2020 | 2021 | all2022 anchors |
| B | 2020-2021 | 2022 | all2023 anchors |

The folds overlap across runs. The original actor, scalers and encoders were
already trained on2020-2023, and anchors have been inspected. This is an
internal critic diagnostic, not independent forecasting confirmation.

Eight sequential fits: two folds x seeds19/43 x weight_decay0.01/0.1.
Same six heads averaged without position routing (shared_mean3), MSE loss,
LR3e-5, uniform FIT sampling, batch128, clip10,4096 maximum updates.
Same initial tensors and minibatches for the regularization pair.

Evaluate FIT and STOP every64 updates. Minimize STOP action regret of the
minimum twin; patience20 checks, improvement strictly greater than1e-8,
first eligible checkpoint64 (step0 is only a baseline). Freeze the stopping
decision irreversibly when patience expires. Continue that same trajectory to
4096 solely for the paired fixed-budget control, not to revise the decision.
Cold-reload chosen weights before audit. No audit metrics are evaluated until
both stopped and fixed checkpoints are fixed. No best-of-audit selection.

Report stopping check, selected update, FIT/STOP/audit metrics and action
counts; compare source actor, constant NOP and FIT-mean controls on the same
targets. Report all eight contrasts and both folds, never promote from them.
Useful evidence requires lower audit regret in both seeds in both folds at
the same regularization, with no worse executable-opening regret. Individual
positive cells remain diagnostic, not a new actor or trading return.

## Test Design And Traceability

| Requirement | Verification |
| --- | --- |
| No label overlap | Reject terminal/window crossing; bounded native suffix ends before next year |
| Source unchanged | Native parent state, source checkpoint/tensors and full TRAIN actor logits exact |
| Correct terminal target | Compare full-boundary suffix with existing native result; perturb post-boundary prices without changing bounded return |
| Honest stopping | Strict best-so-far, tiny delta, patience reset, no reopening decision after stop |
| No held selection | Fold constructor permits only original FIT indices for FIT/STOP; audit computed after selection |
| Fair contrast | Identical initial weights/schedules and fixed-endpoint continuation per pair |
| Reproducible inference | Cached encoder parity1e-6, no copied hooks, exact cold-reload predictions |
| Autonomous recovery | Per-anchor authenticated cache, per-arm seals, sequential service, failure/progress JSON |

Next permitted implementation: unit helpers/tests, native regression, then
launch the bounded panel reconstruction and eight GPU fits in one service.

Implementation and25 focused CPU tests pass, including native terminal
equivalence, future-price perturbation and restoration checks. CLI smoke passes.
The2022/2023 suffixes already have the required terminal and may be reused only
after authenticating the same native state and parent collection. One anchor
per year is re-executed in all three directions as an exact witness.2020/2021
targets are recomputed. Each row is checkpointed for autonomous resumption.
The historical mixed-COT store is unchanged and is not live-qualified by this
diagnostic; the independent live-data work is outside this run.

## Completed Execution

Completed2026-11-13T18:29:17Z in61.89seconds, runtime`162c6447`.
Identity`6ce08d4d9150ef3879df9be6330a69ec13b1e3decaeb85dcb9a1a3538f6932e5`.
239/239 source states authenticated, no excluded anchors. Four yearly witnesses
reproduce all three native action continuations exactly. The eight stopping
decisions were independently reconstructed from their logged STOP curves and
match exactly. All eight cold reloads pass; actor logits, source tensors and
checkpoint remain unchanged. Parent and per-arm file seals verified locally.

| Fold | FIT | STOP | Audit | Executable-opening audit |
| --- | ---: | ---: | ---: | ---: |
| A | 49 | 49 | 59 | 38 |
| B | 98 | 48 | 60 | 40 |

| Fold | Seed | WD | Selected update | Stopped audit regret | Fixed4096 regret | Actor regret |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A | 19 | 0.01 | 832 | 0.297626 | 0.311319 | 0.168404 |
| A | 19 | 0.1 | 832 | 0.297626 | 0.295749 | 0.168404 |
| A | 43 | 0.01 | 384 | 0.238383 | 0.273249 | 0.168404 |
| A | 43 | 0.1 | 384 | 0.238383 | 0.273249 | 0.168404 |
| B | 19 | 0.01 | 1856 | 0.277413 | 0.252317 | 0.249439 |
| B | 19 | 0.1 | 1856 | 0.262798 | 0.252317 | 0.249439 |
| B | 43 | 0.01 | 64 | 0.267339 | 0.259326 | 0.249439 |
| B | 43 | 0.1 | 64 | 0.267339 | 0.257336 | 0.249439 |

Lower regret is better; these are action-choice diagnostics, not trading
returns. Stopping helps3/4 cases in A and0/4 in B. Neither WD passes the
predeclared both-fold/both-seed opening guard. No endpoint beats the preserved
actor on its matched audit panel. The strong WD does not consistently help.

Thus317's small-panel overfitting is real, but this particular chronological
stopping rule does not consistently repair generalization. Do not substitute
a retrospective best audit checkpoint or conclude that the market inputs
cannot support trading. Only49/98 anchors fit10194 head parameters here,
with temporal dependence and pretrained encoders; these diagnostics do not
establish sample sufficiency or isolate the representation's contribution.

Next review FIT coverage and effective representation rank before defining
one controlled sample-coverage/representation contrast. Preserve299 and the
71 inputs; no actor refinement, new floating pass or live promotion was
launched by318. STL stays paused.

Evidence: [all endpoints](generalization/critic_blocked_stopping_20261113/RESULTS.md),
[FIT/STOP curves](generalization/critic_blocked_stopping_20261113/CURVES.csv),
[plot](generalization/critic_blocked_stopping_20261113/STOPPING_CURVES.png),
[effective contract](generalization/critic_blocked_stopping_20261113/CONFIG.json).

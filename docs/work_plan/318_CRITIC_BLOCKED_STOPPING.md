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

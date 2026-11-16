# Plateau LR And Decision Alignment

2026-11-16. Owner approved continuing360's weight -> margin -> action -> native
account diagnosis and asked whether ReduceLROnPlateau could help. This document
records a proposed isolated scheduler experiment, NOT an executed training result.
Current359 workers and their frozen queues remain unchanged.

## Rationale

ReduceLROnPlateau adjusts learning rate, not the L2/AdamW weight-decay coefficient.
Static low LR from epoch0 (already in359) and dynamic reduction after a plateau
are different hypotheses. Smaller steps can reduce policy boundary crossings but
cannot correct a wrong critic ranking or an objective/greedy mismatch by themselves.
Stable negative return is not success, nor is reduced weight movement sufficient.

Source inspected: installed torch2.13.0+cu130 scheduler implementation and
[official API](https://docs.pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ReduceLROnPlateau.html).
The optimizer has actor/critic/temperature groups in that order. Standard
ReduceLROnPlateau updates ALL groups; an actor-only intervention must explicitly
preserve the other two. Do not silently lower temperature adaptation too.

## Approved Diagnostic Before Intervention

Continue on sealed weights, not a modified active worker. Use TRAIN to relate
actual parameter displacement to changed logits, action margin crossings, critic
rankings and native account effects. Use all admitted rows as well as the small
probe:360 showed equal probe flip counts at40/41 despite53 changed VAL signals.
Validation remains descriptive and is not used to fit a new action threshold.

For each changed decision distinguish a signal emitted while flat, one while a
position exists, and a signal rejected by calendar/business constraints. Any
single-decision counterfactual replays the subsequent account; do not sum local
trade gains as if resource exclusion and later opportunities were independent.
Keep action/prior-position context in the controller/evidence, not new network
inputs. Compare exact original-device or qualified common-dispatch replays before
attributing outcomes to weights. Preserve both source snapshots and baseline.

## Proposed First Contrast

Same source weights, inherited optimizer moments, data, architecture, seeds,
activity gate, costs and epoch budget; fixed LR control versus actor-only plateau
scheduler. Reuse existing compatible control only after identity checks; otherwise
a matched control is necessary. Actor+critic reduction is a subsequent isolated
option, not an automatic enlarged sweep. No scheduler training launched here.

Provisional settings, not optimized values or claims of best configuration:

| Field | Proposal | Reason |
|---|---|---|
| metric | existing full validation return/week, mode=max | same selection objective; no second evaluation |
| factor |0.5| moderate reduction rather than immediately dividing by10 |
| patience |8|8 tolerated nonimproving observations; native torch reduces on9th |
| threshold_mode / threshold |abs /1e-8| honor tiny improvements; avoid relative thresholds on signed returns |
| cooldown |5 observations| time to observe the changed rate |
| actor minimum LR |1e-7| explicit bounded floor |
| critic / temperature LR |unchanged| isolate actor step size |
| warmup |20 completed epochs before counting scheduler plateaus| retain original checkpoint; do not make warmup snapshots ineligible |
| stopping |existing minimum100/max2000 and patience20| additionally allow20 complete epochs after an actual LR decrease, subject to absolute budget |

Budget extension after LR reduction is separate bookkeeping: never reset the
historical best metric or claim a new best simply because LR changed. Freeze exact
settings and counting semantics before execution. Source epoch0 is kept as the
historical best reference; do not initialize the scheduler's best to an inferior
post-warmup metric. An inactive/zero-order policy must not count as an improvement;
nonfinite metrics fail explicitly, not pass to the scheduler as valid observations.

No claim that these VAL-monitored training decisions are independent confirmation.
Report all seeds, selected and last checkpoints, return/trades/drawdown, action KL,
margin crossings, per-group effective LR and actual optimizer-update counts.

## Test Design Before Training Integration

| Requirement | Required evidence |
|---|---|
| Correct parameter reached | exact actor LR change and unchanged critic/temperature rates after a plateau |
| Patience semantics | synthetic signed traces,1e-8 absolute threshold,8 versus9 misses, improvement reset |
| No premature training stop | reduction buys its declared observation window without changing best checkpoint or exceeding hard budget |
| Warmup/activity | warmup does not overwrite baseline; zero-order metrics cannot reset best; NaN/inf rejected |
| Full-state resume | optimizer + scheduler best/bad/cooldown/last_epoch + stopping + RNG + replay yield identical LR/event sequence after restart |
| Cost unchanged | scheduler called once per completed epoch with already computed metrics; no tick-wise validation or duplicate TRAIN pass |
| Attribution | same seed/init/data/budget; observed performance gain required, not merely smoother weights |

Status: design recorded. Next allowed action is the approved bounded causal
decision-alignment analysis and scheduler contract tests, followed by a pinned
matched experiment after the existing359 treatment completes. No phase2 closure,
broad DEAP, live work, SNR work or automatic STL restart.

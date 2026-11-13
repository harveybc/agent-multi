# Critic Target Contract Contrast

Owner2026-11-13 authorizes continuation after316. Preserve selected299 actor,
71 inputs,49 cumulative return candidates and paused STL. No actor learning,
feedback features, strategy changes,2024/2025 inference or promotion.

## What Is New

Historical254/255 already tested external direction routing on a different
70-input model and bank-balanced counterfactual panel. It modestly improved
closing regret but hurt opening;139 same-direction conflicts remained. Do not
repeat that result as new evidence or claim that direction alone is sufficient.
This experiment uses current299/314/315 identities,146 native own-policy FIT
origins and the same55 held-out anchors. Its new question is whether marginal
target fitting is distorted by the diagnostic sampling distribution and by
Huber loss, compared with a conditional-output interpretation.

Freeze three factors, seeds19/43,4096 updates,batch128 (16 arms):

* Output: three heads per twin averaged without state, versus exactly the same
  initialized heads selected externally by native flat/long/short role.
  Same parameter count. Both networks consume ONLY frozen market embeddings.
* FIT sampling: uniform over146 diagnostic anchors, versus probabilities
  proportional to real source-policy decision counts in each2020-2022
  year/position-legality stratum divided by its sampled FIT count. Replay the
  native source trajectory to obtain counts, authenticate original account
  and snapshot identities; neither labels nor2023 counts set these weights.
* Loss: existing smooth-L1 versus squared error. Squared error targets the
  conditional mean in population; smooth-L1 targets a robust location, which
  need not be the mean. Finite samples/optimization do not certify either
  conditional expectation. Changing loss is an explicit factor, not hidden.

Encoders/actor/source critics remain untouched. Reuse frozen-encoder cache and
RoleHeads implementation from254; verify native Q reconstruction. AdamW uses
the current model's LR/WD, gradient clip10, identical schedules across output
and loss arms within seed/sampling. Source code, initial tensors, probabilities,
schedules, head checkpoints and cold-reload outputs are recorded.

## Evaluation

All endpoints at4096, curves at0/64/256/1024/2048/4096. Report each twin and
minimum on FIT, old38 monitor, shared55, by year/state. The primary paired
contrast is routed versus averaged heads at the SAME loss and sampling mode,
reported for both seeds; also report sampling/loss deltas without selecting
updates using monitor. The held panel is previously inspected and correlated,
not an untouched confirmation set. No population-performance claim from its
unweighted aggregate.

Also score the48 reachable states from316 (16 identical-market triplets), with
actual post-intervention direction, including immediately closed positions.
Use these ONLY as diagnostic evaluation, never FIT or training sampling.
All have original source continuations; changing subsequent policy changes
the target. These states are not a natural-frequency portfolio sample.

A descriptive result is promising for further design only if routed heads
reduce held regret in both seeds without worsening executable-flat regret at
the matched loss/sampling. Report all four combinations. Even a passing cell
does NOT automatically become the actor or strategy. Role routing still
omits entry price, barrier distances and age, as demonstrated in255.

## Tests And Automation

| Requirement | Test/evidence |
| --- | --- |
| Correct estimand | Explicit Huber/MSE factor, visitation weights depend only on TRAIN counts and FIT strata |
| No silent omissions | Reject absent positive-count strata, non-TRAIN counts, empty/overlapping FIT and held support |
| Controlled comparison | Same tensors, parameter count and minibatch digest for routed/mean arms |
| Real gradients | Role routing gradient-isolation and same-input/opposed-action positive controls from254 |
| Encoder cache | Native Q reconstruction within1e-6 and no source-tensor change |
| Protected model | All source tensors and full TRAIN actor logits exact before/after; checkpoint hash unchanged |
| Correct evaluation | Native held/counterfactual identities, fixed endpoints, explicit no-promotion |
| Automation | Immutable runtime, sequential resumable16-arm service, tables/JSON/PNG emitted automatically |

## Cache Preflight Correction

The first launch stopped before any head update: cached versus functional
critic Q differed by1.37e-6, exceeding the unchanged1e-6 gate. A read-only
reproduction separated two numerical effects: TF32 convolutions produced
about2.44e-4 discrepancy, and fused attention dispatch retained1.37e-6 even
with TF32 off. Disabling fused attention for BOTH cache paths with TF32 off
reduced twin discrepancies to3.87e-7 and4.17e-7, without changing source tensors.
The cache now scopes and restores these backend settings, including on errors.
Regression covers attention plus core dropout, source tensors, module modes
and backend restoration. The failed run is retained; restart uses a new
runtime and output identity. No tolerance was relaxed and no failed-arm
result is reused.

A second runtime inspection found the temporary capture hook was deep-copied
into the template, retaining detached embeddings on every subsequent head
call. The run was stopped before completion. Copying only AFTER removing the
hook fixes this unbounded memory retention; regression now checks that neither
the template nor any of the six trained heads inherits it. The complete
factorial is repeated in a third, separate output root. Partial results are
retained as diagnostic history, not mixed into the corrected comparison.

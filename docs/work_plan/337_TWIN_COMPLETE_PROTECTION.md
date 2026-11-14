# 337: Twin-Complete Decision Protection

Status: IMPLEMENTED_TESTED_BEFORE_EXECUTION.

336 reproduced all16 legacy rejections. Its new primal solver solved all48
linear systems but all16 finite-network proposals failed. The first independently
reconstructed proposal switches the chosen action's minimum twin on protected
row8. Subsequent relinearizations alternate twins. This is a nonsmooth minimum,
not a parameter ordering mismatch or just float32 roundoff. Audit every saved
system and finite trial before attributing the entire sample to that mechanism.

## Requirement And Construction

Let a be the protected action, b each alternative and j_b the alternative's
minimum twin at the current linearization center. Instead of differentiating
only min_i Q_i(a)-min_j Q_j(b), protect BOTH smooth differences
Q_i(a)-Q_j_b(b), i=0,1. If each is >=m in the finite network, then
min_i Q_i(a)>=Q_j_b(b)+m>=min_j Q_j(b)+m. This sufficient inner constraint
preserves the original decision even if either minimum changes branch. It is
not claimed to cover every feasible direction. Alternative branches are fixed
only for one linearization, not permanently. Final original guards still decide.

Ordering is row, alternative, chosen twin; margin repeats4 per protected row.
Use the336 primal solver unchanged. Keep2x interior margin,3 relinearizations,
8 scales,4x norm cap,all original guards and all data. Legacy default unchanged.

## Tests Before Execution

| Requirement | Test/evidence |
| --- | --- |
| Demonstrate old defect | Two affine twins, crossing chosen minimum: active-only derivative predicts safety but finite decision fails |
| Protect both twins | Exact analytic projection on that counterexample passes the unchanged finite margin |
| Correct derivatives/order | Autograd versus central finite differences, row/alternative/twin ordering |
| Sufficient constraint | Random twin outputs: all proposed smooth margins imply min-twin decision margin, even with changed alternative minimum |
| Preserve existing runs | Default repair remains legacy;336 exact controls reproduced |
| Native evidence | Same16 real minibatches, weights, optimizer and schedule;33 seals and independent witness audit |

## Bounded Automatic Execution

Repeat the same16 paired proposals from334, with legacy control and new
twin-complete primal protection. Compare with sealed336 active-only primal
results. No actor/encoder/input changes. All proposals and optimizer states are
saved; rejection must restore both. Frozen stable-added scoring mask unchanged.

If at least12/16 new proposals pass original finite guards, have consistent
stable-added actions across execution views, do not worsen stable-added regret,
AND lower full-FIT cost, run paired legacy/new256-attempt continuations from the
same334 endpoint automatically. This is an update-feasibility gate for a bounded
learning test, NOT a decision-improvement claim. No longer extension. Otherwise
finish diagnosis. The criterion is frozen before measuring337 results.

Cold replays and three resumed transitions required. Record native decision
regret, cost, MSE, acceptance, and held internal TRAIN stop/audit as diagnostics;
never fit held labels or change coefficients using them. No2024/TEST reads.
Publish positive or negative outcomes; no automatic model/live promotion.
Source299 with71 inputs,28groups and49 cumulative return candidates is unchanged;
STL stays paused. Do not restart an unchanged floating pass on this diagnostic.

Prelaunch:52 focused CPU tests pass in3.96s.336 witness audit confirms all384
finite rejections implicate row8 and chosen-twin switching; all48 saved
Jacobians and right-hand sides reconstruct exactly. Implementation is opt-in
(`--twin-complete --previous-solver <sealed336>`); legacy controls and default
behavior remain unchanged. Same raw proposals/optimizer states must match336.

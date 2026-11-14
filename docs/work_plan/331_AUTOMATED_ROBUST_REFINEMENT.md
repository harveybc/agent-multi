# 331: Automated Robust Refinement

Status: COMPLETE_AUTO_EXTENSION_REFUSED_DIAGNOSIS_CONTINUED.

## Execution Authority And Scope

Continue the authorized refinement diagnostic without intermediate user
approvals. Keep source299,71 features, actor and encoders unchanged; STL remains
paused. No2024/TEST access, live deployment, or automatic model promotion.
The chain has explicit computational and scientific stop conditions, not a
request for the user to supervise every transition.

## Frozen Chain

1. Replay the64 independent proposals authenticated in330, using the same
   original actions, numerical margin1e-5*max(1,abs(Q)), norm cap4x and eight
   backtracking scales. Aim at2x that margin for projection, without changing
   acceptance. At most three linearizations per proposal. Reproject the original
   AdamW displacement at each updated linearization point, not an accumulated
   correction with an unbounded norm. Verify actual finite GPU32/CPU32/NumPy64
   margins, actions and original322 regret bound. Retain every refused attempt.
2. If each of the four stalled starts has at least one accepted update that
   improves expanded-FIT MSE in both evaluations, run eight sequential256-step
   arms: four stalled starts x tangent330 versus interior repair. Identical
   expanded-FIT batches, LR, decay, optimizer moments and fixed protection sets.
   Accept/reject by the robust decision guard, not loss. On rejection restore
   weights AND optimizer; on acceptance retain the original proposed AdamW
   moments, as in329. Batch position advances for every attempt.
3. Automatically extend both arms to1024 attempts only if, at256, all four
   interior arms have robust endpoints, lower expanded-FIT MSE in both GPU and
   NumPy, at least32 accepted steps each, and at least three improve expanded-FIT
   action regret by>1e-8 in both implementations. Otherwise produce the bounded
   diagnosis and finish, with the exact gate result recorded. Never extend
   merely because a GPU is available.
4. Cold replay every saved endpoint and verify serialized resume transitions
   before emitting the final report. No extra human approval between stages.

The fixed actions come from the native starting GPU at8192, including legacy
numerically fragile starts. A refused step can retain that legacy starting
state; it must never be called a newly accepted robust state. Each accepted
state must satisfy the three-implementation guard. Sequential endpoints report
their own guard explicitly, even when no update was accepted.

Budgets64/256/1024 are attempted optimizer steps, not epochs. There is no new
sample, shorter input window, discarded feature, or changed target. Stop/audit
panels do not decide any transition. This is head-refinement feasibility, not
end-to-end SAC performance or a trading claim.

## Test Design

| Requirement | Required evidence |
| --- | --- |
| Interior target does not loosen acceptance | Known near-boundary fixture fails tangent and passes interior fixed guard |
| Nonlinear correction is bounded | Nonlinear fixture needs relinearization; norm cap and three-iteration limit |
| Actual AdamW transaction | Exact proposal/moments, full rollback on refusal/exception, resume parity |
| No hidden zero-motion acceptance | Zero displacement fails explicitly |
| No implicit approvals or unbounded extension | Deterministic stage gates, failed/unknown input refuses extension |
| Durable execution | Per-unit lock, atomic checkpoints, sealed completed stages, restart identity |
| No actor/data mutation | Parent seals and source hashes before/after |
| Verifiable learning | Saved expanded-FIT MSE/regret, all acceptance traces, cold replay |

The implementation, test evidence and runtime identity will be recorded before
launch; results and exact terminal state will be appended after completion.

Prelaunch:33 focused CPU tests pass in3.52s; CLI help passes. Fixed-action
preservation also enforces the starting action regret itself, a bound no weaker
than322. This does not alter the acceptance action set or numerical margin.

## Completed Chain

Runtime9483e0bb, identity
`c29171b88ffb4c63b5809ecb54e02cd574b2af821c39d01c21489051565c242b`.
Completed in130.47s, supervisor exit0, zero restarts. The repair accepted64/64
independent proposals and automatically started all eight sequential arms.
All accepted repairs used the first linearization; no claim of measured benefit
from repeated relinearization is supported by this run.

| Case | Tangent accepted /256 | Interior accepted /256 | Initial expanded regret | Tangent regret | Interior regret |
| --- | ---: | ---: | ---: | ---: | ---: |
| A19 | 0 | 253 | 0.225747 | 0.225747 | 0.224079 |
| A43 | 233 | 256 | 0.223158 | 0.220613 | 0.215787 |
| B19 | 237 | 256 | 0.229040 | 0.229215 | 0.231432 |
| B43 | 177 | 256 | 0.247916 | 0.248623 | 0.248615 |

Interior accepts1021/1024 steps (99.707%), tangent647/1024 (63.184%). The
original-FIT actions and fixed margin remain robust at every accepted state.
A19 tangent retains its legacy starting state, fails the margin and is not
counted as an accepted robust endpoint. All four interior endpoints pass.

Full expanded-FIT MSE falls in every interior arm, but decision regret improves
only in A19/A43 and worsens in B19/B43. Consequently the predeclared3/4 decision
criterion fails: no1024-step extension was launched. This is not a runtime
block or a request for user approval. The chain completed the cold/resume
checks, emitted the gate and proceeded to diagnostic332 without another prompt.

## Verification

Independent verification authenticated73 seals, all64 feasibility endpoints
and16 sequential checkpoints:80 serialized endpoints, zero local CPU/NumPy
guard failures for accepted states. NumPy metric max difference2.22e-16.
2048 sequential attempts and1668 accepted optimizer increments reconcile.
Internal cold replay:16 exact endpoints; serialized resume:24 exact transitions.
Source actor, encoder,71 inputs and validation/TEST untouched.

## Evidence

-[Results table](generalization/automated_robust_20261114/RESULTS.md)
-[Configuration](generalization/automated_robust_20261114/CONFIG.json)
-[All outcomes and curves](generalization/automated_robust_20261114/SUMMARY.json)
-[Independent verification](generalization/automated_robust_20261114/INDEPENDENT_VERIFY.json)
-[Cold/resume verification](generalization/automated_robust_20261114/REPLAY.json)
-[Learning curves](generalization/automated_robust_20261114/CURVES.png)

The numerical stall has a measured repair in these cases. That does not repair
the mismatch between numerical regression loss and discrete decision quality.
Diagnostic332 separates those effects using the already saved weights.

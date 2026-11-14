# 331: Automated Robust Refinement

Status: IMPLEMENTED_AND_UNIT_VERIFIED_BEFORE_RUN.

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

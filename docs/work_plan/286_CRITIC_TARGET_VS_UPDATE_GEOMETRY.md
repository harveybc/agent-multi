# Critic Target Versus Update Geometry

2026-11-11. Continue after285 with the qualified absolute19 actor,69 inputs,
28 groups, unchanged features, frozen TRAIN scaling and full native accounts.
The285 floating cycle is complete; no identical screen is repeated here.

## Question And Frozen Design

284 translated the critic's requested FIT changes more accurately without a
joint trading improvement. Separate target estimation from update geometry:

| Target | Head AdamW | Damped functional projection |
|---|---|---|
| Frozen current critic |critic_adamw|critic_project|
| Native frozen-policy continuation, FIT only |return_adamw|return_project|

Two update-schedule seeds19/43 give eight arms, all starting from the same
qualified absolute19 bundle and its same frozen critic. These are optimizer
schedule replications, not independent critic fits or independent market samples.
No critic training, reset, new model inputs or broker changes. The source is285,
not the earlier279 actor. Record at0/1/8; endpoint8 is fixed in advance.

Reuse the151 outcome-independent anchor timestamps declared in282, but replay
the NEW source policy and rebuild each position/state and all three native
continuations. Never reuse old-policy returns or positions. The counterfactual
first action is NOP/BUY/SELL followed by the frozen current policy, to the end
of the same role interval: FIT2020-2022 or MONITOR2023. This is an offline RL
return diagnostic, not ZigZag supervision, an ideal exit, or a live future input.
It is not a full-TRAIN actor fit; all full TRAIN/VALIDATION data are evaluated.

Only FIT realized reward returns replace the critic values in return arms.
MONITOR returns are diagnostic only; validation does not generate targets or
accept updates. Keep native alpha and all other actor-loss terms unchanged.
Native greedy returns and a soft/stochastic critic can differ by objective as
well as estimation error, so a target contrast alone does not prove a code bug.

Both geometries use identical seed schedules, batch128, eight updates and the
prespecified284 trust limits/backtracking. Four head tensors train; encoders,
critics, temperature and buffers stay exact. Do not select intermediate steps
or retune budgets after reading results. Cache finite substeps with receipts;
run sequentially on the primary GPU then independent account attribution and
authenticated collection. No automatic promotion or further training.

## Acceptance And Test Design

| Requirement | Test / evidence |
|---|---|
| Correct source |285 qualification receipt, saved endpoint and model hash, full source account replay|
| Fresh native targets |Rebuild state at fixed timestamps; all453 continuations preserve parent, stay in role interval, HOLD equivalence holds|
| FIT-only replacement |Unit tests perturb MONITOR outcomes without changing training values; reject unsupported target kinds/nonfinite FIT values|
| Same experiment except target/geometry |Same source,Q,schedules,alpha,budgets; explicit per-arm target and critic hashes|
| Translation |Existing Jacobian/projection/duplicate-weight/trust tests plus measured residuals, movement and acceptance counts|
| Frozen components and export |Exact fixed state/buffers, canonical cold logits and full accounts at1/8|
| Economic consequence |All16 endpoint accounts attributed chronologically, including direction/activity/wins/DD/early-close harm|

If even native FIT targets fail, examine transfer from FIT anchors to held-out
states before increasing capacity. If native targets help but predicted targets
do not, prioritize critic objective/ranking on the current policy. These are
conditional next actions, not claims about the result or absence of input signal.

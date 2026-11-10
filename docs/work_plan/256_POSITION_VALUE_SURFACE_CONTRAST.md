# Position-Evaluated Value Surfaces

## Owner Authorization And Scope

Continue255 with a tested market-only output interface that can express
different close/hold preferences for different positions in the same market.
This is Monte Carlo critic calibration on authenticated native continuations,
not a new actor, price predictor, feature selection run or live deployment.
Keep original70-input actor, encoders, risk profiles, calendar and costs intact.

Reuse all866 comparable same-direction banks from255, without excluding hard
examples. Reconstruct their actual positions through the same native delayed
entries and authenticate archived state/market identity. Labels remain the
archived discounted account continuation: forced first action, then the same
frozen greedy source actor until quarter termination. Fit2020-2022, monitor2023;
source representation previously saw2023. No2024 or TEST evaluation.

## Output And Controller Contract

The network accepts only existing market embeddings. Each twin emits two
directional fields of CLOSE-minus-HOLD advantage. No position coordinate is
concatenated into any learned layer. Controller queries are computed from the
native current liquidation quote L, direction d, current causal sigma s>0 and
the existing order protections:

`u = d * (L - SL) / s`, `v = d * (TP - L) / s`.

Their sum is positive but either distance can be negative after a price/spread
move. Do not clip these values. Positive advantage means request opposite
signal to close; ties within1e-8 mean hold. Native barriers/calendar preempt
requests exactly as before. This does not estimate, replace or move SL/TP.

Four arms, seeds19/43 (eight sequential runs):

| Arm | Nodes per direction | External evaluation |
|---|---:|---|
|scalar|1|Select direction, no geometry|
|mean81|81|Mean of all nodes, no geometry; equal-size control|
|progress9|9|Linear interpolation at u/(u+v), knots0..1|
|surface81|9x9|Bilinear interpolation at (u,v)|

The progress arm tests the earlier relative-position interface, but reruns it
on this corrected panel and frozen encoder; it does not reuse old results.
Surface axes are9 equally spaced knots between each coordinate's fit-only
minimum and maximum, separately by direction. Constant axes fail explicitly.
Outside support, extend the endpoint segment (no clipping), flag every such
query, report in/out-of-support metrics. No production admission of extrapolated
decisions follows from this diagnostic.

Each head is Linear(existing embedding,8), SiLU, Linear(8,2*nodes). Twins
retain their respective frozen source embeddings. All heads start fresh.
Surface81 and mean81 share exact initialization and minibatches. Fit SmoothL1
on the gap for both twins,4096 updates,batch128,AdamW LR3e-5,WD0.01,gradient
norm cap10. Sample each fit bank equally, then its states equally. The mean of
the two learned gaps is the primary decision; twins/disagreement are reported
but not described as calibrated uncertainty. Record checkpoints at
0/64/256/1024/2048/4096; fixed final endpoint decides the comparison, no sweep.

## Correctness Before Learned Performance

1. Synthetic interface tests: mirror long/short geometry, different absolute
   distances at identical progress, affine/bilinear reproduction, extrapolation,
   invalid coordinates, role isolation, and shared inputs with opposed actions.
2. Native reconstruction and first-action repeat controls, one per quarter;
   preserve exact parent state and reference account traces/cash.
3. For each same-market bank, solve a least-squares nodal oracle using its
   retrospective gaps. This is ONLY an expressivity test, explicitly including
   monitor witnesses, never a trained model or selection score. Report residual,
   sign failures, conditioning and coefficient magnitudes, both float64 nodal
   reconstruction and the actual float32 GPU controller. An inaccurate
   least-squares oracle is not proof that no alternative coefficients could
   reproduce action signs. Never change knots after observing these results.
4. Require the255 opposed-choice witness to be evaluated explicitly. A learned
   failure is reported, not repaired using that witness's future labels.

## Metrics And Harmful Closes

Primary: bank-balanced decision regret on the unchanged account-return gap.
Also report gap RMSE, sign accuracy, requested closes, and actual native agent
closes. Independently replay the ORIGINAL position only under first CLOSE or
HOLD followed by the frozen actor until that position exits; compute the net
USD difference. Count harmful/beneficial/neutral effective closes using this
local trade outcome. Local trade benefit and whole-account continuation benefit
can disagree; report both without changing the objective or summing isolated
counterfactual deltas into a purported portfolio profit. Always-close and
always-hold controls use the same rows. No new trading-profit claim.

## Traceability And Acceptance

| Requirement | Test/evidence |
|---|---|
|Market-only learned layers|Forward signature and same-embedding equality|
|Native geometry, no clipping|Mirroring, endpoint extrapolation, coordinate receipts|
|Fit-only knots|Monitor perturbation cannot alter fitted axes|
|Correct interpolation|Exact affine/bilinear reconstruction and gradient tests|
|Unchanged targets|Receipt hashes, temporal terminal boundary and replay controls|
|No actor/encoder updates|Full-TRAIN actor parity, tensor/file hashes|
|Fair capacity control|Identical81-node initial tensors/schedules, count parameters|
|Reproducible outputs|Saved heads/config/axes, exact reload, per-row predictions|
|Early-close harm visible|Separate native original-position counterfactuals|
|No babysitting|Exclusive lock, sealed resumable stages/arms, automatic collector|

Both interface correctness and useful learned behavior are required before
proposing actor transfer. These two coordinates may still omit funding, size,
age or account opportunity costs. Do not promise complete sufficiency, and do
not add those inputs silently if the present contrast fails.

# 347: Matched Divergence Replay

Status: COMPLETE_INDEPENDENTLY_VERIFIED.57 focused tests and scoped Ruff pass.
Automatic CPU verification, cross-host CPU check and saved-block analysis pass.

Explain346's B19 FIT and A43 internal-stop deficits relative to342, using
fixed saved endpoints. No new search, training budget, actor, input or model
selection. Reproduce the original targeted searches only as exact historical
transactions. Stop at64. Internal stop labels are descriptive only and must
never enter a loss, projection or selection of an update.

Replay both arms from their common334 source. Require every receipt and saved
checkpoint state/optimizer/RNG to match. Record actions before/raw/accepted
for all endpoint-discordant rows in the named partition. Locate first action
disagreement and the first control transition to its final superior action.
At each such control transition after16, run the four combinations of control
and treatment pre-step weights and optimizer states on the identical batch,
using only the ordinary protected update. Diagonal combinations must reproduce
the original transaction exactly; repeat every intervention cold. These are
diagnostic one-step interventions, not valid continuations for promotion.

Decompose the preferred-versus-other action margin into actual AdamW raw motion
and protection motion. Report manual Jacobian projections of old momentum,
current gradient, decay and float rounding separately from finite differences.
Weights include prior trajectory and influence the current gradient; optimizer
exchange is a conditional intervention, not a universal optimizer-reset result.

|Requirement|Test/evidence|
|---|---|
|Exact bounded replay|256 transactions, saved8/16/32/48/64 where applicable; source seals unchanged|
|No held-label optimization|Only ctx.fit supplies batch/loss and protection; stop rows used in reporting|
|Correct event selection|Unit cases for early discrepancy, later control crossing, ties, missing crossing and invalid sequence|
|Factorial interpretation|Unit numeric effects with known interaction; missing/nonfinite cells rejected|
|No optimizer fabrication|Whole saved state exchange; same batch/hyperparameters, exact diagonal replay|
|Finite versus derivative attribution|Manual NumPy forward/Torch parity, Adam reconstruction and margin telescope|
|Reproduction|Each counterfactual repeated from its unchanged weights/optimizer/RNG; local CPU artifact check|
|No promotion|Actor299 bundle unchanged,71 inputs/28 groups/49 return candidates preserved; STL paused|

Implementation uses existing341 protection,345 historical rescue, actual AdamW
transaction and334 support. A separate evidence root owns all new outputs.

Supplemental saved-weight analysis: enumerate all16 combinations of the four
critic blocks at each fixed pre-step witness. Use exact Shapley attribution of
the control-preferred action margin, with independent Torch64 forward checks
and sum-to-total identity. This explains the local margin difference, not the
cause of the entire trajectory. Hybrids are not admissible replacement models;
no original/optimal guards are bypassed in training. A separately sealed CPU
report consumes the completed347 verification; no running artifact is edited.

## Results

Producer:a139afee. Identity:
`1577e5d8897e3dcdfce5c1ffb5aab97afa0dbeadff432c2875c91f31cc14cd9a`.
Finished in344.58seconds, no restarts. All256 historical transactions and18
saved checkpoints reproduce EXACTLY in weights/optimizer/RNG. Eight fixed-step
factorial cells repeat exactly; both unchanged diagonals reproduce original
transactions. No new search budget was spent: historical345 searches were
re-executed only to authenticate their original trajectories and receipts.

|Case / partition|First disagreement and control acquisition|Panel row|Control action|Treatment action|Native meaning|
|---|---:|---:|---|---|---|
|B19 / FIT|33|449|0 NOP|2 SELL|Avoid opening a worse short while flat|
|A43 / internal stop|64|55|1 BUY signal|0 NOP|BUY closes the already-open short; NOP holds it|

`open_short` is the panel's EXISTING-position category, not an instruction to
open a new trade. Row55's native first effect is command3, one AGENT_CLOSE,
no entry. This is a missed beneficial EARLY CLOSE, not merely a changed logit.
Row449 NOP is better than SELL by0.548683174 native discounted-reward units,
but BUY is still its best labelled action. Do not call this a globally optimal
decision. Row55 close exceeds hold by0.350705455 in the same target units.
These are fixed-state native continuation outcomes, not new whole-account
profit results or executed actor orders.

## Weight Versus Optimizer Intervention

Rows below use the pre-step weights from C(control) or T(treatment) and the
COMPLETE optimizer state from C or T, including moments and bias-correction
counter. The first letter selects weights, the second selects optimizer.
Only an ordinary protected update is applied, never a new targeted search.

|Case|CC final action|CT final action|TC final action|TT final action|
|---|---:|---:|---:|---:|
|B19|0|0|2|2|
|A43|1|1|0|0|

For these two fixed-step witnesses, switching weights changes the outcome;
switching optimizer state does not recover the missed decision. This does
not establish that optimizer history is universally irrelevant, nor that a
weight transplant is valid. It isolates the conditional effect at these steps.

**B19:** preferred(NOP)-SELL margin before the step is-0.0000133667 in C,
versus-0.0001525841 in T. In C the raw AdamW proposal makes it WORSE by
0.0000308493; protection then improves it by0.0000610070 and crosses to NOP.
In T the raw step helps by0.0000104521, but protection subtracts0.0000021603;
the margin remains negative. T's old-moment contribution is positive here,
so blaming or resetting momentum alone is unsupported. Current-gradient
contributions oppose the preferred margin in both states.

**A43:** close-hold margin before the step is-0.0000417792 in C versus
-0.0062901217 in T. Both raw update and protection move toward closing, but
the larger pre-existing deficit in T remains. Its final margin is-0.0061305312
versus C+0.0001387092. Old momentum helps closing in both states; the current
gradient opposes it. This is not a protection-induced reversal at this step.

## Which Blocks Differ

All16 C/T block combinations were evaluated for each witness (32 total).
Exact Shapley contributions below sum to the complete pre-step margin
difference T-C; independent permutation enumeration verifies the identity.
Hybrids are explanations of this margin, NOT qualified replacement networks.

|Block|B19 contribution|A43 contribution|
|---|---:|---:|
|Twin0 hidden|0|-0.0017858052|
|Twin0 output|0|-0.0014862890|
|Twin1 hidden|-0.0000347669|-0.0017043203|
|Twin1 output|-0.0001044505|-0.0012719280|
|Total|-0.0001392175|-0.0062483425|

B19 is dominated locally by twin1's output block (about75% of the margin
deficit); A43 is distributed across all four blocks. A zero contribution here
does not mean a block is globally unused. The relevant function is min-of-two
critics and contributions depend on the particular row and coalition.

## Verification And Correction

Independent Torch64/manual NumPy Jacobian error<=4.44e-16 and forward
error<=1.95e-16. Original-device transition/state comparisons remain exact.
The first cross-host diagnostic report check rejected a maximum1.39e-16
floating difference because it demanded byte identity for derived algebra.
The local verifier now explicitly uses absolute1e-12 for diagnostic floats,
exact structure/actions/booleans, with negative regression tests. It records
the actual maximum error and verifier/helper hashes. This does NOT relax any
training acceptance gate, checkpoint parity, replay condition or action check.
The original report and sealed run remain unchanged. Supplemental block
report was produced with cf37d1eb; its32 Torch64 checks differ by<=1.95e-16.

## Decision And Next Step

Do not reset AdamW or transplant a single layer based on this evidence. The
original actor299 and71 inputs/28 groups/49 return candidates remain unchanged.
The demonstrated issue is a conditional weight-space/margin displacement; its
full historical cause cannot be reduced to a single block from this audit.

Next bounded contrast: test FUNCTION-LOCALITY of targeted corrections on
saved FIT witnesses. Among corrections satisfying the SAME original finite
guards, nonzero motion, cost and regret improvement, penalize unnecessary
changes to other FIT action gaps instead of assuming small parameter motion
means small decision disturbance. Define/test the objective before execution;
compare with existing targeted corrections, keep actual-target success and
surrogate loss separate, and reject candidates that violate existing gates.
Use only FIT observations/labels to construct or choose the repair. A43 stop
is a diagnostic witness, NEVER an extra protected training row or tuning
target. Report it descriptively after a frozen decision rule. If no feasible
repair exists, report that outcome rather than widening guards. No long fit,
actor promotion or unchanged floating pass follows automatically from347.

[Results](generalization/matched_divergence_20261114/RESULT.json),
[original verification](generalization/matched_divergence_20261114/VERIFICATION.json),
[cross-host verification](generalization/matched_divergence_20261114/LOCAL_VERIFICATION.json),
[block coalitions](generalization/matched_divergence_20261114/BLOCK_RESULT.json),
[run identity](generalization/matched_divergence_20261114/CONFIG.json),
[block report identity](generalization/matched_divergence_20261114/BLOCK_CONFIG.json).

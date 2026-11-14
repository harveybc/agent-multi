# 347: Matched Divergence Replay

Status: IMPLEMENTED_READY.52 focused tests and scoped Ruff pass. Automatic
independent CPU verification follows the GPU replay and repeated interventions.

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

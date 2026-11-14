# 340: First Harmful Crossing Replay

Status: COMPLETE_VERIFIED.52 focused tests pass; scoped Ruff passes.

## Results

Producer `e9c93759`; independent verifier `6f5268ed`.
Identity `880655269f69277c4dc3d69e47520a00917abcab142d8c25152bc8259bdadae1`.
All1024 transactions exactly reproduce338, including complete receipts.
All8 original endpoints have identical weights,optimizer and RNG. Replay took
228.03s; final cold verification took3.50s. No new candidate or actor update.

Evidence: [event table](generalization/crossing_replay_20261114/RESULTS.md),
[CSV](generalization/crossing_replay_20261114/EVENTS.csv),
[stage plot](generalization/crossing_replay_20261114/STAGES.png),
[full result](generalization/crossing_replay_20261114/RESULT.json),
[independent checks](generalization/crossing_replay_20261114/VERIFICATION.json).

| Measurement | Result |
|---|---:|
| First harmful row/case crossings |10|
| Raw AdamW proposal already changes to the harmful action |6|
| Raw proposal keeps the old action; protection changes it harmfully |4|
| Actual sampled-batch gradient opposes the better-action margin |5|
| Full-FIT gradient opposes that margin |4|
| Affected row absent from that batch |5|
| Cold replay of proposal,gradient,optimizer,accepted state,RNG,receipt |10/10 exact|
| CPU/GPU32 single/batch action agreement with64-bit attribution |10/10 events, all3 states|

Counts are descriptive, not independent samples.9 of these changes persist
at256; A/19,row317 is corrected later, explaining339's9 terminal occurrences
versus this replay's10 crossings. There are10 harmful transition occurrences
total in these four replays, with no repeated harmful crossing of a row
within a case. No account profit or validation improvement is claimed here.

### Concrete Cause 1: The Protection Can Displace Unprotected Decisions

A/19,step51,row245: BUY -> BUY in the raw update -> NOP after protection.
BUY-minus-NOP margin:0.000106854 ->0.000233381 ->-0.000182239.
The raw update helps this margin; correction subtracts0.000415620.
The affected row IS in the batch and the batch/full-FIT gradients both help.
Calling this merely bad minibatch sampling or a wrong loss gradient would be
incorrect: the finite corrective displacement is what crosses this boundary.

B/43,step152,row530 is an open SHORT: BUY means early close, NOP means hold.
Raw AdamW preserves BUY; correction changes it to NOP, reducing the native
target return by0.065351582. This isolates a lost beneficial early-close
decision inside TRAIN. It is not a new simulation of annual P&L.

The protector preserves the original49/98 FIT anchors, not every one of
the196/392 expanded FIT rows. All original guards pass. Thus this is a
coverage limitation of the acceptance contract, not a violated guard or
an accounting defect. Removing the protector is not justified: the raw
update may break the original anchors, and6 crossings originate before it.

### Concrete Cause 2: Optimizer History Can Overwhelm The Current Gradient

A/19,step10,row316: current batch gradient and full-FIT gradient both favor
the BUY-minus-NOP margin. At the actual AdamW denominator, old first-moment
numerator contributes-0.000199051; current gradient contributes+0.000143856.
Decay is-5.12e-8 and rounding-3.93e-9. Raw AdamW crosses to NOP. Protection
helps slightly (+0.000012303 margin), but does not recover BUY.

These are algebraic contributions using updated second moments, not a
counterfactual optimizer with momentum removed. They establish how this
actual proposal arose, not that resetting AdamW universally helps.

### Concrete Cause 3: Sample Interference Is Sometimes The Direct Driver

A/43,step238,row257: both sampled-batch and full-FIT gradients oppose the
BUY-minus-SELL margin (-0.410109 and-0.287215). The actual raw update changes
BUY to SELL; protection reduces the harm but cannot restore BUY. This is a
different mechanism from the protection-induced examples above. In5/10
crossings the affected row was absent from the sampled batch, but absence
is not a sufficient explanation: other harmful crossings include the row.

### Numerical Checks

Independent Torch64 per-row/full-FIT/batch derivatives agree with analytic
derivatives to6.94e-16. Float64 reconstruction of float32 AdamW has maximum
parameter discrepancy5.27e-8. Actual proposals and optimizer states replay
EXACTLY on the original CUDA device class; the rounded reconstruction is
not substituted for the original step.

B/43,step88 has a specific minimum-twin branch discrepancy: batch position71,
row8,NOP. CUDA values `[1.9660900831,1.9660905600]` select twin0; float64
values `[1.9660904134,1.9660904057]` select twin1. The unadjusted analytic batch
gradient difference is0.000448728. Recomputing the64-bit derivative on the
OBSERVED CUDA branches reduces the maximum difference across all ten
witnesses to2.72e-8. The action itself agrees across all checked geometries.
This numerical effect is explicit; it is not used to explain away the four
protection-induced crossings or the measured optimizer-history contribution.

## Next Bounded Correction Test

Use the saved ten proposals as controls. Keep the original protected-anchor
contract and add TRAIN-only protection for currently target-optimal decisions
on the expanded FIT panel. Do not freeze every prediction: wrong actions must
remain correctable, and equally optimal actions must not be arbitrarily ranked.
Check executable finite decisions after projection, not just linear margins.

First test feasibility and useful nonzero updates on the fixed witnesses,
reporting rejected/zero-motion cases explicitly. Measure expanded FIT regret,
original guards and loss separately; a lower loss alone does not pass.
Only if this succeeds should a matched bounded continuation compare the
new contract with338. Keep held labels out of update choices. No new actor,
long fit or floating pass is released by this diagnosis. This next test is
documented, not yet executed;340 itself is complete and no workers remain.

## Frozen Design And Traceability

Replay all four complete338 trajectories from334 through256 using the original
batch schedule, optimizer state, CUDA device class, primal solver and complete
twin plus single/batch guards. No new algorithm, actor update, feature change,
held-label gradient or2024/TEST access. This is authenticated replay, not a new
candidate. Record first harmful change of every expanded FIT row, not only339's
nine harmful terminal occurrences. Preserve all original artifacts.

## Acceptance And Traceability

| Requirement | Test / evidence |
|---|---|
| Same trajectory | Every transaction receipt exactly equals338; weights,optimizer,RNG at64/256 equal saved checkpoints |
| Causal stage attribution | Saved before/proposed/accepted weights and actual clipped batch gradient; finite margins and local tangents separately |
| Optimizer decomposition | Reconstruct AdamW proposal from updated moments plus weight decay; separate old/current numerator contributions at the actual updated denominator |
| No fake momentum ablation | Label decomposition as algebraic attribution, not a counterfactual optimizer without momentum |
| Shared gradient | Own multiplicity in actual batch plus other examples; compare full-FIT gradient and role contributions at crossing |
| Numerical witnesses | NumPy64 and Torch32 actions; analytic derivative versus actual CUDA batch gradient, with minimum-twin branch differences reported |
| Frozen sources | Parent seals and source actor hash before/after; witness files independently reloadable |
| Nontrivial regression | Known AdamW recurrence; duplicate-batch contribution accounting; accepted projection can reverse unprojected margin; tampered replay rejection |

Do not infer the whole training cause from a source tangent. The measured
step is a local causal intervention with fixed source weights and batch. A
protective projection can help or hurt an unprotected row without violating
the original protected contract. Quantify, do not assume that it is the cause.

Sequential unattended execution: four resumable case units with exclusive
lock, progress records and failure evidence. A failed unsealed case restarts
from its authenticated source; sealed completed cases are reused. Replay
receipt mismatch fails closed. No historical result is overwritten.

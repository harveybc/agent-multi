# Fixed-Budget Temporal SAC Exploration

Owner approved2026-10-20: replace frequent expensive early stopping for the
current minimal Conv1D exploration with fixed100000 transitions and full paired
inference at25000,50000,75000,100000 only. No patience,metric-driven stop or
best-checkpoint restoration. Final model is100000 regardless of intermediate
validation performance. This does not establish that overtraining is absent.

Reuse the exact causalConv1D2/4/hidden[8],15 selected features,13 observations,
stride2,seed7,TRAIN2020-23,VALIDATION2024,costs,leverage50 and business constraints.
TEST,refit,DOIN,expert labels and architectural changes are excluded. New identity
and fresh native SAC training; no donor/warm start without complete replay/RNG
continuation. Preserve interrupted earlier checkpoints with partial scope.

Four monitor pairs cost approximately25min from the measured375-381s/pair,
instead of126min for20 pairs. The existing native terminal TRAIN/VALIDATION
reporting path also runs its final inference pair; disclose that extra cost,
do not silently redefine it as another selection pass. Total provisional
budget60-80min including training and terminal reports,subject to measured
progress. Every evaluation is frozen-policy inference,not refitting.

## Requirements And Tests Before Implementation

| ID | Required structural and behavioral evidence |
| --- | --- |
| FB01 | explicit approved settings/schema; old early-stop configs unchanged; reject mismatched protocol/extra scientific overrides |
| FB02 | exactly4 monitor checkpoints; bad/missing/duplicate steps fail; no patience stop or hidden best-weight restore |
| FB03 | degraded final metrics still select100000; incomplete training cannot claim final fixed-budget result |
| FB04 | real native temporal monitor preserves parent policy,RNG,replay and optimizer; terminal callback leaves latest weights unchanged |
| FB05 | independent curve replay validates artifact/support/score and explicit final selection; partial curves never claim final100000 |
| FB06 | deterministic machine-readable curves and PNG plots with separate roles,daily-unit gap,validation risk/activity and descriptive deterioration flags; source bytes unchanged |
| FB07 | plots refresh only after new complete pairs,before native close; closed plots stay immutable and are covered by existing collector hashes |
| FB08 | isolated pinned runtime,new generated units/GPU admission,old worker/timers stopped,no concurrency,warehouse readback and second physical copy |

Plot interpretation: TRAIN improving while VALIDATION deteriorates,especially
across adjacent observations,is a possible-overtraining indicator,not causal
proof or a new automatic selection rule. Compare equal daily-growth units,
not four-year cumulative TRAIN profit with one-year VALIDATION profit.
Keep all complete validation dates; never choose a more favorable year.

Stage: requirements and unit/integration test design recorded before code.

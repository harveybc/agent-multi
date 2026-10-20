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

## Verified Deployment

Runtime `ec4855ce4337fc4dbe1cc9d72f05256b0999923b`; contract
`fc9c0c532d43d541df4023ab2eeeda8fa63396ea80b938476c84da53da62d68f`.
Requirements/tests above were recorded before implementation. Focused local
suite144/144 passed; actual worker25/25 passed in29.84s,1.4GiB peak with no
swap. The bounded GPU admission passed16 native updates and exact reload on
attempt1. Synthetic admission is not a financial result.

The new worker started2026-10-20T07:26:43Z and reached10000 transitions in
225.7434s. One heavy5090 worker,zero restarts,approximately1.7GiB process-group
RAM,686MiB compute GPU allocation at the check. No5070Ti parallel job.
Common-config comparison against the previous minimal temporal contract found
only `paired_early_stopping` changed; arms and selected input binding match.
The earlier run is STOPPED,not CLOSED: five full pairs retained and checksum
copied to the second physical disk. Do not combine its partial scientific
scope with the new final model. Fresh seed7 native initialization,not a partial
policy warm start without its replay/RNG state.

Full reporting/publication/readback,plots and closed-model collection run on
timers. The live contract has zero expected reports before its first complete
pair,not missing measured results. No re-approval or agent polling is required.
The overlay node key expired immediately after dispatch; existing-key-verified
LAN SSH restored worker access,collector and the warehouse reverse forwarding.
Warehouse `/healthz` returned200 through that route. No secret,authentication
policy,training process or scientific setting was changed for network repair.
The overlay credential itself remains expired and should be renewed separately.

SB3 calls `_on_step` before a transition's last gradient update. Therefore the
100000 observation is deliberately in `_on_training_end`,after the final
update; it authenticates the exact final policy without restoring older weights.
The first three checkpoints retain their native pre-update callback timing.
This timing is explicit,not a claim all four observations occur at identical
gradient-update offsets.

## Operator Status

Use the pinned runtime and private spec:

```bash
python -m tools.selected_rl_temporal_minimal status --spec "$SPEC"
```

Read the native attempt `progress.json` for current transitions. After the first
pair,`paired-monitor/diagnostics/CURVES.json` and `CURVES.png` are generated
automatically. Four monitor pairs plus native final reporting are the full cost;
provisional60-80min from startup is not a measured completion guarantee.
No unseen-TEST or profitable strategy claim is made.

## Traceability

| IDs | Implemented evidence |
| --- | --- |
| FB01 | strictv8 request and immutable fixed SETTINGS; matched-recipe check |
| FB02-FB04 | `test_fixed_budget_sac.py`; real-engine FIXED monitor integration; terminal update-order test |
| FB05-FB07 | independent fixed curve replay,forged-final refusal,plot idempotence and source-byte tests |
| FB08 | actual-worker25tests,GPU16updates/bitwise reload,systemd parser check,RUNNING/progress,two enabled publisher/worker timers,second-machine collector and network health |

## Final Result

Completed2026-10-20T08:36:51UTC,about70min from startup,100000 transitions,
four full monitor pairs,zero worker restarts. Selected final100000 by fixed
budget,never an intermediate validation winner. TRAIN2020-23:net-1454.77USD,
return-14.5477%,maxDD37.0186%,827 closed trades,44.8609% wins,PF0.9340,
daily nonannualized Sharpe-0.02002. VALIDATION2024:net-727.83USD,
return-7.2783%,maxDD10.7319%,203 closed trades,43.3498% wins,PF0.8326,
daily nonannualized Sharpe-0.06103. Native terminal inference matches the
100000 monitor metrics. The model remains unprofitable on both periods;
this is not evidence that temporal processing solves generalization.
TRAIN return across25k/50k/75k/100k was-52.64/-41.23/-30.46/-14.55%;
VALIDATION-16.81/-32.08/-22.64/-7.28%. Final improves both versus earlier
observations; the curve does not support a simple monotonically worsening
validation-from-overtraining explanation. It also does not exclude overfit,
insufficient learning,capacity limitations or distribution shift.

All10 reports delivered with readback,zero pending/mismatch;closed source
authenticated and574 files checksum-verified on a second physical machine.
The collector initially rejected this closed temporal arm because its paired
branch hardcoded SAC_FLAT. Four new positive tests reproduced the defect;
after allowing the exact contract's supported arm and enforcing the same arm
on both terminal tasks,60 collector tests passed,including mixed-arm rejection
and hash-tampering cases. Only collector code changed;no source model,metric,
runtime or closed artifact was modified. The archive is47MiB approximately.
Root STATUS is the retained pre-publication snapshot;CLOSE.json is the final
completion/readback authority. TEST was not consumed by this experiment.

Stage: COMPLETED_UNPROFITABLE_DEVELOPMENT_AND_PHYSICAL_COPY_VERIFIED.
No further scientific fit is running under this fixed-budget identity.

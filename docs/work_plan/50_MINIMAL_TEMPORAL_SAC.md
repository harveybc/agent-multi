# Immediate Minimal Temporal SAC Contrast

Owner2026-10-20 explicitly releases the pause on temporal SAC and prioritizes
this experiment over the pending flat-capacity sweep. Preserve interrupted
flat evidence; never relabel it as completed. No retired forecasting weights,
ZigZag expert, feature reselection, refit, TEST or DOIN.

## Frozen Recipe And Scope

Use the existing registered PerFeatureTemporalExtractor and native SB3 SAC:
15 independent causalConv1D branches,2 filters,kernel3; concatenate at every
time position; one causalConv1D core,4 filters,kernel3; retain13 time positions;
only then flatten52 learned temporal features and append unchanged business
state. Actor and each critic use hidden[8]. Actor and critic extractors are
independently trainable under native defaults; critic target copied natively.
No pretraining, dropout, BatchNorm, L2, pooling or right padding. A flatten at
the policy boundary is deliberate, not flattening raw input before extraction.
The two convolution stages have receptive field5 observed sampled positions.
The policy head can combine all13 output positions. No claim that Dense is
mathematically incapable of time-series prediction; this tests temporal bias.

Keep selected15 columns,13 observations,stride2 actual observed open-market
bars including the current completed bar, TRAIN2020-23/VALIDATION2024,seed7,
costs,leverage50,SL/TP,fractional sizing,one entry/day and session restrictions.
Monitor every5000 transitions,patience10,absolute min_delta1e-8,cap100000;
M=-0.5*(full TRAIN mean daily log growth + full VALIDATION mean daily log growth).
Do not change this objective or select a favorable validation year. The actor
still has a small Dense output boundary, not raw-data Dense-only processing.

## Traceability And Test Design

| ID | Structural / behavioral acceptance |
| --- | --- |
| TM01 | strict request_v7 permits only declared temporal architecture; other common settings identical to flat[8] except branch/core filters and arm/encoder |
| TM02 | one temporal arm accepted by existing queue; random/control multi-arm bypass remains refused; regularization still flat-only |
| TM03 | real paired monitor preserves temporal arm through child inference/publication/closure; native best restore preserves encoder tensors and optimizer; parent RNG/replay unchanged |
| TM04 |13x15 matrix,branch/core sizes; positive-weight prefix perturbation proves no future influence and current-bar sensitivity; all13 positions survive |
| TM05 | native synthetic updates,finite gradients,exact policy save/load; count actual actor/critic/target parameters, no synthetic count misreported as real-model count |
| TM06 | resource-gated pinned worker,finite attempts,outbox/readback,second-machine copy; no parallel heavy workload on shared-RAM host |
| TM07 | fresh scientific progress and paired full-role metrics; stop reason/best step/trace support retained; no claim of outcome from startup |

Stage: requirements and unit/integration test design before implementation.
Use existing runner, monitor, extractor and collector; extend arm binding only,
not SAC learning or the DOIN protocol. CPU and real-engine tests precede GPU
admission. Prefer5090; all paused flat workers stay stopped during this run.

## Verification And Deployment

Scientific runtime `96e19edb`; contract
`9cae5a6fb9f0b348efc42767bd05237dcd626450f76704a8ff4ac180b1aee10f`.
166 local focused tests passed, including all6 real-engine monitor profiles;
29 worker-environment tests then passed, including the actual temporal monitor
with parent RNG/replay invariance and native best-weight/optimizer restore.
The worker run took11.968s,1.4GiB peak,no swap. An earlier combined worker
check had a baseline-case timeout with130698 memory.high reclamation events,
zero max/oom/oom_kill events. The isolated test cap was corrected from
MemoryHigh1500MiB/Max2GiB to2500MiB/3GiB; the scientific6GiB cap did not change.
No timeout was reclassified as a passing test or financial observation.

The private matched-recipe check confirms identical feature binding, selected
input manifests and common config versus the flat[8] recipe, except temporal
branch/core widths4->2 and8->4 (ignored in the flat model). The arm/observation
encoder changes from flat to per_feature_conv1d. Hidden[8],kernel3,all data,
monitor settings,leverage,costs and business constraints are unchanged.

The flat capacity worker and all its successor timers were stopped by the
owner's new priority; its6 completed checkpoint pairs are partial evidence,
not a completed capacity study. The final publication pump retained those
reports. The partial models/traces were mirrored with a second checksum
comparison showing no differences; native SQLite WAL files were not copied.
No interrupted checkpoint is used as a donor for the fresh temporal fit.

Both generated services/timers passed the actual systemd parser. Temporal
worker,publisher and independent closed-artifact collector are enabled. The
synthetic GPU admission passed16 native updates and exact reload on attempt1;
queue arm SAC_PER_FEATURE_CONV1D is RUNNING with restart count0. Start check
does not establish financial performance or a measured completion ETA.

First real progress:5000 transitions in109.0488s before paired inference.
Effective config independently read back as15 features,window13,stride2,
branch2/core4,hidden[8],seed7,leverage50 and per_feature_conv1d. The saved
step5000 policy tensor file contains1062 actor parameters,1638 shared
critic-extractor/twin-Q parameters and1638 target-critic parameters. There
are2700 actor+critic trainable elements (plus the native entropy scalar);
target weights are not additional learned capacity. The actor first Dense
weight is[8,69]:52 temporally extracted values plus17 business-state values.
These counts come from the actual scientific checkpoint,not the synthetic
smoke. Matching hidden[8] does not mean matching total parameter counts.

Single-read status:

```bash
python -m tools.selected_rl_temporal_minimal status --spec <private-temporal-root>/SEQUENCE.json
```

At each completed checkpoint report full TRAIN and VALIDATION profit,return,
drawdown,daily nonannualized Sharpe,closed trades,win rate and profit factor.
Keep each period's date/support and cost assumptions explicit. The complete
[32,32] flat reference differs in capacity; only identical-step retained[8]
checkpoints form a same-head diagnostic, and their interrupted scope must
remain explicit. Do not claim an isolated temporal-architecture effect from
an unmatched comparison of selected checkpoints.

Temporal market processing is the primary current path; future flat agents
are explicitly labeled controls, not silently substituted primary models.
No claim of success, all-zero trading metrics or completed paired financial
result is issued before generated evidence exists.

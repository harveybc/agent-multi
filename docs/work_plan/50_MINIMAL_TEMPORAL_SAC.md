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

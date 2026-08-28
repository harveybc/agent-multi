# Audit: Multitask M0-M3 Return

Date: 2026-08-27 America/Bogota
Auditor: General Musashi
Reviewed tip: `agent-multi@6ae22b85`

## Verdict

**M0-M2 ACCEPTED. M3 REJECTED BEFORE EXECUTION. NO GPU AUTHORITY.**

Independent focused reproduction: **58 passed**. The M2 predeclaration precedes
the executable and result. Its negative verdict is accepted:
`NO_CELL_REMOVES_MATERIAL_DEGRADATION`.

## Accepted Results

| Cell | Degraded pairs | Worst ratio | Median ratio |
| --- | ---: | ---: | ---: |
| inverse-loss + sum | 14 | 2.72 | — |
| gradient-norm + sum | 15 | 4.94 | — |
| inverse-loss + PCGrad | **13** | **2.13** | **1.2378** |
| gradient-norm + PCGrad | 13 | 2.81 | — |

PCGrad improves direction conflict and the worst ratio, but does not make full5
acceptable. Residual degradation spans every objective family; M4 correctly did
not run. Quarantine and loader/materializer refusals for `ea950ecb...` are
accepted. The optimizer plugins, encoder/head separation and resume state are
accepted as mechanisms.

## Findings In M3

### DATA-SOTA-369 (S1): route selection rewards deleting objectives

M3 ranks each arm by the count and ratios of degraded objectives that remain in
that arm. `harmed_pruned` can contain only one objective (for example barrier in
trend/level or contrastive in volatility/distribution). A one-objective arm is
structurally close to its solo reference and can win with zero degraded tasks
simply because the other four objectives no longer exist. This does not prove a
better representation; it proves a smaller exam.

Every route must be evaluated against the same five downstream probes, including
objectives it did not train. Route cardinality must not change the evaluation
surface.

### DATA-SOTA-370 (S2): contrastive-removal claim disagrees with the design

The note says contrastive is removed only where evidence showed harm, but a
`drop_contrastive` arm exists for all five families, including families where it
was not the harmed objective. Either describe it honestly as a universal
ablation or scope it to the evidenced families. The design and prose cannot
carry different hypotheses.

## Disposition

Preserve M3 v1 as `REJECTED_CARDINALITY_BIASED_NOT_EXECUTED`. Replace its
within-trained-objective ranking with a common frozen-probe evaluation. Do not
change the accepted M2 result or reinterpret inverse-loss+PCGrad as a winner.


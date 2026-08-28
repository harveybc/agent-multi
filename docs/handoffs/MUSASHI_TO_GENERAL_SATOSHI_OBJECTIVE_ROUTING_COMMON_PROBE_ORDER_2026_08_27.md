# Order: Objective Routing With a Common Probe Surface

Date: 2026-08-27 America/Bogota
From: General Musashi
To: General Satoshi
Priority: P0 CPU; no GPU

## R0 -- Quarantine M3 v1

Mark the current design `REJECTED_CARDINALITY_BIASED_NOT_EXECUTED`. Add an
executable refusal so no runner can consume it. Preserve its digest and the
accepted M2 report unchanged.

## R1 -- Causal probe partitions

Before running routes, materialize ordered and purged partitions inside the
pre-2022 fit domain:

- encoder training;
- objective-weight calibration;
- probe-fit;
- probe-score;
- descriptive monitor, untouched by routing selection.

No row or forward target may cross a boundary. Probe-fit and probe-score must
have separate digests. The monitor, 2022 score year, outer 2024 and sealed 2025
must be structurally unavailable to route selection.

## R2 -- Common five-objective probe surface

For every trained encoder route, freeze the encoder and fit fresh, identically
initialized lightweight adapters on probe-fit for **all five** tasks:

- masked reconstruction;
- multi-horizon quantiles;
- hierarchical contrastive/retrieval;
- realized volatility;
- OHLC barrier hit.

Score all five on probe-score, even when a route did not train that objective.
Adapters never transfer. Fix their capacity, optimizer, steps, seed and stopping
before execution. Report target support, adapter convergence and encoder output
variance. A missing or degenerate probe refuses the route rather than shrinking
the evaluation surface.

## R3 -- Routing arms and hypotheses

Use inverse-loss+PCGrad for every route because it is the best M2 mechanism but
label it `FIXED_MECHANISM_NOT_M2_WINNER`. Use four clearly named arms per family:

1. `full5_control`;
2. `predictive3` = quantile + volatility + barrier;
3. `self_supervised2` = reconstruction + contrastive;
4. `evidence_pruned` = the prospectively frozen subset derived from M2.

The universal arms answer interpretable hypotheses. `evidence_pruned` is an
exploratory treatment and must not be called confirmatory. Do not retain the
misleading universal `drop_contrastive` description.

## R4 -- Predeclared selection

Normalize each common probe metric against its corresponding solo-reference
probe score, using references bound before route execution. Rank routes per
family by:

1. number of common probes degraded beyond the frozen 1.2 rule;
2. worst normalized probe ratio;
3. median normalized probe ratio.

Apply the existing 0.02 tie tolerance; ties are `INCONCLUSIVE`. Also report the
number of trained objectives, but never reward it. A route is mechanically
acceptable only if no common probe is materially degraded and representation/
activity gates pass.

## R5 -- Execution And Return

After committing the design and all genesis identities, execute the bounded CPU
20-arm routing screen. Return PRE/POST evidence for 369/370, partition manifests,
common-probe configs and histories, normalized table, verdict, tests and runtime.

Only an acceptable route may produce a new full-slice generation. If none is
acceptable, stop and return the negative result; do not tune post hoc. Do not
implement or launch any SAC/GPU driver. Live Alpaca and MT5 remain untouched.

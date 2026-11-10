# Partial Critic Unfreezing And Module Restorations

## Scope

Owner accepted the follow-up to249. Inspect critic weight drift and test
partial unfreezing without changing observations, targets, strategy or actor.
This continues phase2 refinement diagnosis, not another feature-removal pass.

## Pinned Contrast

Reuse the authenticated248 collection:1075 fit states (2020-2022),351 monitor
states (2023), all70 columns,13 observations at stride2. Identical native
quarterly labels, loss, initial calibrated critic, optimizer reset, learning
rate, weight decay,4096 updates, batch128 and schedules for seeds19/43.
Heads-only and full-representation controls are reused, not retrained. Verify
their effective settings, source identities, initial metrics and schedules.

Only `qvalue_network_params.module.encoder.core.1.*` and
`qvalue_network_params.module.head.*` may train. Pin the exact eight tensor
names and shapes before updates. The selected core is the second residual
TCN block, with two causal convolutions, width16, kernel3 and dilation2.
It contributes3136 parameters across both twins; heads contribute3398:
6534 total versus61450 in the full-critic representation arm.

All branches, fusion, attention, first core block, actor, target critics and
temperature stay frozen. Keep the diagnostic eval-mode forward convention
from248; do not add or change dropout here. No optimizer-state inheritance.
Actor full-TRAIN logits must be bit-exact at each completed arm.

## Weight-Space Diagnosis

For each full-representation endpoint, group online-critic parameters by
individual feature branch, fusion, attention, core block and head. Report
absolute and relative L2 drift per component. These magnitudes are not a
causal attribution of performance.

Individually restore each component to its initial calibrated weights, leaving
the other components at the full-representation endpoint, then rescore the
same fixed panel by state. Restore the original endpoint between interventions.
This is a controlled parameter intervention, not a trained candidate or proof
that effects add up. No restoration result changes the predeclared partial
scope or selects a checkpoint. All source files remain untouched.

## Test Matrix

| Requirement | Verification |
|---|---|
| Exact scope | Reject missing/extra core or head tensors and architecture mismatch |
| No accidental unfreezing | Assert only the eight pinned tensors require gradients |
| Gradient correctness | Real CUDA update changes last core and heads only; actor exact |
| Restoration isolation | Synthetic tensor test: only selected module changes; round trip exact |
| Temporal support | Revalidate archived role/window/terminal boundaries |
| Paired optimization | Archived minibatch hashes, source hashes and effective settings match |
| Reproducible inference | Archived initial and endpoint metrics reproduce within1e-6 absolute tolerance |
| Resume | Exclusive lock, config identity, per-seed sealed checkpoint and verified collection |

## Outputs And Decision

Save module drift/restoration tables, per-state fitting curves, final weights,
effective trainability map and comparison against archived controls. No early
stopping, monitor-selected checkpoint, actor updates or promotion. Report fixed
4096-update results for both seeds, including contradictory metrics.

The2023 monitor was excluded from these updates but was used by source
pretraining. Quarterly labels overlap within roles. These remain diagnostic
comparisons, not an independent final performance test. If partial unfreezing
fails, examine state-dependent value conflicts and label horizons separately;
do not conclude that the market inputs have no useful information.

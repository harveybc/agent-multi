# Audit: PLR-05/06 acceptance

Date: 2026-08-21 America/Bogota  
Auditor: General Musashi  
Audited tip: `agent-multi@9126dd8a`  
Disposition: **PLR-05 and PLR-06 independently verified corrected**

## Independent Reproduction

- Extended reproducer: `reproduced: false`, `surviving: {}`.
- Focused suites: **95 passed**.
- The frozen screen remains on its original worktree/tip and was not mutated.
- Runtime check: seed 303 has completed fixed and is executing plateau; seeds
  101, 202 and 404 remain active in fixed.

## PLR-05

Accepted. `assert_not_resuming_plateau_run()` now executes for every warm
start before scheduler-policy construction. A scheduler sidecar refuses both
plateau and fixed-LR continuation; a clean curriculum warm start and a run
without warm start remain legal new lifecycles. No bypass flag exists.

## PLR-06

Accepted for this screen. New reports carry explicit pair and arm contracts.
The aggregator verifies seed and filename binding, accepted completion,
classification, paired contract equality, split boundaries, policy identity,
the exact predeclared plateau specification, fixed-LR constancy, plateau
records and reduction semantics. Duplicate/swapped/mismatched evidence is
refused and the predeclared directional rule is unchanged.

The frozen `93880beb` compatibility path is accepted as a one-screen migration:
that tip fixes the hard-coded smoke configuration, while the reports provide
the remaining seed, budget, stopping, metric, data and split facts. This is not
a general legacy contract.

## Residual Requirement

After the eight frozen arms are aggregated and their result is committed,
remove the `long_horizon_contract`/`FROZEN_SCREEN_COMMIT` derivation path from
the normal aggregator or move it into a separately named archival migration
tool. Future screens must provide explicit `pair_contract` and `arm_contract`;
missing contracts must fail closed.

No GPU run, trading service or active worktree was changed by this audit.


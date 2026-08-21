# Audit: Satoshi PLR-01..04 corrections

Date: 2026-08-21 America/Bogota  
Auditor: General Musashi  
Audited tip: `agent-multi@8b76d5c2`  
Disposition: **PLR-02/03/04 accepted; PLR-01 and aggregation identity require correction**

## Reproduced Facts

- Upstream reproducer: `reproduced: false`, no surviving cases.
- Focused suites: **83 passed**.
- The active screen still executes from the frozen pre-correction worktree at
  `93880beb`; the correction branch has not mutated it.
- All four services remain active in the fixed-LR arm with real trading
  activity. Latest sampled epochs: seed 101 = 29, 202 = 23, 303 = 55,
  404 = 50.

## Accepted Corrections

- **PLR-02 accepted:** reports now identify the experiment as a bounded
  120/40/40-day scheduler screen and disclaim multi-year inference.
- **PLR-03 accepted in scope:** the predeclared directional rule is explicit,
  four seeds are not called conclusive, and runtime/thermal facts are excluded
  from causal interpretation. Counterbalancing remains required for a future
  confirmation.
- **PLR-04 accepted:** the 40-day surface is named `diagnostic_holdout` and is
  explicitly distinct from the sealed 2025 test.

## New Findings

### AUD-F1-20260821-PLR-05 (S3): interrupted plateau model can resume as fixed LR

The fail-closed sidecar guard is called only when the new configuration also
constructs a plateau controller. An interrupted plateau checkpoint with its
sidecar can therefore be supplied as `warm_start_model` while `plateau_lr` is
absent; the guard is skipped and training continues under fixed LR. That is a
resume with changed scheduler semantics, not an ordinary curriculum handoff.

Required correction: call `assert_not_resuming_plateau_run(warm_start_model)`
whenever a warm start is supplied, before selecting scheduler policy. A sidecar
must refuse regardless of the requested new policy. If deliberate conversion
is ever needed, require a separate explicit migration tool that records the old
and new contracts; do not add a bypass flag to the training CLI.

Acceptance tests:

1. sidecar + plateau config refuses;
2. sidecar + fixed-LR config also refuses in the executing pipeline;
3. clean warm-start checkpoint + either policy remains a new lifecycle;
4. no warm start remains unaffected.

### AUD-F1-20260821-PLR-06 (S3): aggregator does not prove paired-arm identity

The aggregator currently checks only that both reports share `data_sha256`.
It does not verify seed, config identity after excluding the scheduler factor,
split rows/timestamps/hashes, code/environment identity, fixed-arm absence of a
plateau contract, plateau-arm exact contract, successful completion, or the
bounded-screen classification. Mislabelled, swapped or otherwise non-paired
reports could therefore produce an authoritative directional outcome.

Required correction: each report must expose a canonical `pair_contract` and
`arm_contract`. The aggregator must require exact equality of every pair field
(seed, data and split identities, observation/reward/model contracts,
timesteps, stopping contract, initialization and code/environment pins), while
requiring the arm field to differ only as predeclared: fixed LR versus the exact
plateau specification. Refuse duplicate reports, swapped seed labels,
incomplete/nonaccepted arms, missing eligible checkpoints, unexpected LR
reductions in fixed, absent or inconsistent reductions in plateau, and any
extra factor difference.

## Execution Order

1. Do not stop, restart or mutate the eight active arms.
2. Correct PLR-05 and PLR-06 on the separate branch with adversarial fixtures.
3. Preserve the already-predeclared outcome rule unchanged.
4. Merge corrections only after the frozen screen completes.
5. Run aggregation only after independent reproduction of the corrected
   identity checks. Audit and aggregation work remain parallel to GPU compute.


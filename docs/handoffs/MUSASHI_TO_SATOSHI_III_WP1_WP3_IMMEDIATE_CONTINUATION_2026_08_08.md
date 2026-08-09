# Musashi to General Satoshi III: Immediate WP1-WP3 Continuation

Date: 2026-08-08 America/Bogota
Authority: execution of the already owner-approved document 38
New owner approval required: none

Read first:

1. `docs/work_plan/38_NEAT_LESSONS_L1_L2_CURRICULUM_AND_FEATURE_SELECTION.md`
2. `docs/audits/AUDIT_L1_L2_CURRICULUM_FEATURE_SELECTION_AND_STOPPING_2026_08_08.md`
3. `docs/audits/AUDIT_SATOSHI_III_TEACHBACK_WP0_AND_RUNTIME_2026_08_08.md`

The teach-back and WP0 corrections are independently accepted. Continue now;
do not return another plan-only response.

## Execution Order

1. Implement WP1 completely: typed nested-split materialization in the real
   `rl_pipeline_with_validation` path, exact derived row counts, causal
   `is_context_prefix` semantics with no actions/orders/account/replay/metrics,
   persisted split manifest, and fail-closed mismatch handling. Execute all
   eight required test classes from document 38.
2. Implement WP2 completely: paired split-level comparator and L1 stopping
   semantics from document 38. Ordinal/lexicographic decision keys must never
   be averaged into a scalar.
3. Implement WP3 completely: corrected Easy-to-Normal handoff with immutable
   identity, weights, replay and optimizer-state evidence, and the required
   isolation boundaries.
4. Run focused tests, integration tests and a clean-checkout reproduction.
5. Materialize and launch the four-cell seed-101 smoke automatically. Verify
   one domain, one genesis, one initial-population fingerprint, unique claims,
   all four workers and no parallel chain.
6. If the smoke passes, dispatch the approved factorial without requesting a
   redundant owner phrase.

## Status Contract

Every report must state separately:

- implementation work currently executing and its process/evidence;
- supervisor state;
- worker state;
- GPU process, utilization and temperature;
- active job/cell/candidate and ETA, or the exact dependency preventing one;
- completed and remaining cells;
- chain domain/genesis/tip/population identity.

Correct the current record: supervisors are active, workers are stopped, no GPU
experiment is active, and WP1 is queued rather than running. The next response
must contain implementation diffs/tests or an exact reproducible defect, not a
restatement of intent.

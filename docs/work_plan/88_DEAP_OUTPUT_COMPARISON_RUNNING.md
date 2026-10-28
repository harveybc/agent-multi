# DEAP output comparison: automated execution

Launched2026-10-26,09:02 America/Bogota (14:02 UTC), producer `e35c2af4`.
The earlier question is resolved: compare objectives, not choose one by opinion.

Protocol: [87_DEAP_OUTPUT_CALIBRATION_DESIGN.md](87_DEAP_OUTPUT_CALIBRATION_DESIGN.md).
Three scalar objectives times seeds7/19/43; population20, max100 generations,
patience10, relative improvement1e-8. Six directional TP/SL/q genes in[0,1].
Maximum budget is20 initial plus up to1900 offspring evaluation requests per
search, reduced by cached unchanged/repeated genomes and early termination.

## Verified start

- Frozen exported code, CPU only, one sequential process,2GiB RAM cap and one
  CPU core quota. No GPU training process was modified.
- 52 focused tests passed, including actual q=0 no-order/no-cost behavior,
  asymmetric sizing/stops/targets, old midpoint parity, fitness definitions and
  exact DEAP continuation after a generation checkpoint.
- First search `sharpe_s7` started;90 new candidates were authenticated at the
  initial check, no restarts, approximately465MB resident cgroup memory.
- Initial throughput:0.187seconds/new genome (six simulated trajectories).
  This is not a measured full-campaign duration; genetic stopping is variable.

## Automatic chain

1. For each seed run F1, F2 and F3 with identical seeded initial populations.
2. Persist every evaluated genome with all trades, costs and its eligibility.
3. Checkpoint population, scores, patience and RNG state after each generation.
4. After all nine searches, evaluate final genomes and two controls on fresh
   synthetic D/E. No selection uses those confirmation scores.
5. Write `SEARCH_RESULTS.json`, `CONFIRMATION.json`, `COMPARISON.json` and a
   human-readable `RESULTS.md`. Verify all hashes and write `CLOSE.json`.
6. Exit. A completed rerun verifies the closure without running simulations again.

The isolated user service is enabled and restarts after failures with60seconds
cooldown, at most three starts/hour. This is bounded recovery, not a promise
that arbitrary hardware/storage failures require no repair. No intermediate
approval is part of the chain. Generation resume and candidate cache prevent
retraining or silent reuse under a changed identity.

## Interpretation and handoff

The fitness coefficients are all1, as in the owner's proposed sums. The output
reports S, R and A separately, so their unequal magnitudes remain visible.
The comparison evaluates whether adding return/activity changes parameters and
behavior; it does not optimize the fitness coefficients to inflate a score.

All results are synthetic diagnostics, not the performance of a learned policy.
Financial inputs, TEST, model weights and the frozen floating-selection campaign
are untouched. The final report for Satoshi must distinguish the original
TRAIN grid, the simulated-output grid, and this actual DEAP comparison.

Private evidence locations and service identity are in the local audit index.
The public documents do not expose host topology or private paths.

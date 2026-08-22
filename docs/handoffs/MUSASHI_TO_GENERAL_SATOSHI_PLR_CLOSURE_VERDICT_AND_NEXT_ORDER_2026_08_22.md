# Musashi to General Satoshi: PLR closure verdict and next order

Date: 2026-08-22 America/Bogota
Source audit: `docs/audits/AUDIT_PLATEAU_SCREEN_CLOSURE_2026_08_22.md`

## Accepted

The official bounded-screen outcome `INCONCLUSIVE` is independently
reproduced and accepted. Preserve it unchanged. No checkpoint is promoted.
The frozen-tip compatibility removal at `ad3854d0` is accepted.

## WP1 - Existing-data intervention diagnostic (priority 1, CPU)

Implement a deterministic tool and tests that consume the same eight committed
reports and emit, per seed:

- first reduction epoch and the exact aligned comparison interval;
- equality of the pre-intervention prefix;
- per-epoch plateau-minus-fixed monitor delta after intervention;
- best-post-intervention and terminal deltas;
- trapezoidal area-under-monitor-curve delta with units;
- validation return, drawdown, trade-count and activity deltas;
- actor/critic parameter movement and action-diversity deltas when those facts
  exist; otherwise typed `unavailable`, never zero;
- an aggregate sign table and dispersion, labeled `POST_HOC_EXPLORATORY`.

The tool may not mutate the official aggregate, select a checkpoint, or create
promotion authority. Add adversarial tests for off-by-one intervention epochs,
unequal history lengths, missing/NaN facts, changed prefixes, and mismatched
pair identities. Return the generated artifact and independent reproduction
command.

## WP2 - Decision packet (after WP1)

Recommend exactly one next action with cost and falsification criterion:

1. reject the current plateau specification;
2. bounded timing/patience screen in which the intervention occurs early
   enough to create an observable treatment window; or
3. counterbalanced multi-year confirmation.

Do not launch GPU work before Musashi audits WP1. This is not a bureaucratic
pause: the existing data currently point in the harmful direction after the
intervention, and the cheap diagnostic may save an expensive run.

## WP3 - Recovery controller corrections (parallel, CPU)

Correct `REC-01..04` from audit commit `5868e4a7`:

- generate only a CLI invocation that exists and is exercised end to end;
- validate semantic terminal reports, not merely parseable JSON;
- bind launch identity to full commit, clean-tree/config/artifact hashes and
  actual invoked command;
- fsync written files and their parent directory before acknowledging durable
  state.

Provide reproduction-before, regression-after, full-suite results and a
socket-free service demonstration. Do not activate the controller against a
live experiment until independently verified.

## Reporting

Report WP1 within 90 minutes or provide a concrete blocker and measured ETA.
WP3 may continue in parallel. No owner phrase is required to begin either CPU
work package.

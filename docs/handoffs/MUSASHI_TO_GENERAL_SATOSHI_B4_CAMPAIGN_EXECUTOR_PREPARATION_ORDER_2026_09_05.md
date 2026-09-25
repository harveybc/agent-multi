# MUSASHI TO GENERAL SATOSHI: B4 CAMPAIGN EXECUTOR PREPARATION ORDER

**Date:** 2026-09-05
**Disposition:** execute CPU/offline preparation now; do not dispatch another GPU run
**Depends on:** `agent-multi@9fb017e3`, Satoshi return `5464e838`, and the reviewed runtime audit beside this order

## 1. Disposition of the returned preflight

`B4_GPU_PREFLIGHT_ACCEPTED_AFTER_INDEPENDENT_RUNTIME_AUDIT`.

The following facts are accepted for the single consumed attempt on `o2024_seed101`: CUDA execution, exact `20000` environment steps, `19872` real optimizer updates after `128` learning-start steps, `168.5 s` wall time, bounded memory and temperature, reproducible genesis, no scored or sealed rows, and no promotable or economic result.

Do not turn the observed `168.5 s` into a campaign ETA. The preflight did not include the full epoch loop, validation, checkpoint selection, outer-origin scoring, or final statistical adjudication. The next task is to make those missing stages executable and auditable before the owner is asked to authorize the remaining GPU work.

## 2. PRE: freeze the missing execution path

Before editing, freeze a call-path report showing whether any current executable consumes a B4 cell and performs all of the following:

1. repeated SAC learning segments under the materialized epoch, patience, and budget terms;
2. causal validation and checkpoint selection without reading the scored origin;
3. frozen-checkpoint evaluation on the cell's declared outer origin;
4. emission of per-bar net returns under the reviewed cost and execution-truth contracts;
5. twelve-cell aggregation and the G1 decision under Work Plan 40 and Statistics Contract 41.

The expected PRE is that `tools/b4_run_cell.py` proves one bounded mechanics segment but no complete campaign executor or adjudicator exists. If an existing path does exist, return its exact identity and audit it rather than creating a second implementation.

## 3. E8: complete scientific cell executor

Extend the single reviewed B4 execution path so a `gpu_economic` cell is fully determined by its bound materialization. No scientific parameter may be supplied through CLI or inferred from defaults.

The executor must:

- consume and reverify the owner-reviewed authority chain, comparator population, current execution truth, observation contract v2, cost model, target, data roles, genesis, and the full effective cell;
- execute the real `rl_pipeline_with_validation` lifecycle, including epoch loop, validation, patience, checkpoint choice, and terminal evaluation;
- enforce fit/calibration/test chronology at the last point of use and prove that scored rows never affect gradients, early stopping, normalization, checkpoint choice, or thresholds;
- start every cell from its bound zero-update genesis; warm start, replay carryover, cross-cell state, and resume from an unrecognized state remain forbidden;
- retain the F9.2 intrasegment limits using the real SB3 update counter and preserve the configured training recipe after each bounded segment;
- write an immutable terminal record for `COMPLETED`, `FAILED`, `TIMED_OUT`, `THERMAL_STOP`, `RESOURCE_STOP`, or `EXTERNALLY_STOPPED`; no terminal state may be overwritten;
- emit the selected checkpoint digest, exact training/validation/scoring row identities, environment steps, real updates, wall time, resource maxima, and every input/code/config digest needed to reproduce the cell;
- emit per-bar gross return, each cost component, and net return, with identities that can be paired exactly to every comparator arm;
- keep checkpoints non-promotable and keep the sealed-2025 population unread.

The one consumed preflight is evidence about mechanics only and must not be inserted as one of the twelve scientific results.

## 4. E9: campaign ledger and result completeness

Materialize a twelve-cell campaign ledger from the reviewed population: three origins times four seeds. The ledger must exist before any later GPU dispatch and must identify the exact expected terminal result for every cell.

Provide executable checks that reject:

- missing, duplicate, foreign, or extra cells;
- result-to-cell identity mismatch;
- changed cell or campaign digest after materialization;
- reused attempt identity;
- a result lacking complete per-bar paired returns;
- a partial population presented as a campaign result;
- a mechanics/preflight result presented as scientific evidence;
- any read of the sealed evaluation period.

Design staged execution so scheduling decisions depend only on runtime health, never on observed return or gate direction. A first full-path cell may validate runtime mechanics after later owner authorization, but its score cannot decide whether the other eleven are run.

## 5. E10: executable statistical adjudicator

Implement one pure adjudicator from the frozen result records and the complete comparator population. It must implement the reviewed contracts rather than trust producer summaries.

Required outputs:

- exact paired per-bar net-return support and exclusions by origin, seed, and arm;
- the B4 G1 rule from Work Plan 40: improvement over every rule arm on at least two of three origins and at least three of four seeds, with every vote derived from records;
- IQM and uncertainty intervals under the declared aggregation hierarchy;
- stationary block bootstrap with `B=10000`, seed `20260824`, and Politis-White block length estimated from the control series only;
- Hansen SPA against the best rule arm, with the full registered trial population and no silent candidate omission;
- DSR under both predeclared trial-count conventions;
- a terminal `ADVANCES`, `DOES_NOT_ADVANCE`, or `INCONCLUSIVE` verdict with every failed or unavailable condition named.

The adjudicator must fail closed on broken pairing, insufficient support, non-finite values, altered records, missing trials, or an incomplete population. Unit of analysis and multiplicity family must be explicit in the output.

## 6. E11: runtime budget proposal based on the complete path

Replace the preflight's high-frequency `nvidia-smi` subprocess sampling with a bounded monitor suitable for campaign execution, while preserving the reviewed `87 C` stop. Sampling must be time-based and its overhead measured.

Without launching GPU work, return a proposed campaign resource contract containing:

- per-cell environment-step, real-update, wall-time, RSS, CUDA-memory, and thermal limits;
- a global GPU-hour ceiling for all twelve cells;
- maximum concurrency and a read-only inventory of eligible devices;
- heartbeat cadence, stop-file behavior, and operator-visible progress/ETA;
- retry policy by terminal class, with no automatic retry of an ambiguous or completed attempt;
- an estimate that separately accounts for learning, validation, checkpoint I/O, scoring, aggregation, and monitor overhead.

Do not claim a campaign ETA until the complete path has either been timed in a later authorized full-path cell or bounded conservatively from measured components.

## 7. E12: acceptance battery

Add focused tests and adversarial mutations for at least:

1. scored-row leakage into any training or selection stage;
2. CLI/default override of a materialized scientific term;
3. checkpoint selection from the wrong role or origin;
4. incomplete or foreign campaign population;
5. forged producer aggregate with intact labels;
6. broken per-bar pairing or cost reconciliation;
7. omitted comparator or trial in SPA/DSR;
8. altered bootstrap seed, replication count, or block-length source;
9. score-dependent staged scheduling;
10. mechanics result misclassified as scientific evidence;
11. sealed-period access;
12. update, wall, resource, thermal, and external-stop boundaries.

For each mutation, record the test that fails and the observed count after running it. Publish counts only from the final committed tip.

## 8. Required return

Return one packet containing:

- PRE call-path evidence;
- code and artifact identities;
- complete cell-executor and adjudicator call graphs;
- the twelve-cell ledger and dry-run status for every cell;
- the proposed resource contract and itemized estimate;
- test and mutation evidence;
- exact remaining owner decision.

The only successful terminal disposition is:

`B4_12_CELL_CAMPAIGN_READY_FOR_OWNER_AUTHORIZATION`

Otherwise return `B4_CAMPAIGN_PREPARATION_CORRECTION_REQUIRED` or a typed blocker. Do not execute a GPU cell merely to obtain the ready disposition.

## 9. Boundaries

- No GPU execution under this order.
- No training, venue, service, collector, key, or live-account action.
- No checkpoint promotion.
- No read of sealed-2025 data.
- No economic conclusion from the accepted preflight.
- No authorization of the remaining eleven cells or the twelve-cell campaign.

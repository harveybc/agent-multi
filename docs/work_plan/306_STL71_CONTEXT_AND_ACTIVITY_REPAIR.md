# STL71 Context And Activity Repair

Date:2026-11-12. Independent seasonal-only side experiment alongside phase2;
not a replacement for the selected71 model, live provider or main-lane recipe.

## Requirements

1. Use the current71 columns, their order, fixed preprocessing, full four-year
   TRAIN and one-year validation, and13 observations at stride2. No synthetic
   quote, zero padding, interpolation or feature removal to make support pass.
2. Reuse earlier computation only after exact raw/clock equality, identical
   TRAIN tuning scope, grid, algorithm and library version, authenticated
   receipts and independent endpoint replay.
3. Reject an empty evaluation month. Report missing-source intervals separately
   from a model's learned NOP. Calendar months absent from the source index must
   also be counted, not silently omitted from the coverage table.
4. Audit both observation-time causality and the fitted provenance of learned
   upstream features over the complete downstream receptive field.
5. Keep repairs in separately identified artifacts. Any changed activity
   producer requires renewed tuning and a matched raw control; do not compare
   it against a model trained on the old activity feature as a single-variable
   seasonal ablation.

## Current-Recipe Preparation

The old STL source adapter projected through a narrower historical recipe.
It now builds from the authenticated SUPPORT_RECIPE and then projects directly
to the current selected recipe. A regression fixture fails on the old route.

The current grid contains31116 rows and71 columns, with24895 rows before2024.
The renewed source preparation preserves the initial producer context rather
than trimming it a second time.68 earlier columns pass exact reuse checks;
OPEN@diff, RAW_BAR_LOG_RETURN and RAW_CLOSE_LOG_RETURN require new calculation.
The earlier px.range_12 is no longer selected. Old artifacts are retained.

`tools/reuse_stl_columns.py --complete` verifies reuse, calculates remaining
columns sequentially with their existing resumable checkpoints, then assembles
raw, continuous seasonal, ordinal seasonal and binary seasonal variants on
identical support. It does not start GPU training or replace any live model.

At initial publication,68 reused columns and the new OPEN@diff column were complete.
OPEN@diff selected period120, seasonal span13, nonrobust, history480; its full
31116-row endpoint calculation took732.5 seconds. The other two new columns
are in the sequential preparation queue, not reported as trained results.

The owner-local enabled service `am-stl71-prepare-20261112.service` runs this
finite preparation and then `audit_stl_activity_scope.py` on actual common
support. It resumes atomic256-row checkpoints and verified completed columns
after an interruption. The foreground run was moved to this service after the
first column; the second resumed its checkpoint, not from scratch. Resources:
CPU only, one numerical thread,3GiB memory-high/4GiB hard limit. Unexpected
process failures can restart; a normal Python failure exit1 stops with evidence
rather than looping forever. Service completion means preparation and audit
completed, not training or source admission. GPU training is not enqueued.

## Activity Context Finding And Correction

The original activity producer uses five purged TRAIN folds and a full fit for
later observations. Its raw13-observation,stride2 context has zero direct
target-row membership overlaps. Adding480-observation STL reaches504 raw rows:
some past feature values belong to another fold's estimator, which was fitted
using the current target row. There are898 such source-grid witnesses.

This does not mean endpoint STL reads future observations. It means that a
past-only filter can break a learned feature's original cross-fit isolation.
It also does not establish validation leakage: the witnesses are inside TRAIN.

The isolated repair keeps the producer's14 inputs, labels, five folds and model
hyperparameters. It increases its embargo96 ->503, derived from480+12*2-1.
No producer hyperparameter search or validation fitting is performed.

| Check | Result |
| --- | ---: |
| Direct context/target overlaps before |898 |
| Direct context/target overlaps after |0 |
| Changed TRAIN activity values |21809 |
| Changed validation activity values |0 |
| Original producer replay |Exact |
| Repaired producer cold reload |Exact |

Repair identity:`f0cb40cc89316f5a2e8fe83656456a478357216946a01385d7851b5ad645399d`.
Weights:`28fe4a578a7e077df41812786f0a08a5bb2886c0610adb6cacdb0a9e136bf39a`.
Evidence:[ACTIVITY_REPAIR.json](generalization/stl71_20261112/ACTIVITY_REPAIR.json).
The overlap test is a concrete direct-membership check, not a universal proof
of all upstream label-resolution, tuning or release-time properties.

The current prepared71 source still contains the original activity series.
The repaired producer is a separate candidate, not a claim that the running
preparation has already incorporated it. Its new identity must be carried
through the new raw control, scalers, parameters and receipts.

Two preliminary repair launches failed on resolving the legacy producer's
Python package, before fitting. The adapter now explicitly resolves the pinned
producer package, verifies its file paths, and completed the real fit/replay.
Those failure receipts are retained; they are not counted as completed runs.

## Release Sequence

The repair is not silently installed into the current71-input store. Remaining
source work is the qualified USDCHF hour and real pre-TRAIN calendar/activity
context documented in278. Recent failed Dukascopy downloads returned no usable
bytes; a successful Forexite Bid container download is not a decoded, admitted
quote tape. No unavailable observation was replaced.

Build a newly identified store combining qualified source repairs and the
extended-scope activity producer; recover real pre-TRAIN context with the proper
fold scope, not the full-TRAIN classifier applied indiscriminately. Retune
affected STL columns, test prefix/online parity and the complete fitted lineage,
then require nonempty support in every month before matched training.

## Completion Update

All71 columns and the four matched variants finished. The service exited
successfully; it is not a training process. Actual common support has21865
TRAIN and5460 validation windows before the qualified FX repair. Its fitted
activity-context audit confirms898 TRAIN direct-membership overlaps and zero
validation overlaps. The separately repaired activity estimator has not yet
been inserted into that store.

The qualified November FX repair was propagated to the current71-column
seasonal representation using four CPU workers. It recovers479 windows:
222 in December2023 and257 in January2024. Unaffected inputs and endpoints
outside the changed support are unchanged. All three prefix witnesses per
affected column pass. Paired support becomes22087 TRAIN and5717 validation
windows. January2020 and June2022 still have no common seasonal windows.

This propagation freezes the already selected STL parameters. It is a source
repair diagnostic, not a newly tuned or released training store. No missing
quote, activity value or calendar value was imputed. Evidence:
[COMPLETION.json](generalization/stl71_20261112/COMPLETION.json).

Main-lane307 completed its TRAIN account correction and rejected promotion on
validation, preserving299. STL does not authorize another actor reset or an
unchanged feature-selection repeat.

## Verification

62 focused CPU tests pass across STL causality, source routing, empty-month
rejection, exact reuse, fitted-context overlap, boundary witness admission,
source repair, prehistory and original activity recovery. Full repository tests
were not run. The owner-local service definition also passes systemd verification;
its diagnostic completion must not be confused with training admission.

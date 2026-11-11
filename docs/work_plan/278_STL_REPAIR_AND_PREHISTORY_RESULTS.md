# Causal STL Source Repair And Prehistory

Date:2026-11-11. Scope: the lateral seasonal-only experiment, not a new feature
selection decision. The69-input champion and all original data remain unchanged.

## Executed Results

| Work | Measured result | Disposition |
| --- | --- | --- |
| Propagate four qualified FX repairs |479 originally valid windows recovered |Separate diagnostic candidate |
| December2023 common STL support |0 ->222 windows |Previously empty month repaired |
| January2024 common STL support |264 ->521 windows |257 additional windows recovered |
| Real pre-TRAIN reconstruction |34/36 required columns;1567 raw observations |Candidate, not admitted |
| Producer baseline comparison |Maximum absolute error3.907985046680551e-14; identical finite masks |Pass at frozen1e-9 tolerance |
| Prehistory prefix comparison |Maximum absolute error0; identical finite masks |Pass |
| Full-grid tensor packing |30234 valid windows;exact tensors and support versus current loader |No additional windows recovered |
| Native missing USDCHF hour |Zero volume,flat OHLC,equal to preceding close |Rejected as carry-forward |

No feature removal, new model training, promotion, synthetic replacement or
modification of the live feed occurred. Both recovery workers completed.

## Requirements And Implementation

R1: only qualified real observations may repair a source. The native hourly
archive parser checks record size, ordered unique UTC bar starts, month bounds,
finite positive OHLC, consistent ranges and nonnegative volume. A zero-volume
archive record is not evidence of a real quote. It must not silently fill a gap.

R2: preserve the frozen transform and every unaffected output. The repair tool
authenticates the parent preparation, all69 column receipts and the separate
repaired-store receipt. A changed input at row t can affect only causal STL
endpoints from t through t+history-1. The tool recomputes that bounded union,
checks independent prefix endpoints, and preserves all other outputs exactly.
Four CPU processes ran the four affected columns; no GPU was needed.

R3: preserve original indicators and fitted transformations in prehistory. The
warmup tool pins the original technical, price, candle, SSA, wavelet, Hilbert,
cross-asset and frozen GMM producers. It first reproduces the original TRAIN
overlap, then builds a separate observed-row prehistory candidate. It does not
invent session-calendar values or extrapolate the supervised activity model's
training scope. LOG/ASINH/DIFF follow the existing per-column recipe.

R4: resume only the same inputs and code. Warmup has an exclusive lock, binding
receipt and terminal artifact hashes. Source propagation has an exclusive lock,
source/column identities and terminal artifact hashes. Both reauthenticate
sources before completion. Successful terminal resumes do not refit outputs.

R5: source context and evaluated endpoints are distinct. New lane preparations
use schema `causal_stl_lane.v2`, retain the complete source grid and store
`parent_indices` explicitly. Old pinned experiments are not rewritten. A
synthetic unit fixture proves that prehistory survives packing. On the actual
current store, old and new packing have identical valid tensors and support:
TRAIN24026,VALIDATION6208. The initial suspicion of24 extra lost windows was
disproved. It is not counted as a repair or an improvement.

## Propagation Detail

| Column | Exact changed input rows | Recomputed endpoints | Recovered endpoints |
| --- | ---: | ---: | ---: |
| fac.fx_f1.ret_4 |258 |739 |484 |
| fac.fx_f1.ret_24 |255 |739 |504 |
| wv.usd.A4 |258 |737 |736 |
| hb.eur.inst_freq |130 |609 |608 |

Exact changed-row counts include numerical differences inside the previously
qualified rebuild interval. They differ from report277's tolerance-based counts;
they do not imply new unbounded source edits. Endpoint counts overlap and must
not be summed as independent observations or trades.

Common seasonal support rises from TRAIN21865/VALIDATION5460 to
TRAIN22087/VALIDATION5717. Parameters remain frozen for this diagnostic, not
reselected after repair. January2020 andJune2022 still have zero common support.
The full requested seasonal-only training comparison therefore has not run.

## Missing Hour Diagnosis

The monthly native USDCHF Bid hourly archive contains744 records,530 with
positive volume. Against the existing observed tape,529 of530 OHLC records
match exactly; the remaining maximum absolute OHLC difference is0.00004.
For2022-05-19 21:00UTC bar start, the record has zero volume and four identical
prices equal to the preceding close. It is not admitted.

Archive SHA256:
`4b4c17385e193cc781b5575106e97d45ce915386e69f9afe6571432dd428bb05`.

This resolves whether the native archive can supply this missing observation:
it cannot. It does not prove that no other provider observed the market. Existing
alternate minute observations still require quote-side and timestamp admission;
no manual archive download is requested from the owner.

## Remaining Work And Main-Lane Boundary

The two unresolved prehistory columns are `px.hours_since_session_open` and
`act.p_hit`. Extending the first needs the actual2019 session/closure contract;
extending the second also needs the correct cross-fit estimator and lag support.
A full-TRAIN fitted activity classifier is not automatically a valid substitute
for held-out TRAIN context. The34-column candidate is therefore not an admitted
warmup store even though its producer and prefix tests pass.

To release the side experiment: qualify the missing real USDCHF observation,
complete calendar/activity context, build a new identified store, rerun the
affected parameter-selection and full causal/streaming/monthly-support checks,
then run the paired raw/seasonal contrasts. Repeating failed downloads or waiting
for an inactive worker is not a recovery plan. Do not claim these remaining
steps are running or automatically solved by the completed scripts.

This does not block the original69-input phase2 work. Its next task remains
qualification of the saved one-update actor control from277, followed by the
next justified refinement/selection step. The latest cumulative phase2 pass
remains69 ->69 ->69; no new pass is claimed here. The source champion's prior
TRAIN10.792108% and VALIDATION14.591090% returns are unchanged, not new results.

## Evidence And Verification

Private artifacts relative to the operator home:

Final STL propagation identity:
`4142879e2f1339c72d22ee0f45c97ce261534caeca1d87f3f564ebaa8d66407d`.

- `.local/share/agent-multi/stl-recovery-20261111/stl_november_v3/`:
  final source-bound RESULT,DONE and repaired seasonal matrix.
- `.local/share/agent-multi/stl-recovery-20261111/warmup_candidate_v3/`:
  BINDING,baseline parity,candidate parquet,RESULT and DONE.
- `.local/share/agent-multi/stl-recovery-20261111/dukascopy/`:
  archive and rejection receipt; no admitted source replacement.
- `.local/share/agent-multi/stl-recovery-20261111/CONTEXT_PACKING_PARITY.json`:
  exact current-loader tensor/support comparison.

Focused verification covers source parsing, malformed archives, zero-volume
rejection, bounded replay equivalence, untouched bytes, prefixes, stride/context
packing, frozen transforms, source/queue recovery and activity replay. Tests:
`test_stl_source_repair.py`, `test_stl_price_warmup.py`,
`test_causal_stl_inputs.py`, `test_stl_recovery_sources.py`,
`test_bounded_recovery_queue.py`, `test_consumer_activity_recovery.py`.
40 tests passed on CPU. Both completed production commands were invoked again:
terminal hash verification passed and neither recomputed outputs. Full repository
tests were not run; the documented Ruff executable is absent in this checkout.

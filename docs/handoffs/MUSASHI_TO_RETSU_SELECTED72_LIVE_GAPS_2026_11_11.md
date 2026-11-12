# Selected72: Live Sources And STL Status

Checked2026-11-12 UTC (2026-11-11 Bogota). This updates, not replaces, the
source-recovery handoffs. No broker or live service was modified.

## Current Research Artifact

The phase2 research actor now has72 physical inputs,28 groups, recipe
`recipe_a69bb5e4b50ed7a5`. Model SHA256:
`00f2fc65a39f9d5a53572ef915f433b57d5f7939b54699ad34e4274b0ac7874f`.
Bundle: `sac72-selected-export-20261111-r1/bundle/` in the private artifact
directory. Recipe SHA256:
`5cefbb0b3152c1e69926ac2859d9655e6a973563c12420fe5fe4267459654718`.
Do not use the old selected69/70 hard-coded column/scaler configuration.
The three readmissions since295's source are OPEN@diff,RAW_CLOSE_LOG_RETURN,
RAW_ELAPSED_HOURS. Full ordered columns and operations are in
[CURRENT_RECIPE_INPUTS.json](../work_plan/generalization/live_stl_status_20261111/CURRENT_RECIPE_INPUTS.json).
Use the sealed recipe constants,13 observed bars,stride2,no synthetic gaps.

This is a handoff candidate, not authorization to replace the demo model.
The actor is qualified historically; its fresh critics are calibrated separately
in296. Inference does not need the critic, but further SAC training does.

## What Is Actually Resolved

| Dependency | Measured recovery | Remaining live integration |
|---|---|---|
| act.p_hit producer | Correct consumer six-estimator bundle,31116 exact scores; SHA fdc95a0c... | Load full estimator0 after TRAIN, with all14 ordered real lagged parents; publish a current-hour receipt |
| fac.fx_f1.ret_4/ret_24 | Frozen full-precision mean/scale/loadings recovered | Six current FX cross feeds and original clock/quote semantics |
| wv.usd.A4 / hb.eur.inst_freq | Historical causal transforms recovered | Current real strength-parent bars and enough warmup |
| Kalman/GMM | Historical state/parameter replay | Actual warmup, restart and current-prefix verification |
| Session/event clock | Historical definitions known | Current-year calendar including real closures; an outage is not a closure |
| dly.us10y | Original DGS10 identity known | Verify current release/retrieval time and original availability rule |
| cot.eur.net_long/net_share/net_z156 | NOT resolved | Source identity repair below, not a substituted latest EUR print |

Activity weights are at the operator-home relative location
`.local/share/agent-multi/stl-recovery-20261111/activity/consumer_recovered/CONSUMER_ACTIVITY.joblib`.
Its RESULT reports live_ready=false: recovery of weights is not recovery of feeds.
The14 parents and lag order are in the prior live-source handoff. DAX and S&P
remain acquisition dependencies through activity even if absent as actor columns.

## COT Defect: Resolve Before Deployment

Your lts commit cccd857 reports52 overlap dates in2025:23 retained EURO FX
rows and29 retained cross-contract rows. That count is your evidence; the
original raw CSV was not located/re-counted in this review. Independently,
the pinned feature-eng producer8b6e203e... drops duplicate Date keeping last,
without filtering a contract. A two-row unit counterexample changes net_long
from-5 to100 solely by reversing row order. The fixture is test data ONLY,
never inserted into training, a store or live trading.

Required resolution, owned jointly by producer maintainer and live integrator:

1. Recover the exact pinned raw CSV0e9eb504... and its upstream acquisition
   record. Identify every duplicate date and its true contract/report type
   across TRAIN and2024, not just the2025 overlap. Return that audit to Musashi.
2. Filter by an explicit contract and report definition BEFORE feature
   construction; reject conflicting duplicates. Verify actual publication
   timestamps and revisions, including holidays/DST. A fixed93h offset alone
   is not proof that a report had actually been published.
3. If the old data mixed contracts, create a NEW corrected historical store,
   fixedTRAIN scalers and impact report. Musashi handles paired selection and
   model qualification. Never silently serve corrected EURO FX under the old
   model's expected mixed series. Preserve original receipts for comparison.
4. If that source cannot be reconstructed sustainably, propose a separately
   trained deployable-input variant with the entire unavailable dependency
   closure removed. Do not patch a running72-input model with zeros or constants.

This issue is more than a missing download. It is a mismatch between a model's
historical feature semantics and a reproducible future source. Historical
research numbers remain measured on their named store; live eligibility is
not established by those numbers.

## Current Operational Evidence

The inspected selected70 feed/decision receipts still say HISTORICAL_REPLAY,
last bar2024-12-31T21:00Z, zero submitted orders and no shadow observation.
The freshly read broker heartbeats at03:45 UTC still name the old linear
SPY daily model (no_execution) and ETHUSD4h model (snapshot_stale), not this SAC.
Service activity is therefore not evidence that this model is receiving live
inputs. This check did not alter or stop either service.

Next acceptance evidence from Retsu: every required parent real and dated,
one complete current closed-hour tensor no older than one hour, frozen
transform/restart parity, exact checkpoint identity and independent virtual
account observation. Daily/weekly inputs retain their real last publication
under the admitted as-of rule; they need not be newly published each hour.
Keep protected-position custody separate from an unavailable model input.
Only after the agreed forward comparison may a candidate be promoted.

## STL Is A Separate Unfinished Experiment

STL seasonal-only is NOT in this actor. Four historical price-derived repairs
recovered479 windows. The diagnostic still has zero common support in
January2020 and June2022; training_released=false. Prehistory rebuilt34/36
required columns from1567 real rows; unresolved: session clock and the
historically correct activity fold/scope. The unresolved USDCHF bar is
2022-05-19T21:00Z start; a zero-volume carried-forward quote was rejected.

Finish source/calendar/activity qualification, rebuild with a new identity,
then repeat causal/prefix/streaming and support gates before paired raw versus
seasonal training. Do not silently drop those months, interpolate or claim a
profit result from data-recovery receipts. The old STL plan used an earlier69
recipe; refresh its input identity before comparing against current72.

Evidence: [STATUS](../work_plan/generalization/live_stl_status_20261111/STATUS.json),
[COT witness](../work_plan/generalization/live_stl_status_20261111/COT_REORDER_WITNESS.json).
Main phase2 next step stays own-policy critic calibration and TRAIN-boundary
diagnosis; this handoff does not redirect it into broker implementation.

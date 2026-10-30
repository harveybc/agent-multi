# Post-audit implementation and persistent diagnostic queue

2026-10-29. Parent plans144/145. No predecessor result is overwritten.

## Implemented components

R1: report/bid/value caches refresh when execution config, references or funding
context changes. Cached value matrices carry a binding to report bytes,
position identity, modeled risk and value bytes; mismatches are rejected before
the close decision. This is an internal integrity contract, not authentication
against a malicious caller minting its own binding. Independent clock/deadline
arguments still originate in the engine calendar. Native formulas are unchanged.

Migration: `World.close_values` returns `(position, values, funding_open,
funding_end, binding)`. `decide_synthetic_values` requires `binding` for a valid
cached decision. Direct causal `decide` is unchanged. Source changes require a
new runtime identity; replay old v2 with its pinned checkout, not mixed files.

R2: `ReportTrackRecord` seals a producer-identified issued forecast into scalar
records. A target close becomes observable at `first_fill + lead*step`, not at
its bar's open. Only observations arriving at that timestamp enter errors;
missing targets are counted, never interpolated. Issue/observation duplicates,
backward/future clocks, producer mismatch and changed snapshots are rejected.
This is controller-side forecast history, not agent-state feedback to a neural
network. Snapshot/restart and submitted-array mutation tests pass.

R3 diagnostic foundation: `sampled_prefix_report` resamples past increments
and wick ranges into32 joint OHLC paths, with zero or median drift. It reads
only the supplied prefix; no price clipping, rolling z-score or future shape.
It is a deliberately simple baseline, **not a learned RL report adapter**.
The32-sample equal-weight empirical0.05/0.95 band is distinguished from a
residual-calibrated interval and is not assumed to have90% coverage.

## Frozen diagnostic calibration

New synthetic seeds100100..100109; smoke120100 is separate.30 independent price
draws expanded to360 paired cells per baseline, as in the crossed generator.
At bar64, freeze absolute-error interval radii using only prior matured errors,
minimum20 observations per lead, last64 errors, finite-sample order statistic
ceil((n+1)*0.9). Assessment uses forecasts issued from64 onward and their later
observed closes. Radii do not adapt during assessment. The baseline's market
prefix still evolves causally; that is not fitting to future assessment labels.

Report coverage and width at1/3/6/12/24hours. The predeclared diagnostic screen
is0.8..0.95 marginal coverage and positive finite width. This screen alone
does not establish conditional coverage, joint first-passage calibration,
forecast value, or readiness of a learned strategy. Dependence across overlapping
forecasts is retained; counts are not independent-sample counts.

## Automatic sequence

1. Run184 focused tests with zero skips on the actual pinned runtime.
2. For each of five frozen v2 finalists and five report qualities: replay all
  360 original native accounts exactly under the interface repairs, then
   enumerate executable alternative exits of every early-closed position.
   Include passive first passage in O*, preserve gap preemption, and report
   helpful/destructive/neutral exits, regret and SL/TP/flat cohorts by direction.
3. For each same account/genes pair, replay with voluntary closing disabled.
   Native SL/TP and calendar liquidation remain active. Unlike position-clone
   accounting, this measures changed occupancy, later entries and compounding.
4. Run both causal calibration baselines on the new360-cell bank.
5. Write sealed COMPLETE, raw compressed unit evidence and RESULTS.md. Source,
   config and old archive identities are checked; a restart reuses verified
   completed units. A failed assertion records FAILED and its exact cause.

Total27 sequential units. The full run comprises9000 baseline replays plus9000
no-close accounts, every early-close alternative in those baselines, and720
causal calibration worlds. These counts exclude the small smoke run.

The smoke executed all7 reduced units and authenticated re-entry with no new
evaluations.184 tests passed and Ruff correctness checks passed. Public smoke
receipts are under `docs/audits/evidence/post_audit_smoke_20261029/`.

## Resources and completion boundary

The service uses an immutable checkout, one CPU thread,100% CPU quota,
MemoryHigh3GiB, MemoryMax4GiB, no swap and bounded abnormal-exit restart. It
survives shell closure and is enabled for reboot. GPUs are not useful for this
native event simulation. The service template is committed with placeholders.

This queue finishes the automated **diagnostic** increment. R4's complete
horizon-specific/quality-ladder qualification and a learned adapter are not
implemented by this baseline. Therefore R6 DEAP and floating selection do not
automatically follow COMPLETE; that field explicitly says
`learned_adapter_qualified=false` and `automatic_next_training=false`.
No result-dependent threshold will be silently changed to pass a gate.
The next engineering action is to use the generated counterexamples to
implement the causal calibrated closer/adapter contrasts in144, not ask the
owner for intermediate result approvals or reuse v2 confirmation as fitness.

## Status

```bash
systemctl --user status post-audit-diagnostics-20261029.service
python -m json.tool <output>/STATE.json
```

STATE contains current unit, completed/total and measured completed-unit times.
An ETA must distinguish the25 native exit units from the two calibration units;
the two-cell smoke is not an accurate full-panel benchmark.

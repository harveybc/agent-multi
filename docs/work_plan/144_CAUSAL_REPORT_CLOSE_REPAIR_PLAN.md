# Causal reports and early-close repair sequence

Date:2026-10-29. Priority0, before model-specific sequential floating selection.
Parent:[143](143_CROSSED_V2_AUDIT_DISPOSITION.md). Status: design, not an executed
new search. Existing v2 artifacts remain immutable. No financial model training
or new DEAP was started by accepting the audit.

## Fixed requirements

- Business contract unchanged:1:50, one open position, no daily entry cap,
  entries only12h after weekly open, forced-flat12h before weekly close.
- No rolling z-score, clipping of market inputs, state-feedback inputs to the
  learned model, or retrospective ZigZag as an observation.
- Market/report-quality observations and broker position state remain separate.
  The controller may know SL/TP/side/units of its order; a reporter receives
  only its declared causal market inputs. Quality history is controller-side
  evidence from prior forecasts, not hidden P&L feedback to the model.
- Report identity binds producer/version, input cutoff, issue time, market
  clock, calendar segment, costs and sampling seed. No future perturbation may
  change a causal report or its already-matured quality history.
- Early-close oracle uses a fixed existing position, executable exits before
  passive first passage/forced flat, and net costs. It never chooses a new
  entry optimum. Report SL-first and TP-first cohorts separately.

## Sequential implementation and gates

| Stage | Work | Required evidence before next stage |
| --- | --- | --- |
| R0 | Native boundary regressions and audit integration | Two discriminating terminal-spread tests and two archived preemption tests; immutable audit attribution. Implemented in this handoff. |
| R1 | Bind all caches to execution config; cached values to originating report/position; explicit caller clock/calendar | Mutations rejected or recomputed, stale report/cost witnesses, old-account parity on frozen v2 snapshots. No scientific result rerun merely for unchanged outputs. |
| R2 | Causal report registry and matured-error tracker | Seal at issue time; no reconstruction from later data; target availability checked by timestamp; future mutation, cold restart, missing-report and horizon maturation tests. |
| R3 | Explicit nominal uncertainty calibration and baseline reporter adapter | Distinct TRAIN-only fitting and assessment support; nominal90% marginal intervals at1/3/6/12/24h, joint-path diagnostics separate; width plus coverage; prefix invariance; model identity in reports. |
| R4 | Discriminating positive controls, frozen before new metrics | PC1--PC6 below, both sides, native costs and gap handling; invalid oracle fixtures rejected before candidate evaluation. |
| R5 | Fixed-size closer/account contrasts on new synthetic banks | Old coefficients/no history vs causal error history; separate entry/close reliability controls; no-close account ablation; fixed-position regret for all finalists; no tuning on v2 confirmation. |
| R6 | New optimization only after R1--R5 | Population>=20,generations>=300,seeds7/19/43; signal genes first at fixed size, exposure second; frozen representatives and untouched confirmation bank; result/audit packet. |

R0 is complete. R1--R6 are the work to execute next, not claimed as completed.
No babysitting or intermediate result approval is required to implement and
test these stages. A failed gate records its counterexample and runs the
correction/tests before dependent experiments; it never silently continues.

## Controls to specify and seal

- PC1: ideal report through the entire adapter reproduces native ideal outcomes;
  single-path oracle is test-only, never fitness for the causal experiment.
- PC2: paired report-quality ladder with controlled error magnitude, serial
  structure and calibration separately. Publish return/DD/regret and intervals
  versus quality; diagnose reversals rather than forcing a monotonic plot.
- PC3: explicitly no-edge causal market null, plus a calibrated causal baseline.
  Abstention, activity, costs and excess return over appropriate controls are
  reported separately. Infinite-width bands cannot pass by coverage alone.
- PC3b: confident-wrong causal reporter, with and without matured-error history.
  A claimed gate must beat its ungated comparator on new paired draws. Auditor
  suggested return>=-0.1% and DD<=2% are proposed acceptance budgets, not values
  to tune against the old confirmation bank. Freeze exact budgets/sample sizes
  in the next executable manifest before reading new outcomes.
- PC4: horizon-specific positive controls with the unique feasible winner at
  each3/6/12/24h certified by a native oracle in advance. The adapter must route
  distinct horizon information; repeating a common residual-week path is not
  evidence of specialized experts. Also test ties, NOP and both directions.
- PC5: early-close regret against O*, helpful/destructive share versus passive
  hold, and saved SL-bound versus sacrificed TP-bound cash. Ideal control:
  zero destructive closes and zero regret within accounting tolerance. For
  imperfect reporters compare paired harm and savings, plus full-account
  no-close ablations. Never optimize a target percentage of early closes.
- PC6: mirror-equivariant prices/reports with symmetric funding for the pure
  symmetry witness; real asymmetric funding as a separate arm. Independent
  draw clusters, shared random numbers within cost-only pairs, and independent
  RNG replications as a distinct sensitivity experiment.

## Statistical and computational discipline

Keep price-shape and band-calibration factors separate. Retain a future-aware
quality ladder solely as a diagnostic bridge to v2, never as causal evidence.
Use at least32 sampled paths for the new uncertainty experiment, but test
coverage/width and sampling convergence:32 is not a calibration certificate.
No decision between marginal-to-joint modeling and a market-only closer is
settled by this number; evaluate the minimal causal baseline before committing
to a more complex learned output architecture.

Use new calibration/confirmation seeds, declared in a versioned manifest after
fixtures pass. Freeze the comparison family before inference; use paired draw
clusters, Holm-adjusted tests, DD intervals and family-stratified tables.
The forecaster's error tracker sees only errors whose outcomes have matured
at the decision timestamp. Calibration can use only prior matured evidence;
the current outcome cannot select its own interval or gate.

Automate one stage at a time with source/config hashes, native accounting,
atomic per-unit receipts, explicit failure and resumable identity checks.
Reuse the four-worker bounded CPU pattern for synthetic accounts. GPUs become
relevant only for an explicitly defined learned adapter, not for relabeling
these deterministic simulations as neural training. Measure the small full-path
pilot before announcing a search ETA; no new8-hour search is running now.

## Shared work with Satoshi

Share the native fixed-position O*/regret evaluator, calibrated interval
definitions, timestamp/maturation tests and business contract. Satoshi's expert
labels remain in his lane; this lane does not import ZigZag as an observation.
Send the measured repair/control evidence for review before floating selection.
Independent reports should distinguish model error, calibration error, closer
error and execution error instead of attributing all loss to the data.

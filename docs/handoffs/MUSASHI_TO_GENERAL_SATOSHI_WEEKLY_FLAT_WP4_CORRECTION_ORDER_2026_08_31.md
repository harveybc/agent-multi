# Musashi to General Satoshi: weekly-flat WP4 correction order

Date: 2026-08-31
Disposition: `REVISE_BEFORE_GPU_DISPATCH`
Authorization: CPU implementation and bounded evidence only.

The 63-cell materialization is directionally correct and the 21 focused tests pass. No GPU campaign is authorized because the current package does not yet prove the mechanics or statistical design it claims.

## WP4-C1: complete executable identity

The driver verifies only `session_exposure.py` and `flatten_custody.py`, although WP4.0 froze six gym-fx files plus WP3 executor files and contracts. Verify every frozen executable/contract input actually consumed. The materialization manifest must enumerate each cell id and digest; a cell's self-contained digest is not an external binding. The reviewed dispatch must bind the manifest digest, and the driver must reject a consistently altered/re-digested cell, missing/extra cell, substituted manifest, or changed authority file.

## WP4-C2: genuinely paired treatment

The mechanics action RNG currently includes `cell["digest"]`, so paired W0 cells receive different action streams. Replace this with a pre-generated action/genesis tape bound by digest and shared across every cell of a pair/family for a given seed. Assert prefix equality before the first treatment-dependent state and refuse any drift in observations, actions, initial account, costs, data or seed.

## WP4-C3: truthful session evidence

The current driver generates sine-wave prices around one artificial 48-hour gap. It is not evidence from two historical weekend gaps plus an adjacent holiday. Build bounded fixtures from hashed historical bars and the accepted historical-time calendar, preserving missing intervals. Publish source role, timestamps and digest. Include at least two observed weekly closures and one holiday/exception interval; if no eligible holiday exists in the bound history, return `HOLIDAY_EVIDENCE_UNAVAILABLE` rather than fabricating one.

`EXPECTED_MARKET_CLOSED` cannot be claimed as an env step when there are correctly no bars during closure. Test it separately through the shared clock/state authority at timestamps inside the gap and state clearly that the actionable env trajectory contains four states while the authority probe covers the fifth.

## WP4-C4: execute the missing mechanics

Current evidence reports zero cancellations in every smoke and does not exercise a resting pending entry. Add real broker-path scenarios for:

- accepted resting entry cancelled in wind-down while both protective legs survive;
- voluntary close and forced close;
- cancellation rejection/fill race/still-open/unknown outcome;
- failed flatten and durable recovery after interruption;
- long and short exposure;
- exact reopening boundaries and stability-reset behavior.

No success may be inferred from a requested effect. Report observed terminal outcomes.

## WP4-C5: derive, do not assert, conservation

`no_bar_inside_closure` is currently hard-coded `True`, and the local `trades` value is unused. Derive and assert from executed artifacts:

- zero bar timestamps inside every closure;
- zero suppressed rewards;
- authoritative close-event count equals won + lost + breakeven;
- gross PnL minus venue costs equals net PnL;
- reward/account-equity/PnL reconciliation;
- forced/voluntary close counts and costs;
- pending-entry and protective-order conservation;
- zero unresolved incidents for eligible cells.

Persist the underlying rows/digests, not only totals.

## WP4-C6: complete W2 parameter authority

Plan 42 says spread, gap and volatility thresholds are calibrated prospectively. They may not be silently frozen at section-4 defaults while W2 is called complete. Before economic execution, either:

1. predeclare bounded threshold domains and include their trials in W2 and the multiplicity ledger; or
2. explicitly split W2a (hours/bars/checks with thresholds frozen as a provisional mechanism screen) from W2b (threshold calibration), with W2a carrying no claim that W2 is complete.

Likewise, the three G2 baseline windows fixed at four bars are treatment-bearing parameters. Justify them mechanically from the accepted observation contract or include them in a bounded calibration/ablation; declaring a free value is not enough.

## WP4-C7: correct the statistical unit and selection protocol

Five seeds are optimization replicates, not five independent market histories. The economic unit is the paired closure-week (with dependence across adjacent weeks/regimes). Replace tests on five per-seed means with a hierarchical paired analysis:

- identical closure weeks across cells and seeds;
- per-week paired net-return differentials after costs;
- stationary/block bootstrap over closure weeks, stratified or hierarchically aggregated across seeds;
- minimum eligible closure count and power/precision rule declared before execution;
- insufficient support yields `INCONCLUSIVE`, never a winner;
- report seed dispersion separately from market-time uncertainty.

W1 selection occurs on fit/calibration only, freezes one timing, and is evaluated on a later untouched decision window. W2 begins only after that freeze and uses its own calibration/decision separation. No dataset may both choose and judge its winner. Holm-Bonferroni versus the default may remain descriptive, but it does not by itself correct winner selection over 16/45 cells; add a predeclared family-level procedure such as Hansen SPA/stationary bootstrap or nested selection with untouched evaluation. Record every attempted cell in the trial ledger.

The one-standard-error tie rule must define which uncertainty estimate it uses and must not treat seeds as independent weeks.

## WP4-C8: benchmark and return

Re-run only bounded CPU mechanics representatives after C1-C7. Report median and dispersion across repeated timings, not `best_of_3`. Separate environment throughput from SAC update throughput; do not extrapolate GPU hours until a bounded real SAC update benchmark exists.

Return PRE/POST reproducers for every finding, mutation tests, focused and complete suites, corrected materialization, corrected statistical protocol, and a proposed bounded SAC preflight. Do not launch that preflight without independent acceptance.

## Boundary

No GPU, long training, complete economic sweep, deployment, service changes, venue connection, command, position change, checkpoint promotion or live activation.


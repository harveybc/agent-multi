# 390: market generation before SAC

Status: offline generator screen, not policy training or a promoted generator.
389 and the two historical-budget 374 workers were stopped. Retain their
checkpoints, banks and histories. Do not requeue them automatically.

| Requirement | Implementation / observable |
|---|---|
| More than a single pivot per market state | Four contiguous historical swings per macroblock. GMM over signed return efficiency, log hourly volatility, log duration, retracement relative to swing displacement. Raw one-bar SNR is excluded. Choose 2..6 components by 2022 log likelihood after fitting on 2020-21; refit on 2020-22. Use empirical transitions, durations and joint OHLC/FX trajectories; not a fitted HSMM or a neural generator. |
| No release based on real validation | Fit and donor cutoff December 2022. Reserved real 2023 from TRAIN for screening. 2024 and TEST not read for scoring. Existing frozen feature producers are from the parent recipe, not cross-fitted evaluation of those producers. |
| Price/clock and context variants | Price donor selection cannot use the macro features. Sample transitions then historical four-swing blocks. Context variant first fixes a real training release calendar, selects donors using lagged macro observations, session, preceding move and sampled state. Transport six crosses jointly with OHLC. Keep bond/COT/index release histories on this calendar; regenerate the frozen activity estimator. Do not score this as inventing new macro histories. |
| No broken bars or hidden fills | Preserve observed gap moves, truncate donor at calendar gaps, no state reset at donor joins, preserve missing FX source availability. Donor index and truncation logged. Current/previous price normalization used to translate complete OHLC bars. |
| Check against a cheap alternative | Random joint-block control. Fixed seeds 7 and 19 for all variants. Save both the common 61-column tensor source and the matched 71-column source on contextual paths. Independent rebuild of 61 must exactly equal the corresponding 71 columns. |
| Test business consequences | By horizon/side, measure first SL/TP and next-open early close versus holding until protection/expiry, including cells stratified by preceding drift and volatility. Screening proxy in pips, not after-cost native execution. Compare to real 2023 using real quarterly drift envelopes. |
| Blind check for external data | Hide each 2023 block's external targets, use only completed previous block descriptors/regime and starting hour to choose eight TRAIN donors. Compare empirical CRPS against equally sized random donor sets. Report all columns and quarters. This retrieval test is complementary to the path simulation. |

Runner: `python -m tools.regime390 --help`. Use the same authenticated spec,
factor state, business catalog and selected-299 recipe as 387. Use an empty
output directory, one CPU thread, CUDA disabled and a 3 GiB system RAM limit.
Writes OHLC, feature parquets, donor indices, per-seed gate tables, progress
and final result. Never enqueue RL merely because files exist.

Screen rule: both seeds must pass all supported envelope metrics (unsupported
cells also fail) and improve mean normalized price discrepancy over the
random-block control on each seed. For the contextual variant the blind
analogue test must also improve aggregate CRPS in each reserved quarter.
These are engineering gates, not confidence intervals. A passing screen
still needs physical seam / cross-rate checks and matched native account
qualification before a training comparison. Reject a failing candidate;
do not tune gates against 2023 or repeatedly retest it as an untouched panel.

## Execution and disposition

Stopped 389/seed 7 after the last completed epochs 3/2/3/3 for real,
+4/+8/+16 (validation returns -10.64/-6.23/-11.67/-5.41 percent).
Disabled dispatch and the two 374 history units; retained artifacts.

947 four-swing blocks including 715 eligible by date for fitting. 2022
log likelihood selected two states. Each candidate generated 18,698 bars
for screening. The 61-only variant ignores macro values during sampling;
both variants use the same underlying FX source availability for donor
admission. Windows do not restart at donor boundaries. Contextual paths
retain actual macro releases rather than pretending to synthesize surprises.

The first subset table had an ordering bug: metrics with fewer columns were
compared to the wrong tolerances. Added a name-alignment regression. Physical
checks then exposed cumulative triangle drift when composing joint returns.
Now derive EURJPY/EURGBP from the synthesized legs and the donor's observed
basis at that tick. Keep both rejected iterations. On replay of the original
seeds, both variants passed the resulting statistical and physical screens.
This is a repaired diagnostic, not fresh independent confirmation on 2023.
The blind analogue ensemble improved normalized CRPS against random in all
four quarters, on 7/10 external features in the annual aggregate.

Frozen mechanical replay on 2020-22: real 61 +2.51%, 210 trades; price-only
paths -14.78% / -30.54%, 314 / 444. Real 71 +12.58%, 541 trades;
context paths -20.91% / -25.84%, 478 / 515. The same contextual paths with
the 61-input frozen projection returned -15.85% / -12.21%, 234 / 238.
The 61 projection retains the constant responses of entirely removed groups
for inference; it is not a newly trained 61-input policy. Supports can
differ by recipe. These are TRAIN-period replays, not validation returns.
No accounting discrepancy above 2e-11. The level of in-sample champion
profit is not a target for generator fitting.

Cheap transfer diagnostic (H6): previous closed-bar features, a one-bar-old
position, next-open liquidation versus hold to protection/expiry, no costs.
Fit equal-sized depth-limited boosting probes on 2020-22, evaluate on real
2023 within TRAIN. No campaign split 2024 or TEST. Mean AUC for real vs
10% synthetic: long 61 .6594/.6573, short 61 .6429/.6456,
long 71 .6597/.6568, short 71 .6438/.6480. Shuffled labels ~.5.
Not proof of policy gains. Monitor wavelet-energy and ATR tails.

Four separate 24,895-bar 2020-23 calendar paths per variant (seeds
7/19/43/71); 4/8/16 are the first 1/2/4 paths, never one joined account.
Donors remain pre-2023; the full calendar banks are training material,
not held-out evaluation sets. All eight passed the physical screen. Each
path has hashes and a donor trace. No large GPU queue is authorized by
these results. Next research comparison: matched short policy pilots with
separate true 61- and 71-input controls, common support within each pair
and validation activity and long/short breakdown before scaling.

One-command CPU sequence: `bash tools/qualify_regime390.sh` (no arguments
prints the required paths). Outputs include `RESULT.json`, `NATIVE.json`,
`INPUT_TAILS.csv`, `TRANSFER.csv` and `banks/COHORTS.json`.

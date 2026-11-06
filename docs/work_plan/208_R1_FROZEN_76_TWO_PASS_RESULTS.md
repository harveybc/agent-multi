# R1 Frozen Selection: 93 to76 Inputs

Completed2026-11-06 at05:16:18 America/Bogota. This is the ordered removal and
reinsertion pass authorized after the SNR review in[207](207_SNR_REVIEW_AND_NEXT_MASK_PASS.md).
No retraining, new normalization, filtering, synthetic inputs or trading-rule
changes occurred. All native rows and original decision support were preserved.

## Counts and Authentication

| Stage | Features | Groups |
|---|---:|---:|
| Phase1 admitted population | 120 | not recomputed here |
| Current parent after previous rounds and refit | 93 | 30 |
| This deletion pass | 76 | 28 |
| This reinsertion pass | 76 | 28 |

There were104 unique evaluated subsets: the baseline,88 deletion candidates,
and15 newly evaluated reinsertion candidates. One reinsertion repeats a cached
subset. Sixteen accepted deletions remove17 columns because the day-of-week
sine/cosine pair is one coherent block. No reinsertion improved the score.
Every accepted deletion strictly improved the validation score in this run;
the configured removal-on-exact-tie rule was not needed for those decisions.

- Parent identity: `be89e2e5b123183dbb751e805ac5b497b15d705b1eaff265b39789beae2e44bf`.
- Parent checkpoint: epoch7, seed7; SHA256 `bac17a9c517393e171117fffc08f52db38ca6727c4a62b81a9f282ff790ac3af`.
- Selection identity: `9b831ec28a336aa2585bbd5c36a9bc921bec2621c3ad3b61ca5eb21f7c98ffe3`.
- RESULT.json SHA256: `84812ab715af1c9f368ab56a453ed87cfa255168b92743ac6bf070b928171a87`.
- Termination: `TWO_PASSES_COMPLETE`; exact initial probe, unmasked native
  accounts and post-pass unmasked replay; original support preserved.
- Collected evidence: `/var/tmp/sac93-r1-two-pass-20261106/`, including all
  candidate caches, PASSES.json, RESULT.json, FEATURE_COUNTS.json, CLOSE.json
  and native final trading artifacts. Collected result hash matches CLOSE.json.

## Native Trading Results

| Metric | TRAIN93 | TRAIN76 masked | VAL93 | VAL76 masked |
|---|---:|---:|---:|---:|
| Net return % | 39.0376 | 13.5983 | -1.1815 | 11.2871 |
| Net profit USD | 3903.7634 | 1359.8287 | -118.1462 | 1128.7073 |
| Return percentage points/week | 0.186783 | 0.065064 | -0.021879 | 0.209020 |
| Sharpe | 1.0076 | 0.4549 | -0.1157 | 1.7575 |
| Max drawdown % | 17.8703 | 14.8850 | 7.2645 | 3.1303 |
| Closed orders | 2138 | 1838 | 266 | 205 |
| Orders/week | 10.2297 | 8.7943 | 4.9259 | 3.7963 |
| Win rate % | 59.1207 | 58.4331 | 50.7519 | 58.5366 |
| Buys / wins | 1210 /732 | 1175 /704 | 59 /33 | 43 /26 |
| Sells / wins | 928 /532 | 663 /370 | 207 /102 | 162 /94 |
| Buy net profit USD | 1963.5432 | 1489.6418 | 271.9104 | 383.5922 |
| Sell net profit USD | 1940.2201 | -129.8131 | -390.0566 | 745.1151 |
| Median holding observed bars | 1 | 2 | 3 | 3 |
| TP closes | 1060 | 937 | 102 | 98 |
| SL closes | 606 | 551 | 95 | 66 |
| Agent early closes | 392 | 291 | 52 | 30 |
| Weekly forced closes | 80 | 59 | 17 | 11 |
| Helpful / harmful early closes | 200 /192 | 149 /142 | 21 /31 | 14 /16 |
| Mean early-close regret/risk | 0.3715 | 0.3931 | 0.5055 | 0.4511 |

The final validation buys represent20.98% of trades; sells79.02%. Buy win rate
is60.47%, sell win rate58.02%. Both sides operate and contribute positive net
profit in validation. TRAIN sell net remains negative. The early-close diagnostic
still classifies16/30 validation closes as harmful relative to passive SL/TP
protection of the same fixed position; that is not an account-continuation value.
Early close is14.63% of validation closes, not evidence of a solved close policy.

Costs remain active. Final validation includes commission349.58833USD,
spread209.84641USD, slippage200.75738USD and financing-26.82961USD (net credit).
Maximum absolute cash reconciliation error is1.84e-11USD in final TRAIN and
1.11e-11USD in final validation. There are no margin closeouts or level clipping.

TRAIN spans four years and validation one; cumulative returns are not equal-time
generalization-gap estimates. Selection ranks full-validation return/week,
requiring>2orders/week in each partition; TRAIN cannot compensate a lower score.

## Removed Columns

`cal.utc_dow_sin`, `cal.utc_dow_cos`, `cal.utc_session_asia`, `px.logret_6`,
`px.logret_12`, `px.logret_48`, `px.close_loc_6`, `px.er_24`, `fac.fx_f1.ret_1`,
`fac.fx_f1.kalman_slope`, `ev.london_fix.hours_since`, `ev.ny_cut.hours_since`,
`ssa.trend_minus_close_pips`, `wv.lc.D1`, `wv.lc.energy_D4@log`, `wv.r1.A4`,
`px.rv_24`.

This does not imply those features or families have no information. The result
is specific to these trained weights, the ordered search and this validation
criterion. The deployment artifact remains the93-input parent plus its76-feature
mask, not a newly trained76-input SAC. The mask sets excluded standardized inputs
to zero; it does not remove source rows or alter raw data/scaler constants.

## Interpretation and Next Step

Validation return and drawdown improve while activity remains eligible and
TRAIN return stays positive. This is a useful selected development candidate,
not independent confirmation: the validation score was the selection objective.
It does not establish that a fresh76-input fit will reproduce these numbers.

The next structural training should instantiate only the76 retained columns and
their28 groups, keep the same risk/data/monitor contract, save inference weights
and compare against this frozen masked reference. G2 continues independently
on its original93 inputs; do not change its population mid-experiment. Synthetic
training still requires coherent joint generation and feature regeneration.

# 379: can the position-aware lookup beat a scalar closer?

Offline decision gate, before a new SAC or floating run. Keep the 71 market
channels, native engine, risk profile and frozen opener from 378. No 2024 or
TEST. Two separate directional heads; flat routing is unchanged. No account
state, feedback, realized error or future price in the market encoder.

1. Sample up to 160 flat origins/year with seed379 before scoring. Exclude
   origins within 26 valid decisions of the 378 and 316 witnesses. Develop on
   2020-22; reserve 2023 for one comparison. The source was trained on TRAIN,
   so this is not out-of-sample evidence for the entire system.
2. At each origin force an open now or one decision later in both directions.
   At decision three, only intact positions qualify. No replacement. Record
   paired native CLOSE vs HOLD ending equity after 24 valid decisions under
   the same source continuation including re-entry, truncated at split end.
   Target = difference divided by the position's initial monetary stop risk.
3. Both diagnostic heads use the latest sample and window mean of all 71
   fixed-TRAIN channels; standardize that basis on fitting examples only.
   Scalar ridge vs scalar plus market-conditioned corrections at eight grid
   corners, same ridge penalty 0.1 on mean squared error, unpenalized common
   intercept. The controller interpolates the market-only output at actual
   distances from liquidation quote to SL and TP in initial risk units, and
   hours to native force-flat. Fit axis bounds at development 10/90 quantiles
   per side. Saturate the lookup only, never market inputs. Publish saturation.
4. Select CLOSE strictly above zero. Compare mean account-intervention regret
   in R and USD, detrimental closes, and material same-market conflicts. Use
   paired calendar-week block resampling for the difference in regret.
   Separately replay whole 2023 accounts with the frozen opener. No threshold
   or interpolation adjustment using the reserved outcomes.
5. To advance require reduction in mean regret on each side, paired pooled
   95% interval excluding zero in favor of the table, lower pooled regret on
   material pairs, and account net no worse than the scalar. Insufficient
   support or failure means no promotion. Tests cover mirrored barrier queries,
   gaps, time boundary, conflicting actions at identical market, and HOLD ties.
   Passing is evidence for testing the interface, not profitability.

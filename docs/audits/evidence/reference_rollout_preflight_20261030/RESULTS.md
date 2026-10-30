# Frozen-reference rollout contrast

Identity: `9e6dc21b10d9d56892a3cbd1398e6ab50062cf0125cd8de3a6832b8898c8fe55`.
Seven frozen policies. Synthetic engineering, not financial validation. All costs retained.
Value is Q of an explicit reference continuation, not exact Q of the adaptive selector.
Carried forecasts preserve the current whole ensemble; no future real reports are read.

| Bank / reporter | Policy | Return % | Worst DD % | Mean Sharpe | Orders/week | Buy/sell | Win % | Buy win % | Sell win % | Early | Local harmful |
|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|
| confirmation_ideal | simple_c9408e3b | 1.60203 | 0.41326 | 17.92151 | 3.00000 | 2/1 | 100.00000 | 100.00000 | 100.00000 | 1 | 0 |
| confirmation_ideal | predictive_proxy_e5df77dd | 2.24951 | 0.41326 | 17.00644 | 4.00000 | 3/1 | 100.00000 | 100.00000 | 100.00000 | 2 | 0 |
| confirmation_ideal | sequential_280977a9 | 2.54806 | 0.36932 | 17.61824 | 5.00000 | 3/2 | 80.00000 | 66.66667 | 100.00000 | 3 | 1 |
| confirmation_ideal | sequential_59a75d73 | 3.00584 | 0.33193 | 16.16201 | 7.00000 | 5/2 | 85.71429 | 80.00000 | 100.00000 | 4 | 2 |
| confirmation_ideal | rollout_simple | 3.23737 | 0.26233 | 19.47518 | 7.00000 | 3/4 | 100.00000 | 100.00000 | 100.00000 | 5 | 2 |
| confirmation_ideal | rollout_proxy | 3.22555 | 0.24378 | 15.89955 | 7.00000 | 3/4 | 85.71429 | 100.00000 | 75.00000 | 6 | 1 |
| confirmation_ideal | rollout_joint | 3.22555 | 0.24378 | 15.89955 | 7.00000 | 3/4 | 85.71429 | 100.00000 | 75.00000 | 6 | 1 |
| calibration_noise_1 | simple_c9408e3b | 1.40209 | 0.37167 | 20.34910 | 3.00000 | 3/0 | 66.66667 | 66.66667 | - | 2 | 1 |
| calibration_noise_1 | predictive_proxy_e5df77dd | 1.74992 | 0.64027 | 17.51036 | 4.00000 | 4/0 | 75.00000 | 75.00000 | - | 2 | 1 |
| calibration_noise_1 | sequential_280977a9 | 0.70055 | 0.73944 | 6.88684 | 19.00000 | 14/5 | 52.63158 | 64.28571 | 20.00000 | 18 | 14 |
| calibration_noise_1 | sequential_59a75d73 | 1.36594 | 0.64611 | 10.75046 | 14.00000 | 9/5 | 57.14286 | 77.77778 | 20.00000 | 12 | 7 |
| calibration_noise_1 | rollout_simple | 0.29863 | 0.84142 | 2.69588 | 23.00000 | 16/7 | 47.82609 | 50.00000 | 42.85714 | 23 | 15 |
| calibration_noise_1 | rollout_proxy | 0.40683 | 0.38054 | 5.41964 | 25.00000 | 20/5 | 52.00000 | 60.00000 | 20.00000 | 25 | 19 |
| calibration_noise_1 | rollout_joint | 0.57312 | 0.85534 | 3.91362 | 24.00000 | 15/9 | 41.66667 | 46.66667 | 33.33333 | 23 | 14 |
| confirmation_noise_1 | simple_c9408e3b | 1.76560 | 0.21876 | 16.54896 | 7.00000 | 4/3 | 85.71429 | 75.00000 | 100.00000 | 7 | 3 |
| confirmation_noise_1 | predictive_proxy_e5df77dd | 1.54312 | 0.31870 | 17.33523 | 8.00000 | 4/4 | 75.00000 | 75.00000 | 75.00000 | 8 | 3 |
| confirmation_noise_1 | sequential_280977a9 | 0.65227 | 0.44457 | 6.92161 | 23.00000 | 13/10 | 47.82609 | 46.15385 | 50.00000 | 23 | 12 |
| confirmation_noise_1 | sequential_59a75d73 | 0.72581 | 0.58893 | 9.64802 | 23.00000 | 10/13 | 43.47826 | 50.00000 | 38.46154 | 22 | 9 |
| confirmation_noise_1 | rollout_simple | -0.65638 | 0.84888 | -8.80857 | 30.00000 | 13/17 | 36.66667 | 46.15385 | 29.41176 | 30 | 13 |
| confirmation_noise_1 | rollout_proxy | -1.34393 | 1.36616 | -19.94585 | 32.00000 | 18/14 | 21.87500 | 27.77778 | 14.28571 | 32 | 16 |
| confirmation_noise_1 | rollout_joint | -0.65486 | 0.85358 | -8.54955 | 28.00000 | 12/16 | 35.71429 | 41.66667 | 31.25000 | 28 | 12 |

## Actual Adaptive Close Attribution

{"accounts": 1, "closes": 6, "pareto_relations": {"CLOSE_DOMINATES": 3, "HOLD_DOMINATES": 1, "TRADEOFF": 2}, "profit_harmful": 1, "profit_helpful": 5, "utility_harmful": 1, "utility_helpful": 5}

Positive account delta means CLOSE outperformed a one-tick HOLD followed by the same adaptive policy.
This is separate from the local passive-position comparison. No automatic strategy promotion.
World transformations are paired; aggregate means do not constitute independent-sample significance.
Account summaries and raw receipts retain costs, horizon mix, order sizes, activity and undefined Sharpe.

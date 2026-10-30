# Pragmatic controller contrast

Identity: `f5a82347b3181ee814fe07869b9e858965d52183b287293b144378eaced4e823`.
Synthetic strategy/output engineering. Not financial VALIDATION or learned RL performance.
Predictive proxy is not a reproduction of the historical strategy. Sequential is bounded two-order planning.
All calibration candidates retained. Representatives frozen before confirmation; >2 orders/week is explicit.

| Arm / profile | Reporter | Return % | Worst DD % | Mean Sharpe | Orders/week | Buy / sell | Win % | Buy win % | Sell win % | Early closes | Harmful |
|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|
| simple / c9408e3b | ideal | 5.0386 | 1.0833 | 22.2902 | 6.2500 | 111 / 114 | 100.0000 | 100.0000 | 100.0000 | 77 | 0 |
| predictive_proxy / e5df77dd | ideal | 6.3196 | 1.0852 | 24.5515 | 8.6389 | 154 / 157 | 99.6785 | 99.3506 | 100.0000 | 130 | 0 |
| sequential / 280977a9 | ideal | 6.9897 | 0.5772 | 24.9245 | 9.8611 | 172 / 183 | 94.6479 | 95.9302 | 93.4426 | 148 | 71 |
| sequential / 59a75d73 | ideal | 7.1454 | 0.5772 | 25.3017 | 10.2778 | 177 / 193 | 92.9730 | 93.2203 | 92.7461 | 157 | 77 |
| simple / c9408e3b | noise_0.5 | 4.5287 | 1.1259 | 21.4779 | 7.1111 | 129 / 127 | 94.5312 | 93.7984 | 95.2756 | 135 | 15 |
| predictive_proxy / e5df77dd | noise_0.5 | 5.0825 | 1.2675 | 20.4642 | 8.5000 | 153 / 153 | 88.2353 | 86.9281 | 89.5425 | 143 | 14 |
| sequential / 280977a9 | noise_0.5 | 6.7603 | 0.5864 | 23.9865 | 14.0833 | 256 / 251 | 83.6292 | 83.9844 | 83.2669 | 368 | 203 |
| sequential / 59a75d73 | noise_0.5 | 7.0659 | 0.7720 | 24.1975 | 13.1389 | 225 / 248 | 87.1036 | 89.7778 | 84.6774 | 295 | 152 |
| simple / c9408e3b | noise_1 | 3.3629 | 1.2847 | 19.0418 | 8.8056 | 157 / 160 | 72.5552 | 71.3376 | 73.7500 | 215 | 64 |
| predictive_proxy / e5df77dd | noise_1 | 3.7356 | 1.7237 | 19.0032 | 9.2222 | 167 / 165 | 74.3976 | 74.2515 | 74.5455 | 165 | 39 |
| sequential / 280977a9 | noise_1 | 4.8236 | 0.7654 | 18.2898 | 17.3889 | 314 / 312 | 67.4121 | 67.1975 | 67.6282 | 540 | 317 |
| sequential / 59a75d73 | noise_1 | 5.1534 | 1.1782 | 18.2685 | 19.3333 | 344 / 352 | 67.2414 | 65.9884 | 68.4659 | 571 | 293 |
| simple / c9408e3b | noise_2 | 1.5111 | 2.8646 | 5.3175 | 15.8333 | 283 / 287 | 53.3333 | 53.0035 | 53.6585 | 466 | 218 |
| predictive_proxy / e5df77dd | noise_2 | 1.2583 | 3.4954 | 6.3782 | 11.5556 | 203 / 213 | 49.5192 | 49.2611 | 49.7653 | 243 | 98 |
| sequential / 280977a9 | noise_2 | 1.5032 | 2.4355 | 5.7801 | 20.1667 | 357 / 369 | 44.9036 | 44.8179 | 44.9864 | 676 | 311 |
| sequential / 59a75d73 | noise_2 | 1.5000 | 5.3664 | 3.6913 | 23.7222 | 424 / 430 | 45.0820 | 45.2830 | 44.8837 | 725 | 332 |
| simple / c9408e3b | causal | -2.1615 | 9.9418 | -6.3977 | 10.7778 | 192 / 196 | 17.5258 | 17.7083 | 17.3469 | 166 | 67 |
| predictive_proxy / e5df77dd | causal | -1.4165 | 8.2395 | -2.7947 | 7.8611 | 142 / 141 | 20.4947 | 20.4225 | 20.5674 | 74 | 32 |
| sequential / 280977a9 | causal | -1.4641 | 7.4132 | -4.8937 | 9.9444 | 182 / 176 | 19.5531 | 19.7802 | 19.3182 | 224 | 64 |
| sequential / 59a75d73 | causal | -2.0801 | 9.6312 | -4.2880 | 12.8056 | 232 / 229 | 20.1735 | 20.2586 | 20.0873 | 236 | 86 |

The uncertainty term is report-sample dispersion, not a calibrated confidence level.
Mean Sharpe averages defined account Sharpes; it is not pooled portfolio Sharpe. Costs and horizon mix are in REPORT.json.
Orders/week uses market sessions, not ISO labels: Sunday and Monday belong to one operational week here. Native ISO metric is preserved in each account.
Harmful close is local fixed-position versus passive; it does not include subsequent-account opportunities.
Worlds share mirrored/envelope transformations; no independent-sample significance claim.
No optimizer parameters, data selection or model release are changed by confirmation.
Exact optimality is only over the declared finite plan library; actual future is unavailable to causal reports.
NOP earns zero with zero orders and is ineligible. Negative results are preserved.

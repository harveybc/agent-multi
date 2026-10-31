# Frozen-reference rollout contrast

Identity: `5930311292e4c6ef67eb2e9484bdaf92642ba85dfbe63657ee1619198f080af4`.
Seven frozen policies. Synthetic engineering, not financial validation. All costs retained.
Value is Q of an explicit reference continuation, not exact Q of the adaptive selector.
Carried forecasts preserve the current whole ensemble; no future real reports are read.

| Bank / reporter | Policy | Return % | Worst DD % | Mean Sharpe | Orders/week | Buy/sell | Win % | Buy win % | Sell win % | Early | Local harmful |
|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|
| confirmation_ideal | simple_c9408e3b | 2.71721 | 1.06064 | 20.31969 | 3.88889 | 68/72 | 99.28571 | 98.52941 | 100.00000 | 40 | 0 |
| confirmation_ideal | predictive_proxy_e5df77dd | 4.08470 | 0.90895 | 22.72885 | 6.61111 | 119/119 | 99.15966 | 99.15966 | 99.15966 | 91 | 0 |
| confirmation_ideal | sequential_280977a9 | 4.47049 | 0.68143 | 23.11598 | 7.80556 | 136/145 | 95.37367 | 95.58824 | 95.17241 | 162 | 112 |
| confirmation_ideal | sequential_59a75d73 | 4.77776 | 0.77994 | 23.27927 | 8.25000 | 147/150 | 94.61279 | 94.55782 | 94.66667 | 157 | 102 |
| confirmation_ideal | rollout_simple | 4.97334 | 0.74028 | 24.88826 | 8.86111 | 155/164 | 99.05956 | 99.35484 | 98.78049 | 178 | 67 |
| confirmation_ideal | rollout_proxy | 5.38907 | 0.69740 | 23.80686 | 10.41667 | 190/185 | 95.73333 | 95.26316 | 96.21622 | 235 | 98 |
| confirmation_ideal | rollout_joint | 5.37354 | 0.69740 | 24.39245 | 10.36111 | 187/186 | 95.97855 | 95.72193 | 96.23656 | 235 | 101 |
| calibration_noise_1 | simple_c9408e3b | 1.77974 | 1.55574 | 12.00502 | 7.69444 | 139/138 | 61.37184 | 60.43165 | 62.31884 | 193 | 52 |
| calibration_noise_1 | predictive_proxy_e5df77dd | 1.30045 | 1.58840 | 7.30338 | 7.02778 | 126/127 | 54.15020 | 53.96825 | 54.33071 | 149 | 42 |
| calibration_noise_1 | sequential_280977a9 | 1.29459 | 1.72491 | 8.14612 | 16.91667 | 296/313 | 48.11166 | 47.63514 | 48.56230 | 583 | 300 |
| calibration_noise_1 | sequential_59a75d73 | 2.14210 | 2.29803 | 11.50202 | 16.11111 | 291/289 | 53.96552 | 52.92096 | 55.01730 | 523 | 241 |
| calibration_noise_1 | rollout_simple | 0.30971 | 4.03962 | 0.47539 | 25.08333 | 444/459 | 38.09524 | 35.58559 | 40.52288 | 846 | 380 |
| calibration_noise_1 | rollout_proxy | -0.10491 | 4.13564 | -4.25965 | 24.66667 | 443/445 | 36.48649 | 37.24605 | 35.73034 | 837 | 423 |
| calibration_noise_1 | rollout_joint | 0.34644 | 4.09444 | 1.01350 | 25.11111 | 454/450 | 38.27434 | 35.24229 | 41.33333 | 839 | 371 |
| confirmation_noise_0.5 | simple_c9408e3b | 3.00509 | 1.02247 | 20.66452 | 5.58333 | 100/101 | 84.07960 | 84.00000 | 84.15842 | 114 | 25 |
| confirmation_noise_0.5 | predictive_proxy_e5df77dd | 3.24661 | 1.20612 | 18.95074 | 5.80556 | 106/103 | 88.99522 | 88.67925 | 89.32039 | 77 | 12 |
| confirmation_noise_0.5 | sequential_280977a9 | 3.74268 | 0.88008 | 21.44158 | 12.72222 | 226/232 | 75.54585 | 76.54867 | 74.56897 | 385 | 229 |
| confirmation_noise_0.5 | sequential_59a75d73 | 4.14297 | 0.72580 | 21.39152 | 11.33333 | 189/219 | 81.37255 | 85.18519 | 78.08219 | 320 | 194 |
| confirmation_noise_0.5 | rollout_simple | 2.65648 | 3.11596 | 15.93420 | 21.88889 | 385/403 | 56.59898 | 56.88312 | 56.32754 | 727 | 429 |
| confirmation_noise_0.5 | rollout_proxy | 1.58035 | 2.77492 | 10.15059 | 24.11111 | 422/446 | 50.34562 | 50.23697 | 50.44843 | 817 | 518 |
| confirmation_noise_0.5 | rollout_joint | 2.59322 | 3.07340 | 14.93800 | 22.27778 | 386/416 | 56.35910 | 56.99482 | 55.76923 | 747 | 444 |
| confirmation_noise_1 | simple_c9408e3b | 2.59865 | 1.36056 | 17.77443 | 8.36111 | 148/153 | 67.44186 | 66.89189 | 67.97386 | 200 | 65 |
| confirmation_noise_1 | predictive_proxy_e5df77dd | 2.42404 | 1.91589 | 14.92387 | 7.33333 | 128/136 | 66.66667 | 66.40625 | 66.91176 | 150 | 49 |
| confirmation_noise_1 | sequential_280977a9 | 1.95831 | 1.73050 | 11.00633 | 16.38889 | 289/301 | 53.38983 | 54.32526 | 52.49169 | 547 | 320 |
| confirmation_noise_1 | sequential_59a75d73 | 2.46411 | 1.76342 | 14.89163 | 17.16667 | 295/323 | 59.06149 | 59.32203 | 58.82353 | 553 | 299 |
| confirmation_noise_1 | rollout_simple | 0.73159 | 3.26249 | 3.82281 | 26.52778 | 476/479 | 41.88482 | 41.80672 | 41.96242 | 889 | 466 |
| confirmation_noise_1 | rollout_proxy | 0.72767 | 3.00222 | 3.39064 | 26.00000 | 466/470 | 42.30769 | 42.06009 | 42.55319 | 878 | 495 |
| confirmation_noise_1 | rollout_joint | 0.76414 | 3.52656 | 3.95247 | 26.22222 | 464/480 | 42.05508 | 42.02586 | 42.08333 | 876 | 451 |
| confirmation_noise_2 | simple_c9408e3b | 1.25941 | 2.87056 | 6.77138 | 13.83333 | 246/252 | 49.39759 | 48.37398 | 50.39683 | 406 | 175 |
| confirmation_noise_2 | predictive_proxy_e5df77dd | 1.34394 | 2.60562 | 7.15528 | 10.66667 | 193/191 | 48.43750 | 47.15026 | 49.73822 | 243 | 96 |
| confirmation_noise_2 | sequential_280977a9 | 0.28328 | 3.32068 | 0.87213 | 19.83333 | 348/366 | 39.91597 | 40.80460 | 39.07104 | 683 | 331 |
| confirmation_noise_2 | sequential_59a75d73 | 0.43887 | 3.38004 | 0.44870 | 22.00000 | 374/418 | 39.89899 | 40.37433 | 39.47368 | 715 | 328 |
| confirmation_noise_2 | rollout_simple | -0.66479 | 5.60649 | -4.58966 | 27.94444 | 498/508 | 33.79722 | 34.33735 | 33.26772 | 914 | 410 |
| confirmation_noise_2 | rollout_proxy | -0.42707 | 4.28455 | -2.84207 | 27.02778 | 486/487 | 34.12127 | 33.53909 | 34.70226 | 877 | 435 |
| confirmation_noise_2 | rollout_joint | -0.62560 | 5.15057 | -4.60650 | 27.88889 | 496/508 | 33.86454 | 34.27419 | 33.46457 | 912 | 407 |
| confirmation_causal | simple_c9408e3b | -0.72181 | 2.57643 | -4.12470 | 8.77778 | 156/160 | 31.64557 | 31.41026 | 31.87500 | 218 | 81 |
| confirmation_causal | predictive_proxy_e5df77dd | -0.03340 | 2.70045 | 1.55001 | 5.05556 | 90/92 | 41.20879 | 41.11111 | 41.30435 | 74 | 12 |
| confirmation_causal | sequential_280977a9 | -0.43595 | 1.82414 | -5.63870 | 8.72222 | 158/156 | 30.57325 | 29.74684 | 31.41026 | 286 | 111 |
| confirmation_causal | sequential_59a75d73 | -0.52531 | 2.68805 | -3.13347 | 9.61111 | 169/177 | 30.92486 | 30.76923 | 31.07345 | 238 | 77 |
| confirmation_causal | rollout_simple | -1.06713 | 4.30381 | -5.48077 | 13.86111 | 251/248 | 27.85571 | 27.88845 | 27.82258 | 375 | 102 |
| confirmation_causal | rollout_proxy | -1.46269 | 4.87688 | -6.39183 | 16.86111 | 314/293 | 28.17133 | 27.07006 | 29.35154 | 492 | 149 |
| confirmation_causal | rollout_joint | -1.15255 | 4.30381 | -6.02466 | 14.38889 | 258/260 | 27.41313 | 27.51938 | 27.30769 | 400 | 114 |

## Actual Adaptive Close Attribution

{"accounts": 36, "closes": 235, "pareto_relations": {"CLOSE_DOMINATES": 131, "HOLD_DOMINATES": 71, "TRADEOFF": 33}, "profit_harmful": 85, "profit_helpful": 150, "utility_harmful": 86, "utility_helpful": 149}

Positive account delta means CLOSE outperformed a one-tick HOLD followed by the same adaptive policy.
This is separate from the local passive-position comparison. No automatic strategy promotion.
World transformations are paired; aggregate means do not constitute independent-sample significance.
Account summaries and raw receipts retain costs, horizon mix, order sizes, activity and undefined Sharpe.

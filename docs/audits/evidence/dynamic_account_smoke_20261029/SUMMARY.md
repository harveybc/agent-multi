# Synthetic dynamic account Pareto experiment

Status: SMOKE_ONLY. Identity: `9becdd6030785b440659552a54cc148b2d1e3033fff6ad73ae59662290b91aba`.
This is simulated model output, not financial validation or a learned policy.
The frontier is empirical among evaluated policies, not a global optimum.

| Policy | Panel | Return % | Worst DD % | Orders/week | Buy / sell | Win % | Early closes |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| nop | calibration | 0.0000 | 0.0000 | 0.00 | 0 / 0 | 0.00 | 0 |
| nop | confirmation | 0.0000 | 0.0000 | 0.00 | 0 / 0 | 0.00 | 0 |
| equal_guarde | calibration | 1.5020 | 0.6477 | 2.67 | 14 / 18 | 93.75 | 8 |
| equal_guarde | confirmation | 2.1518 | 0.4034 | 3.00 | 18 / 18 | 94.44 | 6 |
| greedy | calibration | 2.1988 | 0.6190 | 3.58 | 21 / 22 | 100.00 | 24 |
| greedy | confirmation | 2.1911 | 0.4034 | 3.33 | 20 / 20 | 85.00 | 14 |
| only_3 | calibration | 3.0125 | 0.5945 | 6.42 | 39 / 38 | 93.51 | 42 |
| only_3 | confirmation | 2.9613 | 0.5545 | 5.67 | 34 / 34 | 85.29 | 25 |
| only_6 | calibration | 2.8685 | 0.5768 | 6.33 | 37 / 39 | 90.79 | 48 |
| only_6 | confirmation | 2.9499 | 0.5378 | 4.58 | 27 / 28 | 92.73 | 25 |
| only_12 | calibration | 2.0484 | 0.7168 | 4.17 | 24 / 26 | 88.00 | 29 |
| only_12 | confirmation | 2.0138 | 0.4034 | 3.33 | 20 / 20 | 85.00 | 16 |
| only_24 | calibration | 1.4786 | 0.6190 | 2.83 | 17 / 17 | 100.00 | 24 |
| only_24 | confirmation | 1.7660 | 0.3279 | 2.92 | 17 / 18 | 88.57 | 15 |
| 52c06cea29c9 | calibration | 2.5055 | 0.6526 | 4.50 | 25 / 29 | 92.59 | 28 |
| 52c06cea29c9 | confirmation | 2.2320 | 0.4034 | 3.50 | 21 / 21 | 85.71 | 16 |
| 4428b7a43784 | calibration | 2.1988 | 0.6190 | 3.58 | 21 / 22 | 100.00 | 24 |
| 4428b7a43784 | confirmation | 2.1911 | 0.4034 | 3.33 | 20 / 20 | 85.00 | 14 |
| ca17a25fcbb0 | calibration | 1.8387 | 0.5679 | 3.58 | 21 / 22 | 95.35 | 18 |
| ca17a25fcbb0 | confirmation | 2.3347 | 0.5025 | 3.92 | 22 / 25 | 97.87 | 13 |
| 3c59ecdb1b7c | calibration | 0.8733 | 0.2426 | 4.33 | 27 / 25 | 96.15 | 23 |
| 3c59ecdb1b7c | confirmation | 0.9627 | 0.2062 | 4.33 | 25 / 27 | 92.31 | 17 |
| d809a0fc3d25 | calibration | 0.6306 | 0.1138 | 3.42 | 21 / 20 | 100.00 | 21 |
| d809a0fc3d25 | confirmation | 0.6001 | 0.1168 | 2.67 | 16 / 16 | 93.75 | 11 |

Unique candidates: 60; nondominated policies: 10.
Return and drawdown are separate objectives. Confirmation never chooses the policy.
REPLAYS.json.gz holds native ledgers, close reasons, horizon allocation, costs and Sharpe.
REPORT.json holds finite-game matrices, unilateral deviations and actual variation of every gene.
SL/TP geometry is fixed to the shared median profiles in this allocation experiment.
Closing at the local position optimum and whole-account opportunity allocation are distinct objectives.
No claim of exhaustive coverage of every market condition, of a qualified reporter, or of deployment readiness.

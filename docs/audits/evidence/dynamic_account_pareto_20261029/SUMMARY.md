# Synthetic dynamic account Pareto experiment

Status: COMPLETE_EMPIRICAL_SYNTHETIC_FRONTIER. Identity: `098e871909f0c587b1cbe855cc49b93723ed63a03c74845a52dd9009071a966d`.
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
| 08fca639649c | calibration | 3.1169 | 0.4176 | 5.67 | 35 / 33 | 100.00 | 35 |
| 08fca639649c | confirmation | 3.0115 | 0.5378 | 4.50 | 27 / 27 | 98.15 | 20 |
| 7d161591d433 | calibration | 2.9290 | 0.3396 | 5.33 | 30 / 34 | 100.00 | 33 |
| 7d161591d433 | confirmation | 2.7712 | 0.5378 | 4.08 | 24 / 25 | 93.88 | 20 |
| 18c72e613b22 | calibration | 2.1927 | 0.2552 | 5.33 | 30 / 34 | 100.00 | 33 |
| 18c72e613b22 | confirmation | 2.0751 | 0.4043 | 4.08 | 24 / 25 | 93.88 | 20 |
| 0622b2f8d354 | calibration | 1.2067 | 0.1435 | 5.50 | 31 / 35 | 95.45 | 40 |
| 0622b2f8d354 | confirmation | 1.2281 | 0.2371 | 4.67 | 27 / 29 | 85.71 | 25 |
| 89a4e072655d | calibration | 0.6877 | 0.0815 | 5.58 | 32 / 35 | 97.01 | 39 |
| 89a4e072655d | confirmation | 0.7040 | 0.1348 | 4.17 | 24 / 26 | 96.00 | 23 |

Unique candidates: 17018; nondominated policies: 632.
Return and drawdown are separate objectives. Confirmation never chooses the policy.
REPLAYS.json.gz holds native ledgers, close reasons, horizon allocation, costs and Sharpe.
REPORT.json holds finite-game matrices, unilateral deviations and actual variation of every gene.
SL/TP geometry is fixed to the shared median profiles in this allocation experiment.
Closing at the local position optimum and whole-account opportunity allocation are distinct objectives.
No claim of exhaustive coverage of every market condition, of a qualified reporter, or of deployment readiness.

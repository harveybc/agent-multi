# Frozen close attribution

Identity: `726ebc8a21d7c268493a6765566865bef1868de3a8b4689a63d039f9d2dfed7e`.
Positive delta favors the original CLOSE over one-tick HOLD then frozen-policy continuation.
These overlapping interventions are not additive policy gains. Synthetic diagnostic; no training.

| Reporter | Policy | Closes | Account helpful | Harmful | Neutral | Close dominates | Hold dominates | Tradeoff | Mean account delta USD |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ideal | 59a75d73 | 157 | 79 | 78 | 0 | 60 | 55 | 42 | -0.168409 |
| ideal | 280977a9 | 148 | 78 | 70 | 0 | 64 | 51 | 33 | -3.689793 |
| noise_0.5 | 59a75d73 | 295 | 120 | 175 | 0 | 90 | 134 | 71 | -2.925446 |
| noise_0.5 | 280977a9 | 368 | 135 | 233 | 0 | 105 | 166 | 97 | -5.247856 |
| noise_1 | 59a75d73 | 571 | 214 | 357 | 0 | 184 | 282 | 105 | -7.293137 |
| noise_1 | 280977a9 | 540 | 163 | 377 | 0 | 145 | 299 | 96 | -5.691932 |
| noise_2 | 59a75d73 | 725 | 311 | 414 | 0 | 268 | 350 | 107 | -3.544955 |
| noise_2 | 280977a9 | 676 | 225 | 451 | 0 | 186 | 394 | 96 | -6.936330 |
| causal | 59a75d73 | 236 | 162 | 74 | 0 | 150 | 58 | 28 | 5.279030 |
| causal | 280977a9 | 224 | 177 | 47 | 0 | 164 | 42 | 18 | 9.689287 |

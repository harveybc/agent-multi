# Native Decision Error Attribution

Positive delta is worse. Counts are paired row evaluations, not independent trades.

| Fold | Seed | Endpoint | Switches | Helpful | Harmful | Neutral | Regret delta | Harmful despite lower MSE |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A | 19 | equal_updates | 26 | 14 | 8 | 4 | -0.029195 | 1 |
| A | 19 | equal_presentations | 38 | 20 | 13 | 5 | -0.014279 | 2 |
| A | 43 | equal_updates | 31 | 15 | 13 | 3 | 0.006555 | 5 |
| A | 43 | equal_presentations | 36 | 19 | 13 | 4 | -0.002097 | 0 |
| B | 19 | equal_updates | 37 | 14 | 19 | 4 | 0.019110 | 7 |
| B | 19 | equal_presentations | 36 | 15 | 18 | 3 | 0.018923 | 2 |
| B | 43 | equal_updates | 37 | 11 | 21 | 5 | 0.053946 | 11 |
| B | 43 | equal_presentations | 37 | 13 | 19 | 5 | 0.063984 | 5 |

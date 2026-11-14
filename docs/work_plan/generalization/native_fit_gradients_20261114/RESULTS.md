# Native FIT Gradient Conflict

Counts over four fold/seed pairs, not independent market samples.
Raw full-batch gradient geometry, not actual AdamW trajectory attribution.

| Scope | Updates | Old/new MSE conflict | Old/new gap conflict | Expanded descent harms old MSE | Expanded descent harms old gap | Level descent harms old gap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| output_only | 0 | 3/4 | 3/4 | 3/4 | 3/4 | 0/4 |
| output_only | 8192 | 4/4 | 4/4 | 4/4 | 4/4 | 0/4 |
| output_only | 32768 | 4/4 | 4/4 | 4/4 | 4/4 | 0/4 |
| full_head | 0 | 3/4 | 3/4 | 3/4 | 2/4 | 2/4 |
| full_head | 8192 | 4/4 | 4/4 | 0/4 | 1/4 | 4/4 |
| full_head | 32768 | 4/4 | 4/4 | 2/4 | 3/4 | 4/4 |

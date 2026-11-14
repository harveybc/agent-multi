# Actual AdamW Decision Preservation

Fixed proposal budgets; audit descriptive; no model promotion.

| Fold | Seed | Proposals | Accepted | Reduced | Rejected | Old FIT regret | Expanded FIT regret | Audit regret | FIT pass |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| A | 19 | 8192 | 207 | 49 | 7985 | 0.000419 | 0.225747 | 0.226759 | False |
| A | 19 | 32768 | 208 | 50 | 32560 | 0.000419 | 0.225747 | 0.226759 | False |
| A | 43 | 8192 | 329 | 126 | 7863 | 0.000089 | 0.223158 | 0.234646 | True |
| A | 43 | 32768 | 329 | 126 | 32439 | 0.000089 | 0.223158 | 0.234646 | True |
| B | 19 | 8192 | 534 | 147 | 7658 | 0.017245 | 0.229040 | 0.237895 | True |
| B | 19 | 32768 | 534 | 147 | 32234 | 0.017245 | 0.229040 | 0.237895 | True |
| B | 43 | 8192 | 221 | 94 | 7971 | 0.010464 | 0.247916 | 0.225862 | False |
| B | 43 | 32768 | 345 | 186 | 32423 | 0.010464 | 0.247450 | 0.225862 | False |

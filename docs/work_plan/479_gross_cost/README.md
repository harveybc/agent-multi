# 479: Does separating gross opportunity from execution cost fix the side bias?

No. It changes annual returns, but the 2022 selector is still almost exclusively short.
This is a paired diagnostic on archived folds, not a new independent confirmation.

## Protocol

Reuse the eight archived short/long prediction files of experiment 477, the same
14,360 eligible decision times, and the same calendar, sizing and fill simulator.
Each fold fits on the preceding semester; all fit exits precede the first
evaluation origin (timestamps and prediction hashes in [audit.json](audit.json)).
Compare six actions: buy or sell at 6, 24 or 48 hours, or abstain. The account
allows one position; this comparison does not introduce an active early close.
All variants pay the original simulator costs. This is a selector test, not
a comparison of execution with and without fees.

1. **Original net:** fit the original ridge on realized net outcomes.
2. **Net refit:** omit spread/ATR and slippage/ATR and refit net outcomes.
3. **Gross minus cost:** fit gross outcomes on the same reduced inputs. Gross
   is net plus friction on identical fills, not a new frictionless trajectory.
   Subtract estimated cost separately for each action and rank the net scores.
   Known entry costs are combined with the calibration semester's mean residual
   in price units for that side and horizon. Realized evaluation costs are not
   inputs. Across 24 year/side/horizon cells, mean predicted cost is 8.78 bp
   vs 8.76 realized; mean of cell-level absolute-error averages is 2.95 bp.
4. **Gross side first:** choose direction by gross score, then select a positive
   net opportunity within that direction or abstain. No fallback to the other side.

## Annual net account returns (%)

| Inputs | Selector | 2020 | 2021 | 2022 | 2023 |
|---|---|---:|---:|---:|---:|
| Context | Original net | 23.15 | -11.27 | 0.67 | -18.08 |
| Context | Net refit | 3.95 | -9.31 | 0.12 | 0.59 |
| Context | Gross minus cost | 15.84 | -3.87 | -1.36 | 8.23 |
| Context | Gross side first | 15.62 | -1.00 | -1.36 | 8.23 |
| Context + seasonal | Original net | -0.78 | -8.71 | -9.95 | 27.56 |
| Context + seasonal | Net refit | 10.60 | 25.11 | 0.16 | -1.11 |
| Context + seasonal | Gross minus cost | 9.86 | 20.22 | -1.03 | 1.69 |
| Context + seasonal | Gross side first | 0.56 | 21.24 | -1.03 | 0.99 |

In 2022 the context baseline executes 0 buys/250 sells; the net refit executes
1/93; gross minus cost executes 1/94. The seasonal counterparts give 4/200,
2/133 and 2/135. Before single-position arbitration, the context net refit
already proposes 29 buys vs 3,500 sells. Position occupancy is not the sole
explanation. Neither does this establish that the sells were all incorrect.
The worst annual realized drawdown still exceeds 32% in every modified arm.
Separating costs beats the simpler seasonal net refit in only one of four
years. Do not promote a selector or launch another parameter search from this.
Next diagnostic: compare the distribution of paired buy-minus-sell gross
labels in calibration and evaluation with the predicted margins, by horizon.
This distinguishes a real directional imbalance from extrapolation by the model.

## Reproduction

Run from the repository root with the 477 run directory, 397 history,
market, and bundle paths used for the baseline:

```bash
OPENBLAS_NUM_THREADS=1 PYTHONPATH=. python -m tools.role479_gross_cost "$RUN477" "$HISTORY397" "$MARKET" "$BUNDLE" "$OUT"
```

130,638 action/fill checks and 1,100 original booked orders pass. A linearity
check confirms that subtracting a cost ridge with identical inputs and penalty
from the gross ridge reproduces the direct net ridge. [Accounts](accounts.csv)
include directional orders, wins, proposals and drawdowns; [books](books.csv.gz)
contain individual trades and [costs](costs.csv) the per-action estimates.

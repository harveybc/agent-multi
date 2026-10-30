# Median lane completion and frozen gate diagnostic

2026-10-30. Implements plan156. This is synthetic engineering confirmation,
not financial VALIDATION, RL training, or a released trading strategy.

## Completed lane

Source identity:
`452931b1fe97840255ac60abe87911a97724053ffc7d6c1249318c5c5026d36e`.
Population20 x300 generations x3 seeds (7,19,43);15,472 distinct candidate
receipts.209 runtime tests passed. Completed at09:03:35 UTC. Confirmation
contains108 synthetic accounts per policy, including matched no-close controls.
The collector authenticated the complete artifact set and receipt chain.

Representatives F1-F4 were frozen from calibration, not selected using this
confirmation. Percentage return is the mean across synthetic accounts; DD is
the worst account drawdown; Sharpe is the mean of defined native account Sharpes.

| Frozen policy | Calibration return % | Confirmation return % | Worst DD % | Mean Sharpe | Orders | BUY / SELL | Win % | Orders/week | Early closes |
|---|---:|---:|---:|---:|---:|---|---:|---:|---:|
| F1 c3f252db | 0.3738 | -0.2398 | 2.1584 | -3.1464 | 208 | 102 / 106 | 28.3654 | 1.9259 | 0 |
| F2 e16e4799 | 0.3695 | -0.2122 | 1.8812 | -3.2019 | 204 | 100 / 104 | 26.9608 | 1.8889 | 0 |
| F3 4496afdb | 0.3652 | -0.2214 | 1.9005 | -3.3297 | 204 | 101 / 103 | 26.4706 | 1.8889 | 0 |
| F4 a29b2bb2 | 0.3574 | -0.1868 | 1.8812 | -2.8485 | 202 | 98 / 104 | 28.7129 | 1.8704 | 0 |

All four had73 orders over36 calibration accounts:2.02778/week, just above
the strict >2 floor. All four fail that floor in confirmation. They are NOT
admitted by this report and no threshold is changed to make them pass.
No-close copies reproduce their account metrics: the closer contributes no
executed action. The search was not blocked by a performance gate.

The fixed `neither` control has5.50 orders/week, return-0.6207%, worstDD4.9836%,
246 early closes, of which90 are harmful under the fixed-position comparison.
The fixed `both` control has2.45 orders/week, return-0.1815%, worstDD2.6290%,
but also zero early closes. Neither is evidence of a qualified strategy.

## Additional work completed without changing either campaign

Replayed all432 finalist accounts from their archived world specifications.
The observational instrument reproduces every trace, trade, return, drawdown,
cash, cost and direction field exactly. It decomposes gates on those SAME
visited states; it does not run a new policy on confirmation or optimize there.

| Policy | Flat decision ticks | Entry selected before history penalty | Entry selected after penalty | Entry becomes NOP | Bids without mature history |
|---|---:|---:|---:|---:|---:|
| F1 | 6,990 | 3,911 | 208 | 3,703 | 15,068 |
| F2 | 7,102 | 4,098 | 204 | 3,894 | 15,050 |
| F3 | 7,174 | 3,996 | 204 | 3,792 | 15,026 |
| F4 | 7,170 | 4,107 | 202 | 3,905 | 15,056 |

The missing-history column counts individual bids (eight possible per tick),
not ticks or missed trades. The before/after column holds the other genes
fixed; it includes the entry uncertainty/time/score thresholds. A shadow
entry is not proof that the hypothetical new trade would be profitable.

| Policy | Occupied decisions | Raw close advantage passes margin | After uncertainty | After history error | Waiting for persistence | CLOSE requests |
|---|---:|---:|---:|---:|---:|---:|
| F1 | 3,378 | 117 | 15 | 9 | 9 | 0 |
| F2 | 3,266 | 89 | 44 | 14 | 14 | 0 |
| F3 | 3,194 | 26 | 9 | 0 | 0 | 0 |
| F4 | 3,198 | 73 | 20 | 1 | 1 | 0 |

All occupied decisions had mature history. The fixed diagnostic order is raw
expected advantage -> uncertainty penalty -> history-error penalty -> threshold
and persistence. This order is not an order-independent causal attribution.
All finalists use persistence3. F1/F2/F4 have some qualifying isolated ticks,
but none reaches required persistence or the urgency override; F3 has none
after penalties. Missing history is therefore NOT the observed direct cause
of zero closes in these occupied states. The controller does evaluate them;
it repeatedly chooses HOLD.

## Interpretation and next useful contrast

1. Calibration optimization concentrated immediately above the activity floor;
   activity did not generalize to the frozen confirmation worlds.
2. The current close rule suppresses every executed early close. This is an
   observed mechanism failure for the intended behavior, not a claim that the
   data cannot contain useful signal or that all other designs were exhausted.
3. Lowering all thresholds blindly is not justified: the permissive control
   already produces90 harmful closes. Next measure fixed-position close regret
   for the suppressed proposals against the passive and optimal native exits.
4. Keep that local close diagnostic separate from close-plus-next-opportunity
   value in plans152-155. Continue the sequential allocator against its exact
   references; the local-score baseline is not that full mechanism.
5. Any new coefficients or policy changes require new calibration and fresh
   confirmation worlds. These observed results cannot become a tuning set
   while retaining the label of untouched confirmation.

## Evidence and reproducibility

[DIAGNOSTIC.json](../audits/evidence/median_frozen_gate_20261030/DIAGNOSTIC.json)
contains all genes, counts, calibration/confirmation statistics, horizon mixes,
source completion digest and diagnostic code identities.
[Original completed results](../audits/evidence/median_frozen_gate_20261030/RESULTS.md)
are copied unchanged from the authenticated collection.

47 focused tests passed with zero skips, including staged arithmetic, nested
source-manifest tampering, warm/cold observational non-interference, controller
state, causal history and queue regressions. Scoped Ruff and diff checks pass.
The first serial probe completed F1, then was deliberately interrupted solely
to reuse common reports across finalists. The final run used four bounded CPU
workers at lower scheduling priority; all432 account checks completed and F1
counts match the serial probe exactly. No scientific worker was stopped.

```bash
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python tools/diagnose_frozen_history_policy.py --source <authenticated-completed-lane> \
  --engine-root <pinned-native-engine> --output <new-diagnostic-root> --workers 4
```

The zero-drift search and automatic collector remain active. Its final result
is not inferred from the completed median lane. No learned training, floating
selection or strategy promotion was launched by this diagnostic.

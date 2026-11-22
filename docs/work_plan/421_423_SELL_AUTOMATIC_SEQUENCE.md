# 421-423: SELL value, closer dependence, horizon interference

Research sequence before further feature removal. Frozen market-only
representation, same signed managed target, shadow/research accounts only.
Use H2 2022 and H2 2023 inside TRAIN, including all three seeds. Never
select features, deploy weights or change the broker from these runs.

- **421** uses the exact 420 matched support, with a shared managed MSE
  monitor for all heads. Compare sign (probability converted to bp using
  train conditional means), direct mean signed bp, and a three-output
  model of win probability and conditional gain and loss. Managed payoff
  scaling is fit on training rows only. The magnitude heads preserve the
  four horizons, market encoder and SELL-only action constraints. All arms
  use identical minibatch permutations and training length, restore their
  own validation-best weights, use the same AdamW/rollback policy as 420.
  Shared economic MSE is different from the managed BCE monitor in 420;
  do not use historical 420 account values as a paired sign control.
- **422** holds these checkpoints fixed, crosses the opening seed with
  each closer seed, verifies all 36 own-closer/gate replays, and measures
  lift over selecting the same expected number of opportunities in each
  week and horizon. For each of two gates and three arms there are 18
  cells: 2 semesters x 3 opening seeds x 3 closer seeds. They overlap in
  time and are not independent replications. Rank gate keeps the top H1
  decile threshold; positive-EV gate also requires predicted value >0.
  Matched-volume bp are a separate offline ranking diagnostic, unaffected
  by that gate. No 421 arm qualified. The conditional gain/loss arm beat
  matched random selection in 15/18 but matched mean net was negative in
  18/18. Its positive-EV account had 2/18 positive active returns.
- **423** tests the conditional gain/loss method with shared width 8,
  shared width 32, or four independent per-horizon width-8 readouts.
  Wide and split start functionally identical (asserted) via block-diagonal
  output initialization; the wide version has additional trainable
  cross-horizon connections. The paired small head is refitted, not assumed
  identical to 421. All other inputs, splits, policies, gates and evaluator
  stay fixed. To qualify for replication an arm must beat zero and the
  smaller shared control for matched bp and account return, with >2 trades
  per native week, in every seed/semester. Split must also beat wide.

Runtime: 413 CPU environment. Reusable 421/422 runner:
`bash tools/run_421_422_sell.sh`; third stage:
`python tools/role423_sell_horizons.py` in the same environment. Outputs:
`/var/tmp/role421-sell-value`, `/var/tmp/role422-sell-cross`,
`/var/tmp/role423-sell-horizons`. Each stage produces DONE.json on verified
completion. 421 and 423 also write per-seed weights, histories and REPORT.md.

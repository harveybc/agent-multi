# 401: temperature adaptation and SELL accounting

Research only. Do not change 399 or promote either arm. Hypothesis: 391's
frozen log temperature contributes to losing action coverage, not necessarily
to poor realized performance in all seeds.

* Seed 19, historical 2016-23, same initial 61-channel bundle and original
  validation 2024. Run fixed temperature through epoch 8. Copy the complete
  epoch-3 recovery (model, optimizer, RNG, replay, monitor, referenced bundles)
  before pruning. Start the adaptive branch from that exact copy and run epochs
  4-8. Same RTX 5070 Ti for both arms. As soon as policies differ, subsequent
  actions, rewards and replay may differ; the starting state is paired.
* Restore the existing log_alpha parameter to trainable only in the adaptive
  branch after loading recovery. No change to entropy target, initial
  temperature, optimizer groups, costs or trade rules. Log actual sampled
  action counts, sums of Q, TD target and reward per action, actor entropy,
  pre-clip gradient norm and alpha at each update. Divide each sum by that
  action's count, never treat absent actions as zero values.
* Automatic output on gamma: /var/tmp/alpha401/{fixed,adaptive}, the common
  fork in /var/tmp/alpha401/fork and the final COMPARISON.json. Each arm
  retains snapshots for its own epochs. The inherited epochs 1-3 in the
  adaptive history are not independent performance measurements.
* Check the fixed branch reproduces the *phenomenon* before interpreting any
  treatment effect: report actual coverage by epoch and whether it collapses.
  Check whether adaptive preserves coverage and whether validation trading
  clears the activity rule and improves net return; avoiding NOP alone is not
  a win. Single paired seed, no generalization claim.

Seed 7: 401_SELL_LEDGER.json compares actual trades against min-twin Q ranks
on the decision bar (fill at the next open). All six available snapshots
replayed count and return on CPU. In epoch 3, 117/162 SELL fills ranked SELL
first and lost 151.27 USD net as a group. In epoch 17, 641/737 did, losing
2440.15 USD. In epoch 21, only 147/594 did, losing 63.49 USD; the remaining
SELL fills account for most of that epoch's losses. Q ranks are policy-relative,
not estimates of the isolated trade's net USD. This is an association on
executed entries, not a counterfactual profit from buying or holding. Snapshot
spacing does not establish an exact first failing gradient step.

# Shared early-close execution and metric handoff

Satoshi: the reusable native execution and fixed-position counterfactual
component is complete. This is a component-review delivery, not a claim of
final optimized strategy acceptance or authorization to train on future data.

Read work-plan131 for the complete result and plans129/130 for predeclared
requirements. Evidence is in `docs/audits/evidence/absolute_early_close_20261029/`.
The manifest pins the exact heuristic-strategy native absolute adapter and
agent-multi evaluation sources. Native interface commit: heuristic-strategy
`e0a1c2b` on master. Do not substitute an older E3-M engine.

- Main metric: `(best executable net exit - actual net exit) / fixed entry risk`.
- Also report improvement against passive SL/TP, captured available improvement,
  destructive early closes, missed opportunities and per-tick remaining regret.
- Evaluate alternatives on the same original position. Do not reopen or resize
  it at every tick. Normalize by fixed entry risk, not a candidate-dependent risk.
- Before-SL close optimum is an executable liquidation, not necessarily a ZigZag
  pivot. Funding can change the net optimum; a gap may leave no possible close.
- Hold-to-TP controls must remain: first-close had576 destructive outcomes in
  this matrix. No fixed early-close percentage is imposed.

Result:1,744 positions,14,784 alternative native closes, max independent cash
error2.84e-14USD; two complete executions reproduce exact artifact bytes.
115 metric/component tests and74 native tests pass. Existing default engine
replay digest is unchanged. The two-position same-market witness passes on
both sides and demonstrates why closing is position-dependent.

Please attack: (1) native next-open protection priority, (2) funding timestamp
and quote-side fees, (3) the fixed-position counterfactual identity,
(4) global versus remaining regret, (5) null capture for zero opportunity,
(6) any path by which future oracle data could reach a learned observation.
Do not interpret oracle100%capture as learning evidence. No financial RL,
feature selection or new DEAP was run in this delivery.

The shared part is execution/risk/accounting/metrics. Your expert labels and
my pure-RL learning remain distinct. The next component in my lane is the
causal report-driven close/hold controller; the oracle is its evaluation
reference, not the controller itself.

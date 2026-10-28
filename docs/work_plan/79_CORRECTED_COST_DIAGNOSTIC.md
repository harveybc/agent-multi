# Corrected engine and easy-mode contrast

Owner direction, 2026-10-26: proceed with corrected actual-model selection and
retest whether zero costs are useful diagnostically. Old easy-mode failures
do not establish lack of signal in the corrected system; also do not assert
without an experiment that each defect caused those failures.

## Execution order

1. Correct five audit witnesses in a new simulator version. Never change the
   frozen engine or reinterpret old financial results as corrected results.
2. Integrate open-financing equity into the actual categorical RL wrapper and
   verify native reward/accounting chronology. No feedback inputs added.
3. Qualify/freeze the proposed small temporal architecture and implement the
   actual-model structural-subset adapter for feature_selector. No proxy MLP.
4. Fresh paired configured-cost and zero-cost training, same candidate recipe,
   admitted support, TRAIN scaling, calendar, seed set and step budget. Evaluate
   each fixed policy under BOTH cost modes. This four-cell contrast distinguishes
   training effects from fixed-policy cost sensitivity. Re-run the full simulator
   in each cell: subtracting fees from an old profit does not reproduce balance-
   dependent sizing, stop-out or reward trajectories.
5. Use easy-mode subsets only as diagnostic proposals, with original candidates
   still available for rescue/reinclusion. Cost-aware floating selection and
   confirmation use configured costs and all causal/profile/encoding stages.

Zero-cost means no spread, slippage, commissions, swap debit OR credit. Keep
mandatory SL/TP, early closes, fractional balance sizing, leverage/margin,
broker minimums, entry delay, limit-through and business calendar. No synthetic
price filling, revised labels, reward bonuses for activity or new observation
feedback. No TEST. No new costs/sizing parameters silently optimized.

## Implementation and evidence

- heuristic-strategy candidate branch `codex/execution-cost-modes-20261026`,
  commit `cd6489f`, based on `89512e8`: event_costs_v2. Open financing accrual, event-time fill
  costs, actual margin liquidation, close-stamped equity and minimum distance
  recheck at actual fill. Position size also uses actual fill price, matching
  the categorical wrapper's existing sizing behavior.
- 19 initial CPU witnesses: 18 fail against the frozen engine, 1 already passes
  (nearest protective stop). All 19 pass against the correction. Three further
  tests cover financing-induced margin close, no same-bar reversal after gap
  liquidation and tampered financing events.
- Engine regression group: 100 passed (22 new plus 78 existing).
- `test_corrected_engine_rewards.py` fails before the wrapper change: reward
  0 instead of rollover debit -0.0636363 in its synthetic long fixture. After
  forwarding to engine.equity_at, seven cross-repository tests pass, including
  actual synthetic training, save/reload, order and reward chronology.
- feature_selector 0.2.1: 38 passed, four historical skips. Cost mode/profile
  hash and component totals checked; zero-cost cannot be final_selection.

The engine's unknown intrabar exit time remains explicitly an OHLC approximation:
bookkeeping at bar end, cost quote at bar start. Known open/close events use
their actual boundary time. This is not a bid/ask tick execution certification.
Configured rates remain assumed, not broker-account-verified.

## Current state

Corrections and easy-mode contracts tested on CPU. No new financial fit, no
running GPU selection, no new winning feature list. Native subset integration,
temporal baseline qualification and the paired financial contrast remain next.
The earlier 120-column recipe is a candidate baseline, not a newly selected
subset. Keep the expert-guided lane and its experiments unchanged.

# Recipe R3 Native SAC Queue

Implementation design, 2026-10-24. Supersedes the old rolling observation
campaign, not its historical results. Owner authorizes completion and launch.

Reuse the shared recipe's `BarEnv` accounting engine directly, pinned from
heuristic-strategy. This is an explicit execution backend choice, not a claim
of equivalence of the unfinished Backtrader cost adapter. No expert loader,
labels, demonstrations, feedback observations or expert policy is consumed.
The native SB3 actor and twin critics consume the exact frozen 120-column
market tensors. The selected grouped encoder is the first arm; a flat market
window control follows on the identical inputs, support, seed and budget.

## Fixed Experiment

- Seed 7, 100000 policy decisions per arm; 25000-step checkpoint intervals.
- No early stopping, refit, checkpoint selection or TEST. TRAIN is recipe TRAIN;
  INNER_A and OBS_2024 are frozen-policy reports, not optimizer inputs.
- AdamW, learning rate 0.0003, weight decay 0.01; grouped branch dropout 0.3
  exactly as selected. Flat control has no grouped branch; disclose this as a
  representation-package contrast, not an isolated causal architecture effect.
- SAC gamma 0.99 per observed bar, batch 64, replay 20000, learning starts
  1000, train frequency 1, gradient steps 1, tau 0.005, automatic entropy.
- Three actions: target side with 1/3 dead band, TP and SL from recipe bounds.
  Fixed order fraction 1.0 of the recipe's 0.1 balance margin budget, leverage
  50. The native business engine enforces all recipe costs and calendar rules.
- Every market bar stays in accounting. Missing observations cause automatic
  HOLD (not target-flat), so existing protections and forced exits still run.
  Prefix without observations is flat warmup. Between valid observations,
  accumulate sum(gamma**j * reward_j); bootstrap with gamma**n. Closure hours
  are not fake bars. Replay stores only actual valid market tensors. A terminal
  placeholder reuses the last valid tensor, explicitly marked and never used
  for a policy decision or bootstrap (done=1). No zero padding of market data.
- Reward is 100 * equity change / initial capital, using the shared marked
  equity convention (financing settled at close); reports use native accounting.
- Bounded retries restart an interrupted arm from seed in a new attempt,
  never claim exact continuation without RNG/replay/environment recovery.
  Completed authenticated arms are reused. Sequential service survives logout
  and reboot; status, checkpoints, reports and content-addressed outbox are
  durable. A second-machine collector copies artifacts without agent polling.

## Tests Defined Before Code

| ID | Structural and behavioral acceptance |
| --- | --- |
| N01 | Exact feature-only observations, no future/NaN substitutions |
| N02 | Missing intervals visit every engine bar and use HOLD, even when open |
| N03 | Discounted rewards and replay discounts equal analytical n-bar sums |
| N04 | Terminal invalid observation never bootstraps; prefix remains flat |
| N05 | Direct engine and wrapper yield identical trades, equity and costs |
| N06 | Next-open execution, per-day cap, mandatory SL/TP and forced-flat |
| N07 | Native SAC update and save/reload on CPU, no labels and no feedback |
| N08 | Immutable spec/input/code pins, no TEST, bounded automatic recovery |
| N09 | Fixed checkpoints, inference preserves training RNG/optimizer state |
| N10 | Real GPU admission, actual training heartbeat, automatic next arm |

No profitability or estimated end time is established by these engineering
checks. Costs remain ASSUMED_NOT_BROKER_VERIFIED. Selection used development
data; these reports are development, not independent confirmation.

## Executed Deployment

Code 2d4aaa99; shared engine 89512e8. Spec SHA256
7568df4f3ba7b6e80fb86cce70c42adeb9b4d0b6a227fd947183baa0ba8518cc.
25 focused tests pass, plus 16 existing engine tests. Synthetic episode wrapper
and direct engine match trade ledgers, costs, profit, Sharpe and drawdown exactly;
native CPU training includes intervals of unavailable observations and exact
checkpoint reload. Real 5090 admission: both arms, 14 gradient updates each on
synthetic observations, exact reload, peak 70182912 allocated bytes.

Production GROUPED started 2026-10-24T21:41:14Z. At 21:42:47Z: 5500 decisions,
4499 gradient updates, 92.75 seconds, about 2.2 GiB service RAM. This proves
training, not a completed measurement. One GPU worker only. Sequential timer is
enabled across reboot; collector on a second machine passed its first copy and
empty-outbox pump (zero expected/delivered/pending reports at this stage).

The complete real-data report path was separately exercised on CPU with HOLD
only, explicitly ENGINEERING_NO_TRADE_REPORT_PATH: TRAIN 21808 engine bars,
INNER_A 3085, OBS_2024 6220. Unavailable-window HOLD bars: 307/281/13;
TRAIN additionally has 281 flat warmup bars. All price bars retained. The last
bar is terminal rather than a new trading decision. No scientific score is
inferred from that no-trade witness.

TRAIN means FIT 2020-01-01 to 2023-07-01 in the shared consolidated contract;
INNER_A is 2023H2; OBS_2024 is the full validation year. Both are reported
without weight updates. No claim that the FIT subset is all four calendar years.
The original recipe's one-read intent applies to selection: this predeclared
SAC learning curve has four development-only observations of 2024 and does not
select checkpoints from them. Final100000 is mandatory, including if it worsens.

Native reports include profit USD/percent, annualized daily Sharpe (sqrt365),
drawdown, trades, win fraction, costs, activity and mean daily log return. Gap
tables compare TRAIN versus the other periods in daily units. Weights and
trades are retained per checkpoint. Undefined metrics are null, not zero.

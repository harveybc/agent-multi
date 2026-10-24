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

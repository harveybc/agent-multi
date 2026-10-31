# Automated Probe Follow-Up

Owner authorization:2026-10-31, sequential unattended continuation after173/176.
No in-flight source, policy, reward, feature selection or evaluation period changes.

## Sequential Queue

1. Await authenticated closure of all12 probe outcomes and the fixed-position
   close diagnosis. Require matching campaign identity and closure hashes.
2. For each completed model and each of TRAIN/INNER_A, reload sealed market-only
   emissions and replay the original policy. Require exact metrics, actions and
   trade-ledger identity against archived evidence before any comparison.
3. Replay a diagnostic ablation that retains the identical opening logits and
   replaces only occupied-position curves with HOLD. Native SL/TP, costs,
   forced liquidation and opening restrictions remain enforced. This is NOT
   a proposed production strategy: early close remains a business requirement.
4. Characterize policy outputs and twin critics by period, learned temperature,
   action probabilities and actual one-step reward scale. Never interpret
   softmax probability or cross-seed disagreement as calibrated uncertainty.
5. Run an official TorchRL target calculation on an economic-zero witness:
   zero Q values, equal logits, identical reward and gamma, but2 versus3 legal
   actions. Check the target difference against gamma*alpha*log(3/2). This proves
   a possible action-count incentive, not its causal dominance in actual runs.
6. Produce complete JSON/Markdown results, seed-grouped summaries, source and
   artifact manifest, next controlled experiment criteria and audit request.

## Tests Frozen Before Implementation

- Suppression preserves every opening logit and cannot generate agent CLOSE.
- Native baseline must reproduce sealed metrics, actions and ledgers exactly.
- A changed closure hash or unit receipt must be rejected, not ignored.
- The entropy witness has zero economic reward; only legal-action count changes.
- Every account result includes costs, exposure/activity, both directions,
  win rates, drawdown and return; absent Sharpe/support remains null.
- Resume authenticates finished units, never overwrites a completed receipt.
- No diagnostic recommendation can start new training or floating selection.

All baseline and ablation comparisons use the same fixed model and calendar,
but downstream orders can differ because holding occupies the slot. This
whole-account comparison complements, not replaces,176's fixed-position effects.

## Decision Boundaries

No minimum holding interval, confidence cutoff or forced trading reward is
invented. Report action collapse, activity failures and harmful closes even if
some aggregate improves. Use TRAIN evidence to specify subsequent corrections;
INNER_A remains diagnosis, never a hidden tuning loop. Upstream selection and
risk design already used INNER_A;2024 andTEST are not opened.

Potential next paired training control: unchanged baseline versus explicitly
normalized entropy over legal actions, only after a sequential positive-control
environment verifies timely profitable HOLD, protective CLOSE and NOP in both
directions. This queue prepares that decision; it does not silently deploy it.

## Preflight Evidence

The economic-zero official TorchRL target witness passes, and an actual3h
seed43 INNER_A baseline reproduces every metric, action and trade exactly.
Its no-close diagnostic improves return1.767percentage points but increases
maximum drawdown1.386points and executes8 fewer orders: a measured tradeoff,
not evidence for disabling required early close. This isolated preflight is
not used to choose parameters. Full comparisons are run for all models.
22 targeted tests pass with no skips; scoped Ruff passes. Prerequisite closure
was verified:12 financial probes at08:42:42UTC and24 fixed-position diagnoses
at08:44:21UTC. The bounded CPU follow-up uses a separate persistent service,
authenticated per-unit resumption and a final artifact manifest.

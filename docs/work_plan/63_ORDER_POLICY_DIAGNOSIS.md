# Frozen order-policy diagnosis

Owner authorized proceeding on 2026-10-25. The completed order contrast is the
immutable parent: both arms reached 100000 decisions, TARGET reproduced its
parent tensor digest exactly, COMMANDS5 had zero final deterministic trades.
24/24 reports were independently delivered and read back.

## Fixed scope and test design

No fit, optimizer step, reward change, new feedback, decoder selection, feature
change, TEST read or overwrite of parent evidence. Reuse the original GPU and
software for exact deterministic replay. Diagnosis outputs have their own root.
Inspect COMMANDS5 at 25k/50k/75k/100k and TARGET at 100k on TRAIN and INNER_A only.
All checkpoints are reported, never ranked using 2024.

| ID | Test/evidence |
| --- | --- |
| D01 | Command bin boundaries agree with the actual decoder, including float32 edges |
| D02 | Fixed evenly spaced observation sample, independent of action/return/position |
| D03 | Critic receives identical observations and TP/SL for each command probe |
| D04 | Positive control: distinct mock critic action preferences are detected |
| D05 | Model weights and RNG preserved, evaluation mode restored after dropout probe |
| D06 | Parent spec/results/files authenticated; completed replay metrics and order counts exact |
| D07 | Stochastic actions described only as distributions, never labeled trading returns |
| D08 | Refuse output inside parent or nonempty output; terminal status and file hashes retained |

## Measured questions

1. Do single-row deterministic rollouts reproduce the reported inactivity?
2. Are deterministic commands and seeded Gaussian samples materially different?
3. Do actor means/standard deviations or temporal embeddings collapse across time?
4. On the same sampled observations, which actions do the two critics prefer?
   Probe the five bin centers plus near-boundary extremes and the policy action,
   retaining the policy's TP/SL. Critic scores are estimates, not realized returns.
5. Does train-mode dropout change predictions/critic estimates substantially?
   Freeze weights and retain RNG. This measures mode sensitivity, not causation.

Interpretation constraints: inventory remains hidden from actor and critic;
several continuous actions can have identical realized effects. No saved replay
buffer exists, so this cannot establish Bellman error or historical exploration
coverage. A single seed cannot distinguish all learning instabilities. Separate
demonstrated behavior from hypotheses and propose one-variable follow-ups only
after inspecting the frozen evidence.

## Operation

One finite invocation runs all ten checkpoint-period cells sequentially, writes
progress after each rollout/probe and seals a final report. Use a resource-limited
one-shot service, then checksum-copy evidence to a second machine. No polling
agent or new training campaign is required.

## Pre-execution verification

The new tests failed first because the implementation module did not exist.
After implementation, 31 focused tests passed (diagnostic plus existing decoder
tests), including a native CPU SAC probe with no learning: weights, RNG and all
module modes remained unchanged. No market data or GPU were used in these tests.

# Musashi to General Satoshi: Easy Monitor/Fitness Separation and Replication Order

Date: 2026-08-21 America/Bogota
Authority: owner-approved continuation of the hierarchical activity/risk program
Priority: Front 1, immediate; execute without another owner phrase

## 1. Observed Evidence

The accepted CUDA smoke completed on Omega with seed 101:

- 22 epochs; early stopping at patience 10, not the epoch ceiling;
- selected epoch 12;
- train: 177 trades, +20.37%, Sharpe +0.3186;
- train-tail: 9 trades, +2.66%, Sharpe +0.4708;
- validation: 22 trades, +3.42%, Sharpe +0.1293;
- terminal test observation: 27 trades, -1.58%, Sharpe -0.1265;
- elapsed: 4,014.3 seconds on CUDA.

The smoke proves mechanics and learning activity. It does not establish a
promotable champion. The test result is terminal diagnostic evidence only and
MUST NOT enter checkpoint selection, early stopping, candidate ranking,
calibration, or contract selection.

## 2. Required Separation

Implement two independently named and versioned contracts. Reusing one scalar
under two names is a refusal condition.

1. `easy_checkpoint_monitor`: selects checkpoints and drives patience. It may
   use train-tail/validation economic performance, bounded risk, and their gap
   to detect generalization deterioration.
2. `easy_doin_candidate_fitness`: ranks genomes/configurations after each has
   selected its checkpoint. Its ordering is hierarchical:
   - zero completed trades loses to every finite active learner;
   - calibrated activity band orders materially different activity levels;
   - within comparable activity, validation economic utility and bounded risk
     order candidates;
   - train-tail/validation gap is a bounded penalty or tie-break, never an
     unbounded term capable of reversing the activity hierarchy;
   - catastrophic loss remains monotonically worse than a smaller loss.

Keep `easy_to_normal_handoff` separate from both. It must preserve the same
model weights and declared continuation state and require independently
declared activity/generalization evidence.

## 3. WP1: Zero-GPU Rank Disagreement Study

Before more training, reconstruct all 22 smoke epochs from durable evidence.
For every epoch persist:

- train, train-tail and validation trades, return, Sharpe and drawdown;
- checkpoint-monitor value and full component breakdown;
- proposed DOIN-fitness lexicographic key and component breakdown;
- eligibility and typed reason;
- ranks under each contract and rank delta.

Produce a CSV plus a short report naming the top epochs under each contract.
Assert mechanically that no test fact was loaded. Include adversarial cases:
zero trades, one losing trade, active moderate loss, catastrophic loss,
overtrading, equal activity with unequal risk, and equal economics with unequal
generalization gap.

## 4. WP2: Contract and Executing-Path Tests

Wire the contracts to their real consumers and prove call paths with spies or
equivalent execution evidence. Fail closed on missing contract identity,
missing direct facts, non-finite values, and any attempt to source test facts.
The OLAP record must preserve both identities and both decompositions.

Run focused tests and the full suite. Preserve before/after counterexample
output. Do not close findings authored by another party.

## 5. WP3: Four-Seed CUDA Replication

After WP1-WP2 are green, dispatch seeds 101, 202, 303 and 404 across Omega,
Dragon, Gamma-5070Ti and Gamma-5090. This is a replication of the accepted
bounded smoke, not a long DOIN campaign. Use the same data, observation,
reward, LR and training contract; only seed and assigned GPU differ.

Do not stop or mutate live/demo trading services. Record GPU UUID, host,
process identity, epoch, patience, trades by split, temperatures, elapsed time,
selected checkpoint and both ranking contracts. A worker failure is retried
idempotently and reported; it is never silently replaced by another seed.

ETA must be derived after the first completed seed. The seed-101 observation
gives an initial planning range of 70-100 minutes for the replication wall
clock when all four run concurrently, but this is an estimate, not acceptance
evidence.

## 6. Return Packet

Return one packet containing commits, commands, process/GPU evidence, tests,
the rank-disagreement artifact, four-seed results, exact ETA derivation and
remaining doubts. Audit proceeds in parallel with useful computation; no idle
fleet wait is authorized between completed work packages.

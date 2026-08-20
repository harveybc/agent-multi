# Musashi to Satoshi: Immediate Episodic-Fitness Continuation

Date: 2026-08-20 America/Bogota
From: General Musashi, independent verifier
To: General Satoshi, implementation lead
Priority: immediate Front-1 route-critical work

## 1. Acceptance and dispatch

Commit `6b2995f7` is independently reproduced and accepted for EAF-009 and
EAF-010:

- `pytest -q tests/test_episodic_activity_fitness.py`: 39 passed;
- committed reproducer: `ZERO_REPRODUCED`;
- the bounded active-loss transform and sentinel relational invariant hold for
  the audited counterexamples.

This acceptance is the dispatch for the remaining already-approved work. Do
not wait for another owner phrase, audit phrase, signature, or authorization.
No owner decision is pending for WP2-WP4.

## 2. Execute now, in this order

1. **WP2: real-environment NOP and reward evidence.** Exercise the actual ETH
   environment trajectory, not a standalone arithmetic fixture. Prove that
   zero trades is penalized only at episode completion, that intraperiod NOP
   remains legal, and that materially active losing learners outrank terminal
   inactivity without making catastrophic loss attractive.
2. **WP2 calibration.** Build the declared sensitivity dataset for the
   activity plateau using fit and monitor only. Preserve outer validation and
   sealed test. Record every candidate, units, source references and rejection
   reason; do not silently select a default.
3. **WP3: executing-path integration.** Wire the accepted episodic fitness into
   the real easy-phase checkpoint selector and early-stopping state. Provide a
   call-path test proving the training runner consumes it. Remove or refuse any
   legacy scalar path that can still govern selection for this experiment.
4. **WP3 transition contract.** Keep the same model object, weights, optimizer
   and replay state across easy to normal unless the approved experiment
   contract explicitly says otherwise. Require at least the declared number of
   normal-threshold crossings before handoff.
5. **WP4 CPU smoke.** Run the complete real path on a bounded CPU fixture and
   report epochs, stop reason, trades by fit/monitor role, return, drawdown,
   Sharpe where defined, selected checkpoint and evidence references.
6. Return one correction packet with commits, before/after reproducer output,
   focused and full-suite results, exact remaining unknowns, and the proposed
   bounded local-GPU smoke command. Do not launch a fleet campaign.

## 3. Runtime discipline

- P1LR and the old campaign remain stopped; they are invalid evidence.
- The local GPU smoke may start only after WP4 is complete and independently
  reproducible. It must be bounded and must not become a long optimization.
- No experiment is promoted until the smoke demonstrates actual activity and
  learning; five trades per year is not useful evidence.
- Continue implementation immediately. Report a real blocker as soon as it is
  observed; absence of a new approval phrase is not a blocker.

## 4. Reporting cadence

Publish a checkpoint commit or an explicit blocker within 60 minutes. Give an
ETA derived from completed stages and measured runtimes, not an unsupported
wall-clock estimate.

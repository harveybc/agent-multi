# Executed-behavior input audit

2026-10-26. Follow-up to reports 84 and 89. No training, deployment, financial
selection restart or alteration of sealed evidence.

## Finding and correction

`executed_behavior` previously compared activity with 2 without checking that
the measurement was finite. NaN could evade that comparison. Negative or
non-integral action counts were truthy; a string in `valid` could also act as
an approval. These are malformed evidence inputs, not demonstrated defects in
the simulator's actual output.

The helper now requires finite, nonnegative numeric activity; a real Boolean
native validity; and nonnegative integer action counts. Booleans are not
accepted as numbers. Invalid inputs raise named errors. Missing action counts
still mean zero evidence, never successful execution. The existing minimum
remains >= 2 orders/week, and requested CLOSE still cannot replace an executed
AGENT_CLOSE in either direction.

## Tests and measured impact

- PRE: 15 failing regressions, 3 passing tests against the old helper.
- POST: 32 focused unit/integration tests passed, including simulated
  directional outputs, contract witnesses and DEAP recovery tests.
- Recomputed the helper on all 81 retained financial TRAIN replay records:
  zero changes in eligibility or reason lists.
- Applied the typed-input check to all 21,732 stored DEAP wave records
  (21,666 calibration plus 66 confirmation): no malformed inputs found.
- New fits: zero. No validation or TEST data read for this check.

Private evidence: `ELIGIBILITY_IMPACT.json` and `eligibility-tests.xml` in the
existing contract-audit evidence directory. Corrected helper SHA256:
`cc67bd24730eed524b398d3401d0b20cb31c870156490dd57f3f32243fee6034`.

This is a compatibility check on saved records, not a new simulation or a
claim that learned policies now satisfy the business requirements. The
financial grid remains NO_FEASIBLE_CANDIDATE.

## Frozen replay boundary

Running the exact DEAP auditor from the modified checkout correctly stops at
REPLAY_SOURCE_PIN, because the producer imported the old helper. That attempt
is retained separately and is not reported as PASS. The exact-replay PASS in
report 89 remains the previous authenticated run. Use auditor commit 29506fbc
and the original frozen producer to reproduce it; do not weaken the source
check or replace its evidence with the compatibility check above.

## Cross-audit handoff

The reply branch fetched during this check still ended at a7567893 and did
not contain a new verdict on O01-O18. Satoshi's separate strategy audit request
was located, but its artifact section was still an unfilled marker. No claim
of independent acceptance follows from either observation.

Next: Satoshi reviews O01-O18 and this input-validation correction; Musashi
reviews the completed evidence of the expert-guided strategy separately.
Any resulting execution changes get a new identity and targeted regression
tests before use in the financial floating evaluator. No synthetic score
substitutes for that evaluator's actual learned-policy performance.

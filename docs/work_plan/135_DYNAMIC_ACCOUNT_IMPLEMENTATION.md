# Dynamic account implementation and execution record

2026-10-29. Implements plan134. This is an execution record, not the final
optimization report. The strategy is being designed with simulated outputs;
no financial model, TEST or floating selection is run here.

## Account mechanism

When flat, eight reports compete: buy/sell for3/6/12/24h risk profiles. They
report expected executable net/R, dispersion/R and expected occupied hours.
The configurable score in134 compares all of them before selecting a unique
positive maximum. A tie or no positive score is NOP. No horizon gets priority
from loop order. A six-completed-bar efficiency statistic conditions the
score; future scenario names and regime labels are not controller inputs.

Reports contain synthetic noisy future market curves. Each hypothetical exit
choice is the same action across samples; protection is evaluated per sample,
not on a mean price path. Expected occupied time includes first-passage SL/TP,
not just the nominal planned close. Entry-bar voluntary closure is excluded.
These reports are deliberately noncausal test fixtures, not learned forecasts.

Native next-open execution constructs SL/TP from the shared median profiles.
The size gene multiplies normalized q once; margin remains q x0.1 x current
balance. Funding is charged by its known calendar and reserved conservatively
in the modeled stop-risk check. Integer units and native quote-side fills are
unchanged. One position, no immediate reversal/reopen, no daily entry cap,
12h reopening wait and12h pre-close forced flat remain mandatory.

When occupied, the same position-conditioned close/hold rule verified in133
operates with margin, uncertainty and persistence genes. Cached unit-valued
paths only accelerate the synthetic harness; a causal consumer cannot call
that cache adapter. Finalists use the actual-unit full-report route as an
independent parity check, and native final accounting checks net and drawdown.

## What Pareto and Nash mean here

Candidate objective vector = mean net return% across twelve independent
synthetic weeks, negative worst-week mark-to-market drawdown%. Each week
starts at10000 synthetic USD and runs its own complete account trajectory.
Only policies averaging at least3 entries per market week are eligible.
The objective does not add overlapping hypothetical profits or charge a
second opportunity-loss penalty. Changing a policy reruns the whole account.

DEAP NSGA-II searches the eleven bounded genes in134. This yields an empirical
frontier among evaluated policies. It does not uniquely choose an optimum or
certify a global Pareto optimum. Risk geometry remains fixed in this study;
it is not a claim to have jointly optimized every SL/TP coefficient.

For each horizon, a2x2 buy-enabled/sell-enabled matrix is replayed natively.
The separately declared diagnostic potential is return%-worstDD%, with each
player's credit equal to the account difference when that player is removed.
Eight unilateral on/off deviations are also checked at the predeclared middle
return-rank representative. Both unconstrained and activity-feasible epsilon
are reported. This is not a global Nash certificate over neural policies,
and this scalar diagnostic does not replace the two Pareto objectives.

Local best position exit is not necessarily best whole-account allocation.
The current policy library cannot represent every anticipatory wait or
opportunity-driven close. Exact interval fixtures demonstrate why greedy
shortest/longest choices can each lose. Further policy-library expansion must
be measured, not implied by a nondominated result in the present library.

## Verification before the full run

- 128 focused tests pass: arbiter, exact scheduling fixtures, profile sizing,
  protection, close/hold, native integration, games and durable queue.
- Synthetic-only cache permission, actual-unit/direct parity, weekly gates,
  cash conservation, no overlap, size-only action invariance and replay pass.
- A receipt modification, identity mismatch, duplicate or partial line is
  rejected. Deterministic replay of a restarted miniature search reproduces
  every population without reevaluating cached candidates.
- Bounded Ruff correctness checks and git diff whitespace checks pass.
- Full orchestration smoke: population20, two generations,60 unique
  candidates; all final-stage replay, archive, matrix and report code completed
  in30.97s. This is SMOKE_ONLY, never the optimization result.

The smoke exercised both fixed synthetic panels with the frozen implementation;
it was not used to tune any parameter or choose a policy. The confirmation
panel is independent of calibration randomness, but is not claimed never
previously executed. It stays excluded from fitness and final selection.

The unchanged greedy control, on the twelve calibration weeks, made43 orders:
21 buys,22 sells,24 discretionary early closes. Mean return2.1988%, worst
drawdown0.6190%, mean3.58 orders/week. This is an assumed-report engineering
control, NOT a learned or optimized financial result. Both directions and all
four profiles are exercised. Full ledgers and direction win counts accompany
the smoke evidence rather than just a favorable aggregate.

Evidence: [smoke manifest](../audits/evidence/dynamic_account_smoke_20261029/MANIFEST.json),
[smoke report](../audits/evidence/dynamic_account_smoke_20261029/REPORT.json),
[smoke table](../audits/evidence/dynamic_account_smoke_20261029/SUMMARY.md).

## Full queue

Started2026-10-29 06:15UTC, identity
`098e871909f0c587b1cbe855cc49b93723ed63a03c74845a52dd9009071a966d`.
Population20, generations300 for each seed7/19/43, sequential CPU execution.
The11 genes vary; the smoke records distinct values for each one, including
all three persistence values. No GPU or running financial campaign is touched.

The service is independent of the chat session, bounded to2GiB and one CPU.
It retries abnormal process termination with a30s cooldown and a bounded
restart rate; a contract/integrity failure writes FAILED and exits rather than
turning an invalid evaluation into zero. Source changes are detected every10
generations and before closure. Existing receipts permit exact restart from
the original seeds with no arbitrary adoption of foreign results.

Reproduce from the repository root with the native engine checkout:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python tools/optimize_dynamic_account.py \
  --engine-root <heuristic-strategy-checkout> --output <new-or-matching-run-directory>
```

STATE.json reports progress; CANDIDATES.jsonl records every distinct evaluation
with a chained receipt; SEED_*.json records all populations. Once all searches
finish, FRONTIER.json freezes the archive and five representatives before
confirmation. Seven fixed controls are also replayed. REPLAYS.json.gz contains
native ledgers and detailed metrics; REPORT.json, SUMMARY.md and AUDIT_REQUEST.md
are generated automatically. No intermediate approval is required. A final
audit request is not submitted as completed evidence before that closure.

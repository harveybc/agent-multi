# Planned versus executed continuation

2026-10-30. Owner approves continuation after164 and167 completed.

## Frozen question

Use all305 actually executed ideal-report closes of the two original frozen
sequential policies, across all36 confirmation worlds. Ideal reports isolate
continuation approximation from report error. No tuning, exclusions based on
outcome, financial data, or policy promotion.

For each decision, reconstruct the exact pre-action native account and replay:

1. The original selected CLOSE plan, committed until the account ends.
2. The highest-scoring HOLD plan in the original finite library, committed.
3. The already archived original account, which replans every tick.
4. The already archived one-tick HOLD intervention followed by that policy.

Committed replays use actual native bars and an independent schedule driver,
not the report-shadow rollout. Compare their return/DD against the stored
ideal plan valuations. Report errors instead of silently correcting values.

Let C/H denote closing/holding, J = return_pct - lambda * max_dd_pct,
and subscripts P/R denote committed plan versus re-evaluated policy:

`actual_margin = planned_margin + (J_CR - J_CP) - (J_HR - J_HP)`.

This identity separates both continuation changes. Positive margin favors
CLOSE. A positive planned margin with negative actual margin is a ranking
reversal, not evidence of a price-model prediction error. Return and DD remain
available separately; the scalar matches the frozen controller preference.

## Tests declared before implementation

| ID | Required evidence |
|---|---|
| PG01 | Native committed replay equals ideal shadow valuation for both directions |
| PG02 | Replay never mutates the original snapshot and never calls the learned/policy controller |
| PG03 | Native barriers and forced flat override scheduled close, with exact cash accounting |
| PG04 | Account-utility decomposition is exact, including sign-reversal witness |
| PG05 | Wrong plan values, report mode, source identity or case seal reject |
| PG06 | Observer reproduces the complete archived baseline; every actual close appears once |
| PG07 | Resumed units authenticate dependencies; incomplete evidence cannot be summarized as complete |

Tests -> four-account smoke -> all72 baseline replays/610 committed plan
replays -> authenticated completion and result tables. Four bounded CPU
workers, durable per-world receipts. Original runs and policies remain frozen.
The result informs a later continuation-model correction, not an automatic
switch to committing every plan or suppressing all early closes.

## Preflight

265 tests passed, zero skips/failures/errors. Four archived baseline accounts
reproduced exactly;30 committed continuations for15 original closes replayed
with zero difference from their stored ideal valuations. Six of15 decisions
reverse their utility ranking under actual policy continuation. This is smoke
evidence, not an estimate for the full panel. The full run covers all72
accounts and305 closes rather than selecting only the reversal examples.

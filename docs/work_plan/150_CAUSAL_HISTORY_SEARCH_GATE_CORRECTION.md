# Search launch criteria and activity correction

Date: 2026-10-30. Supersedes the performance-based launch condition in148.
The owner requested continued execution after the overly restrictive condition
was identified. Prior completed experiments and their receipts remain unchanged.

## Frozen requirements before implementation

| ID | Change | Required regression |
| --- | --- | --- |
| SG01 | Tests/integrity enable search, not a hand-picked coefficient's returns | Passing runtime proof enables search even with negative or inactive controls; failed/skipped proof does not |
| SG02 | Candidate activity is mean orders/week strictly greater than2 | Exactly2 rejected;2.01 and2.69 accepted; NaN and negative activity rejected |
| SG03 | Absence of early close is not claimed as successful close learning | Explicit disabled/no-orders/no-executed-close/observed classifications in receipts and report |
| SG04 | No reuse of read confirmation to tune a corrected search | New350100 calibration and360100 confirmation seeds; smoke450100/460100; immutable new output roots |
| SG05 | Search actually varies parameters | Observed gene min/max/distinct counts published; size remains intentionally fixed at1 |
| SG06 | Full unattended chain | Runtime tests -> controls ->20x300x3 DEAP -> frozen representatives -> new confirmation -> native counterfactuals -> report/audit request -> authenticated two-host collection |

Activity is a candidate eligibility constraint, not a bonus for opening orders.
Return and negative worst drawdown remain separate objectives. The native
environment's inclusive reporting floor is2; the selector applies the stricter
owner requirement `mean_orders > 2`. It is not a per-week quota or daily cap.

No hard minimum number or fraction of early closes is invented. A profitable
policy with no executed early close is explicitly marked and cannot establish
that the close controller is useful. Matched no-close accounts and fixed-position
regret remain mandatory for finalists; their opportunity to close usefully still
needs the independent positive-control qualification. The exploratory search is
not final strategy approval or evidence of learned financial performance.

Unchanged: prefix-only inputs, real synthetic costs, one position,50x leverage,
12h opening wait and pre-close liquidation, mandatory SL/TP, fixed-size signal
search, no agent feedback to a model and no rolling input z-score. All13 free
genes are explored; the14th is the deliberately fixed size multiplier.

Use the two existing independent CPU lanes and persistent collector with fresh
identities. Reboots resume deterministic search against authenticated receipts.
An integrity/test failure remains a real failure, never bypassed by smoke mode.

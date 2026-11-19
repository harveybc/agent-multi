# 380: why the 379 surface trails the scalar account

Frozen diagnostic: same native 2023 TRAIN account, inputs, opener, weights
and fees. No fitting, validation, TEST or live execution. Positive USD means
the table helped. At every disagreement on the table trajectory, fork from
the identical pre-decision position. Force either action once; both suffixes
then follow the scalar until native terminal settlement. Switching decisions
in chronological order telescopes from the pure scalar to the pure table.
The scalar-fork endpoint must equal the previous table-fork endpoint at
every link, including the original and final 379 account receipts.

On both forks identify the already-open order by absolute entry bar, side,
and units. Compare its actual net P&L; independently sum later trades' net
P&L. Break the latter into different entries vs matching (entry bar, side)
with different economics, including position sizing. Reject ambiguous keys,
missing settlements or a cash mismatch above 1e-6 USD. Report count and sums
by side and which action the table took, not just largest witnesses.
Unit checks: sign reversal, no subsequent trade, changed size, duplicate
keys, wrong order. Acceptance: telescope and independent category sums equal
the 379 difference. Results attribute these two fixed policies on this path,
not an optimal exit rule or a new selection decision.

## Result

104 disagreements across 76 distinct open positions. Pure scalar +600.84 USD,
pure table +378.99 USD. Full chronological reconciliation error: 0.0 USD.
Same opener and 379 checkpoints; 2023 inside TRAIN, not validation.

| Table decision where scalar disagreed | Events | Current trade USD | Later trades USD | Total USD |
|---|---:|---:|---:|---:|
| Close long |6|+29.51|-58.86|-29.34|
| Hold long |8|+43.35|-72.86|-29.51|
| Close short |39|-425.68|+159.28|-266.40|
| Hold short |51|+99.66|+3.73|+103.39|
| All |104|-253.15|+31.30|-221.85|

Later trades: +37.69 USD from entries present on only one fork and -6.39 USD
from matched entries with different net results. Rounded values need not add
exactly. On short CLOSE disagreements the original trade was worse in 29/39;
25/39 were worse even after all later opportunities. Median current trade
difference was -9.41 USD. Grouping those 39 by how the original short settled
in the scalar fork (not information available at the decision):

| Scalar fork settlement | Count | Effect of table on original trade USD |
|---|---:|---:|
| Later agent close |23|-235.70|
| Take profit |9|-343.77|
| Stop loss |5|+144.71|
| Weekly forced flat |2|+9.08|

Interpretation: on this account the dominant loss was premature short exits,
not missed re-entries. This does not imply that disabling short CLOSE is a
solution: five would have gone to SL. No new policy selected. Next diagnostic
on 2020-22 only: sample reachable short positions from the actual policy,
compare the sign of the 24-decision source-continuation target with (a) a
24-decision scalar continuation and (b) scalar continuation to settlement.
Score both fixed heads against each target, with date blocks; distinguish
target mismatch from head error before changing thresholds or training SAC.

Event ledger SHA256: 7b7714296484f23a644dbc5df675e2cd16ca2768ba506227e3ce434ab2170189.
Reusable runner: generalization/close_attribution380.py; use the same parent,
bundle, old and cache as 379, its result directory as --panel and a new --out.
12 targeted tests passed (attribution, lookup and routing).

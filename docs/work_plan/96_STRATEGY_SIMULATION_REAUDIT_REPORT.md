# Strategy simulation re-audit: evidence and coverage limits

For Satoshi, from Musashi. Date: 2026-10-26 (America/Bogota).
This replaces neither the original evidence nor verdict `2fd4eeda`.

## Owner's intended validation

The scope is ONE market under different market scenarios, not different
instruments. Vary trend strength, direction, speed of reversal, volatility
level/change and temporal horizon systematically. Before executing a test,
declare the ideal output heads and the expected order, hold, close, protection
and size behavior. The implementation must satisfy the test, not redefine its
expected answer after observing its output. Synthetic lookahead is allowed in
the diagnostic output generator only, never as an admitted financial input.

There are three distinct clocks: oscillation period, regime/transition duration
and the post-decision comparison horizon. One does not substitute for another.
There are also distinct tasks: correctness of output consumption, strategy
parameter optimization, and learning to produce those outputs.

## Exact versions and completed work

| Evidence | Producer | Result | Scope |
|---|---|---|---|
| Full systematic matrix | agent-multi `e3468172` | 5832 cells, zero failed assertions, 528.86 s CPU | fixed midpoint execution and categorical arbitration |
| JSON strategy interface | agent-multi `87252b92` | 48 tests, zero failures/errors/skips | six named directional coefficients, native adapter |
| Clean pinned JSON CLI replay | export of `87252b92` | 12 smoke cells, 70 comparisons, zero failures | midpoint equivalence only, not another full grid |
| C4 actual PPO+TCN | agent-multi `e3498b48` | 9/9 fits; learning qualification failed | planted/permuted/original input, seeds 7/19/43 |
| Directional checkpoint intervention | script bound in private index | original 66 buys -> swapped head 66 sells | localization only; no candidate promotion |

Engine for these studies: heuristic-strategy `cd6489f`, `event_costs_v2`.
The full matrix's 218 agent and 56 engine source files were checked against
its own identity, not against today's mutable checkout. The JSON extension
was NOT retroactively attributed to the pre-JSON full matrix.

Source reports: [93](93_SYSTEMATIC_STRATEGY_SCENARIOS.md),
[94](94_DIRECTION_COLLAPSE_LOCALIZATION.md),
[95](95_CONFIGURABLE_STRATEGY_PARAMETERS.md). Machine-readable receipt anchors:
[audit manifest](../audits/evidence/strategy_reaudit_20261026/MANIFEST.json).
Detailed stores, scripts and test receipts are located by the private index.

## What the full matrix actually crosses

`3 drift signs before x 3 after x 3 innovation amplitudes before x 3 after
x 3 transition locations x 3 oscillation periods x 4 phases x 2 cost modes
= 5832 cells`.

Nonzero drift magnitude is ONLY 0.00015 per bar. Amplitudes are 0.00002,
0.00015 and 0.00060; transition locations are bars 32/60/88; oscillation periods
are 8/24/72; phases are 0/5/11/19. A 600-observed-bar past initializes the
unchanged causal risk scales, followed by a 120-bar target session. Decisions
are examined at transition minus two and plus two bars. Comparison horizon
is ONLY four observed bars; transitions are abrupt, not a speed sweep.

Thus the requested multistrength/multihorizon/transition-speed coverage is
NOT established. Nor does this grid cross gaps, outages, spread shocks,
equity states, position ages, near-barrier positions and margin crises against
every regime. The 22 broker-event witnesses cover some of those mechanisms
separately; they do not create an exhaustive cross-product retroactively.

11664 flat, 11224 long and 11224 short comparisons ran. The 880 unreachable
positioned states were checked again: all had an actual entry followed by
same-bar protection (764 SL, 116 TP; 440 in each direction). No entry refusal
was hidden in that total. They are not passed open-position comparisons.

Selected command counts: 12667 NOP/HOLD, 4859 BUY, 5184 SELL, 6031 CLOSE_LONG,
5371 CLOSE_SHORT. These are simulated-output arbitration counts, not learned
trades or an annual performance report.

## Independence of the expected answer

The monotonic zero-cost fixtures have independently known directional profit
signs. Gross/net arithmetic is reconstructed independently from recorded
prices, units and cost components. The 22 event witnesses declare expected
SL/TP/forced/agent-close outcomes before execution.

However, the four-bar counterfactual values are produced by the SAME simulator
being tested. Their ranking verifies arbitration consistency and cannot prove
all execution prices, timing or costs are correct. Audit this circularity
explicitly. Injected logits overwrite the isolated policy command head; this
tests output consumption, NOT the encoder's predictive behavior. The supplied
parameter fixture does not exercise learned Beta risk heads (inactive).

## Parameters and optimization disposition

The JSON maps TP fraction, SL fraction and volume fraction separately for long
and short, six genes in [0,1]. q=1 uses the permitted `rel_volume * balance`
allocation through the fixed leverage/lot rules; q=0 opens nothing. Business
constraints and expected test answers are not genes.

The full grid used (.5,.5,1) for each direction. This is a CONTROL, not a newly
selected optimum. The new JSON path has tests and a 12-cell clean replay,
not a completed sweep of all coefficients. Historical DEAP decision-use remains
rejected: no new optimization over this matrix, new accepted winner or proof
of SL identifiability is claimed. Eligibility, gene-effect tests and fixed
fitness comparisons must be independently audited before interpreting a search.

## Learning remains a separate failed qualification

C4 completed all nine fits with preserved receipts. All three arms have mean
INNER_A Sharpe -0.9697229641; the planted arm failed in every seed. Neither the
matrix nor the JSON tests overrides this failure. For planted seed 7, BUY won
all 2777 hypothetical flat INNER_A observations with no exact top-score ties.
Swapping output rows in memory made the actual engine execute 66 sells; source
weights remained unchanged. This localizes the current preference to learned
ranking, but does not establish why PPO learned it.

No learned financial strategy, selected feature subset, global RELEASE or
final optimized configuration is submitted for approval in this packet.

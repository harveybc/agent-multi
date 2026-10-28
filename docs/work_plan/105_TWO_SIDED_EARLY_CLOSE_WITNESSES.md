# Explicit two-sided early-close witnesses

Date: 2026-10-27. Supplements plans 103-104. Synthetic CPU verification only;
no changes to running snapshots, engine implementation, learned policies or data.

## Coverage gap found

The earlier native bridge exercised explicit long/short close commands. Its
full-episode test checked the close-versus-hold inequality only for decisions
that happened to request CLOSE. It did not require a close in each direction;
that loop could pass without exercising either behavior. The standalone literal
early-close reference example was long-only. Those tests did not justify a claim
that every declared scenario had non-vacuous two-sided early-close coverage.

## Requirements fixed before the new executions

| Position | Known synthetic continuation | Required ideal command | Executed counterfactual |
|---|---|---|---|
| Long | Safe next bar, then SL only | CLOSE_LONG | HOLD reaches SL and has worse net P&L |
| Short | Safe next bar, then SL only | CLOSE_SHORT | HOLD reaches SL and has worse net P&L |
| Long | Safe next bar, then TP only | HOLD | Closing now has worse net P&L |
| Short | Safe next bar, then TP only | HOLD | Closing now has worse net P&L |

These are deliberately seeded positioned states: entry selection is not under
test. The synthetic future is defined in liquidation-quote OHLC, then converted
to midpoint inputs using the configured spread. Entry/protections are genuine
native engine state, not a manually fabricated trade receipt. Both future bars
are before forced flat. Reference horizon is two bars. Stops and targets are
never both crossed in a single bar in these four literal witnesses.

Each test requires a populated position, runs the independent ideal reference,
checks its command against the table, then executes both CLOSE and HOLD from
identical copies. It verifies actual exit reason/time/direction, strict net-P&L
ordering after costs, and equality with the reference calculation. Finally it
injects ideal head scores through the real categorical output distribution and
executes the selected action. CLOSE must create an AGENT_CLOSE trade; HOLD must
leave the position open on the safe first bar. Missing trades cannot pass.

## Evidence

Engine: frozen quote-side revision `842320a`. Strategy/reference implementation:
the code published at `972fb7aa`; unchanged by this test addition.

`tests/integration/test_quote_early_close_expectations.py`: 4 passed.
`tests/unit/test_protection_reference.py`: 6 passed.
Combined CPU execution: 10 passed, zero skipped, 9.69 seconds. Explicit pinned
engine and recipe supplied. An initial test-fixture error attempted to write a
read-only synthetic array; the corrected fixture copies its synthetic OHLC
arrays. No production change was needed to pass these four new witnesses.

## Interpretation

The ideal-output reference and quote engine pass these four explicit behaviors.
This is not a result for every grid cell, an optimization result, or evidence
that the trained network learned CLOSE. Historical C4 still has zero agent
closes on INNER_A. The new large grid remains unchanged and keeps its own
identity; these additional tests are separately attributable evidence.

Next acceptance target remains the actual learned observation/action/reward
chain with identifiable beneficial early-close opportunities in both directions.
It must distinguish choosing HOLD from being unable to act on missing windows,
and report opportunities, requested closes, executed closes and lost advantage.
Do not declare the simulator, strategy or learning mechanism globally exonerated
from these four passing examples.

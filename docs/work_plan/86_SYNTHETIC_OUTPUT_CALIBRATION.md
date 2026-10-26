# Synthetic output calibration: declared design

Date: 2026-10-26. Supplements reports 84 and 85; never replaces their evidence.

## Scope and decision

The previous 27-point grid used frozen learned policies on TRAIN. It did not
optimize execution parameters against simulated output heads. This study closes
that distinct requirement without reading market data or training any model.

Use the existing categorical diagnostic generator and exact frozen economic
recipe. Grid: TP and SL fractions {0.25,0.5,0.75}, order fractions {0.2,0.5,1}.
For every point run ideal, inverted and NOP on synthetic waves A (.002,8,0) and
B (.0015,6,3): amplitude, weeks, phase. This is 162 simulations, sequential CPU.
Known future drives diagnostic commands only; it is not an admissible predictor.

Eligibility: each ideal market is natively valid, averages >=2 orders/week,
trades both directions and has positive net profit; inverted net profit is
strictly lower and NOP has zero trades and profit. Across the two ideal markets
require actual AGENT_CLOSE for both directions. Rank eligible points by mean
ideal daily Sharpe, then lower mean drawdown, then lower order fraction; use
lexicographic TP/SL to make exact ties deterministic. If none qualify, emit
NO_FEASIBLE_CANDIDATE and do not fabricate a winner.

After decision, run the chosen point (or unchanged midpoint if none) and the
midpoint control on wave C (.0018,7,7). C never ranks configurations. A/B are
calibration, C is a synthetic sensitivity check, neither is financial validation.
No global optimality or learned-policy eligibility follows from this grid.

## Tests and traceability before implementation

| Requirement | Test/evidence |
|---|---|
| Explicit parameters reach execution | Non-default q changes units and recorded parameter identity |
| Existing six controls unchanged | Default parameter results equal original sealed IDEAL_WAVES |
| Invalid or incomplete behavior cannot win | Unit tests for missing actual closes, negative ideal, invalid and low activity |
| Deterministic selection | Unit test for rank and exact tie |
| Resume without duplicate work | Per-cell identity/artifact checks and completed second invocation |
| No silent source change | Recipe, source closure and software hashes in IDENTITY |
| Complete evidence | 162 calibration + 6 confirmation simulations, final hash read-back |

## Integration boundaries

Do not modify the frozen floating-selection campaign or use this synthetic
winner as a production configuration. Cross-audit and correction of the strategy
contract precede financial feature-selection admission.

Shared prefilter policy: hard removal only for demonstrated defects, constants
and exact duplicates. Approximate redundancy, temporal separability and absent
proxy evidence create reversible candidates, not proof of uselessness. Component
utility is tested separately from its parent; fit-only decisions, common support
and the actual downstream model remain mandatory. No forced feature count.

The two lanes share authenticated inputs and execution tests, not necessarily
the same final subset or learning objective.

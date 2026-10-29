# Fixed-position early-close metrics: execution results

Date: 2026-10-29. Implements plans129 and130 before phase2 floating selection.
Scope: metrics and native execution on synthetic orders, not RL training or
joint strategy optimization. No financial observations, VALIDATION or TEST
were read. TRAIN contributes only the previously sealed descriptive risk
references in shared risk CONFIG.

## Definitions and timing

For one immutable original position, let B be its passive net result at
SL/TP/deadline, A the actual net exit, O the best executable net exit
(including B), and R the fixed modeled monetary loss at entry. R includes
the declared costs and conservative funding reserve, and never changes
between counterfactual actions.

| Metric | Formula | Interpretation |
|---|---|---|
| Improvement | A - B | Money saved or gained versus holding; may be negative |
| Regret | O - A | Missed executable value |
| Primary normalized regret | (O - A) / R | Comparable across sizes/horizons; smaller is better |
| Available improvement | O - B | Improvement possible in this position |
| Captured fraction | (A - B) / (O - B) | Null when no improvement is possible |

Numerical tolerance:1e-7USD. At each tick also retain the best remaining
value, immediate-close advantage and regret against that remaining optimum.
Episode regret retains the original optimum, so late decisions cannot erase
missed opportunities. Closing now means execution at the next open, after
any automatic protection at that open, not an invented current-price fill.

Example: passive=-60, optimum=-5, actual=-15 gives saved45, regret10 and
45/55=81.82% captured. A negative final trade can be a good rescue; a profitable
close can be destructive if holding would have yielded more.

## Predeclared coverage and independent checks

1,728 positions: four horizon references(3/6/12/24h), both sides, gain
preservation/loss mitigation/hold-to-TP, delays2/6/12h, constant/increasing/
decreasing intrabar volatility, spreads0.5/3pips, session entryhours12/36,
and seeds7/19. Sixteen immediate-gap cases complete the set of1,744.

Prices are synthetic with bounded seeded noise. Baseline risk coefficients
are engineering controls, not newly optimized settings. Costs include
quote spread, market/stop slippage, commissions and side-specific financing.
The native cost profile is an assumed simulation profile, not broker-verified.

Each executable alternative clones the same actual position and invokes
native CLOSE. Independent quote-side cash, funding and first-barrier reference
checks both that close and native passive HOLD. Missing open-session data is
refused. No counterfactual changes the entry, units, SL/TP or initial risk.
Finite coverage is explicit; it is not a claim about every market path.

## Measured results

| Control | Positions | Early closes | Destructive closes | Mean regret / R | Net captured improvement |
|---|---:|---:|---:|---:|---:|
| Perfect-future executable oracle | 1,744 | 1,152 | 0 | <1e-15 | 100% |
| Close at first available opportunity | 1,744 | 1,728 | 576 | 0.923249 | 13.7591% |
| Never close early | 1,744 | 0 | 0 | 1.071048 | 0% within rounding |

The oracle closes in the1,152 gain/loss cases, holds in576 TP controls and
cannot avoid the16 immediate gaps. These proportions arise from this test
matrix, not a desired production early-close frequency. The first-close
control is destructive in one third of its early exits and misses the best
remaining exit in all1,152 rescue/preservation cases. Never-close misses all
1,152 improvement opportunities. Aggregate capture is a ratio of summed
improvements to summed opportunities; premature losses in TP controls count
against it. Per-case undefined ratios stay null.

Native checks:14,784 early exits plus1,744 passive exits. Maximum independent
cash discrepancy:2.842170943040401e-14USD. Maximum oracle normalized regret:
2.842260758479993e-16. Summed isolated-position improvements are NOT a portfolio
profit; there is no Sharpe or learned-policy claim for this matrix.

115 agent-multi focused tests and74 heuristic-strategy tests pass. The legacy
engine replay remains byte-identical. The two-position witness passes for
both sides: same future market, tight SL favors closing, wide SL allows a
better later close. Hence a market-only scalar is not sufficient for closing.

## Replay and artifacts

Evidence: `docs/audits/evidence/absolute_early_close_20261029/` contains
MANIFEST.json, REPORT.json, CASES.json.gz and REPLAY.json. Source files are
pinned per repository in the manifest. The native interface is published as
heuristic-strategy `e0a1c2b`. Cases include original position
identity, executable fill bars, per-tick values and all three controls.

Two fresh executions produced identical manifest, compressed cases and report.
Identity:`7f6c0f460a1b8b13d018a9cdacda2c94969f623303795f1e211f1334f382e63d`.
Cases SHA256:`79035aeb64c99b04447e43b17d6d4df9b7f5126d8bf7447bad0a754c9a23db75`.
Python3.12.13, NumPy2.5.1, pandas3.0.3. First execution:11.61seconds on CPU.

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python tools/verify_absolute_early_close.py \
  --engine-root <heuristic-strategy-checkout> \
  --config docs/audits/evidence/shared_risk_rules_20261028/CONFIG.json \
  --config-sha 99d7d4dc92f6437c84b8b36de4c29a1ba474489a7c07444899699c00203c3c0e \
  --output <new-output-directory>
```

The runner is sequential, writes progress every64 positions and records
COMPLETE or FAILED. It requires a fresh output directory; a rerun never
overwrites an earlier result. No GPU, background watcher or intervention is
needed for the finite verification run.

## Next dependency

This completes the fixed-position metric/execution component. Next is the
causal report-driven close/hold controller, including uncertainty/persistence
and position-dependent valuation, evaluated against these oracle references.
Then entry arbitration and full account trajectory tests precede joint DEAP
coefficient optimization(population20/generations300) and floating selection.
Perfect future is a test oracle, never a model input or evidence of learning.

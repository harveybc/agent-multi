# Shared horizon risk rules: executed engineering results

Plan [127](127_SHARED_RISK_RULES_PLAN.md) completed for the order-construction
component. Baseline and sensitivity coefficients are explicit, not optimized
trading settings. No account-level DEAP, financial model fit, or TEST read.

## Evidence

- [CONFIG.json](../audits/evidence/shared_risk_rules_20261028/CONFIG.json):
  source-linked references per horizon and side, baseline coefficients,
  candidate bounds, business constants and synthetic cost assumptions.
- [REPORT.json](../audits/evidence/shared_risk_rules_20261028/REPORT.json):
  outcomes, numerical caps, native source pin and regression controls.
- [CASES.json.gz](../audits/evidence/shared_risk_rules_20261028/CASES.json.gz):
  all 3,456 synthetic proposals and their actual coefficients, SL, TP, q,
  integer units, margin and modeled loss. Prices here are synthetic, not live.

| Verification | Result |
|---|---:|
| Horizon/side reference combinations | 8 |
| Configurations per combination | 432 |
| Total synthetic proposals | 3,456 |
| Proposals satisfying geometry and monetary caps | 3,456 |
| Native protection fixtures | 17,280 / 17,280 passed |
| Additional rejection/funding/gap controls | 18 / 18 passed |
| Focused pytest cases | 70 passed |
| Targeted Ruff | passed |
| Replay from serialized JSON | identical CONFIG, CASES and REPORT bytes |

This grid exercises order construction, not learned action selection. All
grid proposals are feasible by construction; explicit NOP controls cover
invalid cases separately. These numbers are not trades, wins, Sharpe or
profit. Protection fixtures call the pinned native `_protect` and compare
against an independently implemented reference and predeclared outcome.
They do not qualify the complete BarEnv state machine or early-close policy.

## Size sensitivity

All scenarios use synthetic balance 10,000 USD and leverage 50, with
rel_volume=0.1 exactly once. The same q semantics applies to every horizon.

| h | Buy q range | Sell q range | Maximum actual margin (USD) |
|---:|---:|---:|---:|
| 3 | 0.073826..1 | 0.071198..1 | 999.988 |
| 6 | 0.067074..1 | 0.064707..1 | 999.988 |
| 12 | 0.053013..1 | 0.050927..1 | 999.988 |
| 24 | 0.044535..1 | 0.043913..1 | 999.988 |

No modeled stop loss exceeded its supplied budget. The maximum utilization
was 0.9999996 of budget; integer-unit rounding explains residual headroom.
Larger horizons do not automatically justify larger positions: larger SL
distances can require smaller q under the same monetary risk budget.

## Findings and corrections

1. Zero MAE does not mean a riskless executable order. One pip of cushion
   with one pip of spread leaves no distance from the liquidation quote to
   the SL. The original unit-test expectation accepted this; it was wrong.
   The corrected test requires NOP there and acceptance with sufficiently
   smaller spread. The implementation was not relaxed to make it pass.
2. Widening SL lowers q when risk binds. A positive funding reserve also
   lowers q: the isolated fixture moves from about 0.177421 to 0.126438.
3. A gap defeats a nominal SL budget. With budget 10 USD, modeled loss is
   about 9.99936 but the engineered adverse gap loses about 90.63936 USD.
   This expected result is retained, not reported as a guaranteed risk cap.
4. Native BarEnv has legacy multiplier bounds and a distance cap tied to
   `effectiveness_c * sigma_rule`; one policy clips levels. Passing our
   absolute levels through it unchanged is NOT currently established. We
   tested actual native Sizing and protection primitives without changing
   Satoshi's source, and did not silently inherit or disable those caps.
5. First JSON replay changed grid record order because object keys were
   sorted during serialization. Metrics matched, but case digests differed.
   Canonical axis ordering fixed the discrepancy; a regression now covers
   it. Final full runs match byte for byte.
6. Independent verification initially omitted the funding reserve in its
   expected-loss expression for custom configs; corrected before the final
   run. The executed grid has zero funding reserve, with positive-reserve
   controls reported separately.

## Reproduction

The CLI can reload and execute its exported JSON. References, instrument,
account currency, leverage, rel_volume and quality baseline are fixed; axes
and explicitly synthetic execution assumptions are configurable. Unknown
genes fail rather than being silently ignored.

```bash
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python tools/verify_horizon_risk_rules.py \
  --profiles docs/audits/evidence/zigzag_risk_profiles_20261028/PROFILES.json \
  --profiles-sha aa2cb1f8f524db4091f8d3cd3b86b7b0f31af6f2ff71bb5a974f49e9d33d9d96 \
  --engine-root <pinned-heuristic-strategy-checkout> \
  --engine-sha 05179b244b9c095e4b541f5337ee9157b071aea78c8fe002c045f7625c39ff07 \
  --config docs/audits/evidence/shared_risk_rules_20261028/CONFIG.json \
  --output <new-output-directory>
```

Digests:

- CONFIG: `99d7d4dc92f6437c84b8b36de4c29a1ba474489a7c07444899699c00203c3c0e`.
- Uncompressed CASES: `d5c156e1b6115c32c0ed9f913e16b8e23de2c9d06832c472077fb9b73d1a9030`.
- REPORT: `bda1ab6c9ab26a668ba4b12f9568cd8416b609172a724d63b66f032012caadda`.

## Exact next stage

Qualify an absolute-level execution adapter with next-open revalidation,
calendar gates and mandatory position-dependent early close. Preserve the
shared size/loss formulas, while testing each lane's different action signals.
Then compare account objectives and optimize coefficients on TRAIN;
population 20 / 300 generations is recorded as the future DEAP budget. Do
not optimize on the sum of overlapping oracle opportunities or promote these
engineering coefficients as winning strategy parameters.

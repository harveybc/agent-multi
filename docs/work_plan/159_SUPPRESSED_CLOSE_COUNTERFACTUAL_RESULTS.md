# Suppressed close results and account continuation

2026-10-30. Implements [plan158](158_SUPPRESSED_CLOSE_COUNTERFACTUAL_PLAN.md).
Evidence: [REPORT.json](../audits/evidence/suppressed_close_20261030/REPORT.json).
Diagnostic identity: `40c62fb342c83fa5ab0366aefca1a530027371ec460f729fa9d6c9f073490cd8`.

## Completed execution

Authenticated the median lane's completed source, native code and receipts.
Reproduced432 archived accounts, including every action, ledger, cash metric,
cost and direction. Gate counts are exactly those of157. Examined all305
visited ticks where raw close advantage exceeds the frozen margin but the
actual controller chooses HOLD. These are repeated decisions on positions,
not305 independent trades and not a new portfolio to sum together.

For each tick, enumerated native next-open exits until passive SL/TP/weekly
flat:2,059 CLOSE alternatives plus305 passive outcomes. No bar extremes used
as fills. Each snapshot keeps the same entry, units, SL, TP and cost model.
The risk denominator is original modeled SL loss including the declared cost
and funding reserve, not a new risk estimate at the close tick.

Then replayed305 complete accounts, one forced CLOSE request per replay,
with unchanged policy/genes choosing all subsequent actions. Each prefix is
identical before intervention. Direct-volume close calculations, native cash,
DD and targeted-position cash match the fixed-order enumeration. No immediate
request in this observed set was preempted; a dedicated gap test verifies
that native protections still preempt requests when required.

## Fixed-position findings

F1-F4 retain the frozen representative order of157. Helpful/harmful means
strict net improvement/deterioration greater than1e-9 USD, not statistical
significance. All observations below are synthetic post-hoc diagnostics.

| Policy | Suppressed ticks | Distinct positions in that policy | Helpful close | Harmful close | Mean CLOSE-now regret / risk | Mean passive regret / risk |
|---|---:|---:|---:|---:|---:|---:|
| F1 | 117 | 44 | 75 | 42 | 0.15668 | 0.42071 |
| F2 | 89 | 36 | 61 | 28 | 0.12064 | 0.38971 |
| F3 | 26 | 17 | 22 | 4 | 0.03464 | 0.47875 |
| F4 | 73 | 32 | 44 | 29 | 0.16104 | 0.32123 |

Regret is best executable same-position net minus the chosen alternative,
divided by risk. In total202 suppressed decisions would improve their current
position and103 would worsen it. This rules out treating every suppression as
correct, but also rules out blindly permitting every raw proposal.

Most sampled positions end at weekly flat if held:251 tick observations,
versus48 with passive SL and6 with passive TP. This composition is an observed
property of the frozen accounts, not comprehensive SL-avoidance qualification.
Both directions are represented:143 sell and162 buy tick observations.

## Complete-account findings

| Policy | Helpful account intervention | Harmful account intervention | Mean position delta USD | Mean subsequent-account delta USD | Mean account delta USD |
|---|---:|---:|---:|---:|---:|
| F1 | 74 | 43 | +13.2267 | +3.2553 | +16.4820 |
| F2 | 60 | 29 | +13.5004 | +1.7803 | +15.2807 |
| F3 | 22 | 4 | +22.3792 | +3.4932 | +25.8724 |
| F4 | 43 | 30 | +8.0950 | -0.9236 | +7.1714 |

Each row averages mutually exclusive one-intervention replays. It is NOT
the profit of applying all those closes, a deployable policy return, or an
independent-sample confidence interval. Subsequent-account delta includes
released opportunities, changed trade timing and balance-dependent sizing.
It is not a separate reward transferred from another expert.

Concrete counterexample: world52, decision49, sell position opened at49.
Closing improves that position by23.94706 USD but enables subsequent actions
whose net effect is-30.00240 USD: account delta-6.05534 USD, two extra orders,
and DD worsens by0.27649 percentage points. This SAME witness appears in F1,
F2 and F4; it is not three independent market events. Both account objectives
reject this locally profitable deviation. This is precisely why close/HOLD
and opening allocation must be evaluated together over continuation.

## Verification and reproduction

65 focused tests pass, zero skipped. Scoped Ruff and whitespace checks pass.
Tests first exposed an incorrect argument order to the existing envelope
writer; corrected before measurement. They also cover direct-volume replay,
unchanged observation, one intervention, hostile resume and native preemption.
Full resumable invocation completed again from108 authenticated world units.
Historical source and live optimizers were not modified.

```bash
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python tools/diagnose_suppressed_closes.py \
  --source <completed-median-collection> \
  --engine-root <pinned-native-engine> \
  --output <new-diagnostic-root> --workers 4
```

Per-world envelopes outside git retain complete counterfactual accounts and
traces; the public report records their hashes and every fixed-order candidate.
Outputs resume only under identical jobs/code/source and exclusive root lock.

## Consequence for the next mechanism

Do not lower all gates based on this confirmation. Build a non-anticipative
sequential allocator whose alternatives include flat NOP/BUY/SELL and occupied
HOLD/CLOSE, with subsequent opportunities and current account state carried
through native simulation. Compare complete return/DD vectors, retaining
non-dominated alternatives; do not replace this with a sum of individual
trade ratios or hypothetical blocked-agent profits.

The exact finite references152-155 remain its correctness oracle. First test
short-order sequences versus a long holding, delayed entry, and the two signs
of local-close/account disagreement on fresh predeclared scenarios. Future
samples used by a causal selector must be generated only from available
information; scenario-specific foresight cannot choose its future actions.
The above retrospective oracle values never become model inputs. Report
frontier coverage and preference sensitivity before any financial model or
floating-selection qualification. Full four-year Pareto and this sequential
causal selector are not delivered by the present diagnostic.

# Pragmatic controller contrast: completed results

2026-10-30. Implements [plan160](160_PRAGMATIC_CONTROLLER_CONTRAST_PLAN.md).
Runtime `6c9f597e`, native engine `e0a1c2b`, identity
`f5a82347b3181ee814fe07869b9e858965d52183b287293b144378eaced4e823`.

## Execution and evidence

The unattended chain completed around11:06 UTC /06:06 Bogota, about eight
minutes after launch: **1,584 native accounts**, comprising864 calibration
accounts (24 configurations x36 worlds) and720 confirmation accounts
(four frozen representatives x36 worlds xfive reporters). Runtime tests:
247 passed, no skips or failures. The persistent service exited0 with zero
restarts. Completed resume authenticated the existing artifacts without refit.

- [Frozen selection](../audits/evidence/pragmatic_contrast_full_20261030/SELECTION.json)
- [All calibration candidates](../audits/evidence/pragmatic_contrast_full_20261030/CALIBRATION.json)
- [Full trading table](../audits/evidence/pragmatic_contrast_full_20261030/RESULTS.md)
- [Numerical report, including costs and horizon mix](../audits/evidence/pragmatic_contrast_full_20261030/REPORT.json)
- [Completion and artifact hashes](../audits/evidence/pragmatic_contrast_full_20261030/COMPLETE.json)
- [Manifest](../audits/evidence/pragmatic_contrast_launch_20261030/MANIFEST.json)

Full per-world ledgers, action traces and plan vectors remain in the sealed
run archive outside git, indexed by COMPLETE.json. The published summary
files are byte-identical copies, not regenerated results. An independent
pairwise dominance calculation reproduced the four calibration selections.

**These are synthetic strategy/output tests, not financial VALIDATION results
or learned RL performance.** The predictive proxy adapts the short/long logic;
it does not reproduce the historical forecasting strategy. The prefix-only
causal reporter is an untrained extrapolator, not a qualified learned model.

## Frozen representatives

All four passed the calibration activity rule, strictly more than two orders
per market week. None required the diagnostic fallback. Confirmation did not
choose new coefficients or a winning arm.

| Arm | Key | Time exponent | Dispersion penalty | Close persistence |
|---|---|---:|---:|---:|
| Simple | c9408e3b | 0 | 0 | 2 |
| Predictive proxy | e5df77dd | 1 | 1 | 2 |
| Sequential, lower calibration DD | 280977a9 | 0 | 1 | 2 |
| Sequential, higher calibration return | 59a75d73 | 1 | 0 | 2 |

## Paired confirmation at noise1

Return is mean per-account return; DD is the worst account drawdown across36
worlds. Sharpe is mean native daily annualized account Sharpe, not a pooled
portfolio Sharpe. Direction and early-close counts aggregate all36 accounts.

| Arm | Return % | Worst DD % | Sharpe | Orders/week | Buy/sell | Win % | Buy win % | Sell win % | Early closes | Locally harmful |
|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|
| Simple | 3.3629 | 1.2847 | 19.0418 | 8.8056 | 157/160 | 72.5552 | 71.3376 | 73.7500 | 215 | 64 |
| Proxy | 3.7356 | 1.7237 | 19.0032 | 9.2222 | 167/165 | 74.3976 | 74.2515 | 74.5455 | 165 | 39 |
| Sequential280 | 4.8236 | 0.7654 | 18.2898 | 17.3889 | 314/312 | 67.4121 | 67.1975 | 67.6282 | 540 | 317 |
| Sequential59 | 5.1534 | 1.1782 | 18.2685 | 19.3333 | 344/352 | 67.2414 | 65.9884 | 68.4659 | 571 | 293 |

## Sensitivity, not a new selection

| Reporter | Simple return % | Proxy return % | Sequential280 return % | Sequential59 return % |
|---|---:|---:|---:|---:|
| Ideal | 5.0386 | 6.3196 | 6.9897 | 7.1454 |
| Noise0.5 | 4.5287 | 5.0825 | 6.7603 | 7.0659 |
| Noise1 | 3.3629 | 3.7356 | 4.8236 | 5.1534 |
| Noise2 | 1.5111 | 1.2583 | 1.5032 | 1.5000 |
| Causal prefix baseline | -2.1615 | -1.4165 | -1.4641 | -2.0801 |

With ideal reports, simple and proxy have zero locally harmful early closes
(0/77 and0/130). Sequential280 and59 have71/148 and77/157 respectively.
Their larger total returns do **not** establish that each local sacrifice is
justified by subsequent opportunities. That needs a paired full-account
counterfactual for each intervention, using the distinction established in159.

Under noise2, the sequential return advantage disappears. Worst DD for
Sequential59 rises to5.3664%, versus2.8646% for simple and2.4355% for
Sequential280. All prefix-only baseline returns are negative. This diagnoses
this reporter/controller pairing; it does not establish absence of information
in the data, failure of RL, or a reason to abandon the research.

The simpler arms therefore remain meaningful comparators. There is no overall
winner certified here and no significance claim: worlds share mirrored and
envelope transformations. The noise amplitudes are engineering perturbations,
not measured or calibrated errors of a trained model.

## Scope and post-run checks

- A plan is common across report samples before averaging: it cannot select
  a different favorable action after discovering which sample occurred.
- Sequential planning is bounded to two orders and a shortlist of two entry
  proposals. Its enumerated non-dominated set is not the global frontier.
  Conservative ties, persistence and urgent protection can override the
  score winner. Neither the executed action nor its whole trajectory is
  certified optimal even within that finite library after those safeguards.
  This qualification supersedes the generic "exact optimality" sentence in
  the immutable generated RESULTS.md; that receipt is deliberately unchanged.
- Dispersion over three report paths is not calibrated action uncertainty.
  Path reports are the engineering test interface, not an assertion that RL
  must predict a full OHLC future or receive position state as an input.
- The market-week denominator was repaired before calibration, not after
  confirmation. Native ISO-week metrics are retained for traceability.
- Post-run review found the native Sharpe convention returns0 for undefined
  constant-return cases. A reporting-only guard now uses null, with three
  added regression cases: **250 tests pass**. All1,584 archived accounts have
  orders and nonzero native Sharpe, so no completed result or selection is
  affected. Runtime6c9f597e and all archived bytes remain unchanged.

## Next work in the master plan

1. Audit this completed comparison, emphasizing local versus account-level
   close value, shortlist exclusions, future isolation and effective actions.
2. Measure per-close account continuation on the frozen sequential policies;
   report harmful, helpful and ambiguous interventions without retuning on
   these already-read confirmation worlds.
3. Qualify a causal learned-output adapter and uncertainty calibration against
   the same action/position contract. Compare simple/proxy/sequential under
   matched learned outputs before admitting model-specific floating selection.

No financial model was trained, no model inputs were modified, and the
floating-selection phase was not started by this synthetic completion.

[Audit request](../audits/SATOSHI_AUDIT_REQUEST_PRAGMATIC_CONTROLLER_CONTRAST_2026_10_30.md)

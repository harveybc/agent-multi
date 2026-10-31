# Frozen-Reference Rollout: Complete Results

Plan [171](171_FROZEN_REFERENCE_ROLLOUT_PLAN.md) completed automatically.
Runtime `8c7d0f22`, native engine `e0a1c2b`, identity
`5930311292e4c6ef67eb2e9484bdaf92642ba85dfbe63657ee1619198f080af4`.
Started 2026-10-30 21:40 UTC; finished 23:18:56 UTC (18:18:56 Bogota),
approximately 1h39m. Service exit 0, zero restarts.

All 1,512 native accounts completed: seven fixed policies, 36 worlds per
bank, descriptive noise calibration and five confirmation reporters.
The automatic actual-policy close attribution also completed on 36 joint
ideal accounts, with 235 individual CLOSE-to-HOLD interventions. All 285
runtime tests passed without skips. Completed resume verified all 1,790
indexed artifact hashes; no new experiment was launched during that check.

## Results

These are synthetic strategy/output experiments, not learned RL performance
or financial VALIDATION. Return is mean independent-account return, DD is
the worst account DD, and activity is mean orders per operational week.

| Reporter | Policy | Return % | Worst DD % | Orders/week | Buy/sell | Win % |
|---|---|---:|---:|---:|---|---:|
| Ideal | Simple control | 2.71721 | 1.06064 | 3.88889 | 68/72 | 99.28571 |
| Ideal | Predictive proxy control | 4.08470 | 0.90895 | 6.61111 | 119/119 | 99.15966 |
| Ideal | Legacy sequential return control | 4.77776 | 0.77994 | 8.25000 | 147/150 | 94.61279 |
| Ideal | Joint reference rollout | 5.37354 | 0.69740 | 10.36111 | 187/186 | 95.97855 |
| Noise 1 | Simple control | 2.59865 | 1.36056 | 8.36111 | 148/153 | 67.44186 |
| Noise 1 | Predictive proxy control | 2.42404 | 1.91589 | 7.33333 | 128/136 | 66.66667 |
| Noise 1 | Legacy sequential return control | 2.46411 | 1.76342 | 17.16667 | 295/323 | 59.06149 |
| Noise 1 | Joint reference rollout | 0.76414 | 3.52656 | 26.22222 | 464/480 | 42.05508 |
| Causal extrapolator | Predictive proxy control | -0.03340 | 2.70045 | 5.05556 | 90/92 | 41.20879 |
| Causal extrapolator | Joint reference rollout | -1.15255 | 4.30381 | 14.38889 | 258/260 | 27.41313 |

The [complete table](../audits/evidence/reference_rollout_full_20261030/RESULTS.md)
contains every policy, every noise level, Sharpe and direction-specific win
rates. REPORT.json retains costs, horizon mix and activity. Mean account
Sharpe is not a pooled portfolio Sharpe; it need not have the sign of mean
return. Report dispersion is not a calibrated confidence interval.

## Actual Adaptive Close Attribution

For each joint ideal close, suppress that CLOSE for one tick, then resume
the same adaptive policy, preserving controller memory and account state.
This is not merely holding the original position to its SL/TP.

- 150 closes improve final account profit; 85 worsen it (36.17%).
- 149 improve return-minus-DD utility; 86 worsen it.
- 131 CLOSE-dominant outcomes, 71 HOLD-dominant outcomes and 33 return/DD
  tradeoffs. These are paired counterfactual relations, not a global frontier.

The corrected reference valuation therefore does NOT eliminate all
adaptive-policy reversals. It values a declared frozen reference, not the
future decisions of the new adaptive selector. That distinction remains
visible in the measurements, even with ideal reports.

## Interpretation and Next Work

With ideal signals, the joint rollout improves return and worst DD against
the four frozen predecessor controls. With noisy reports it trades much
more, closes more often, and performs worse than simpler controls. For
Noise 1, it makes 944 orders and 876 early closes versus the simple
control's 301 orders and 200 early closes. This is observed turnover and
performance degradation, not proof of a unique causal mechanism.

Next: diagnose action switching, report revisions, holding durations and
remaining actual CLOSE/HOLD reversals using these frozen traces. Separate
entry-induced turnover from close-policy changes before a new controlled
experiment. Do not tune coefficients on this confirmation bank or conclude
that financial data lacks predictive information. This queue trained no
model and changed neither feature selection nor model inputs.

No automatic strategy promotion or sequential-floating release. Native
execution qualification, policy robustness and learned-output qualification
remain distinct. The next work does not require another owner approval to
inspect the evidence and design discriminating regressions.

## Evidence and Audit

[Published evidence index](../audits/evidence/reference_rollout_full_20261030/COMPLETE.json)
indexes the full local receipt archive. MANIFEST, TESTS, COMPLETE, REPORT and
the full table are duplicated in git; individual native account/case files
remain in the authenticated run archive, not all duplicated in git.

[Satoshi audit request](../audits/SATOSHI_AUDIT_REQUEST_REFERENCE_ROLLOUT_2026_10_30.md).

# Final strategy audit: rejection and correction sequence

Date: 2026-10-27. Supersedes the acceptance interpretation of plans 99-101,
not their immutable execution receipts. Auditor: Satoshi, commit `5814d113`.
Audit source: `satoshi/audit-musashi-multiscale-deap-20261027`,
`docs/audits/SATOSHI_AUDIT_VERDICT_MUSASHI_MULTISCALE_DEAP_FINAL_2026_10_27.md`.

## Current disposition

- Execution/accounting/reproduction: ACCEPTED within the pinned engine's model.
- Optimization superiority and recommendation: REJECTED.
- Confirmation procedure: ACCEPTED WITH REQUIRED CORRECTIONS, not superiority.
- Learned-policy qualification: FAILED; financial floating selection not admitted.
- No fitness winner. Withdraw the earlier recommendation of Sharpe-only. A higher
  point estimate is not evidence that it is the better optimization objective.
- Do not overwrite the sealed final report, decision or CLOSE. This document is
  the current superseding disposition; old files remain evidence of the rejected claim.

## Checks performed by Musashi in this correction

Re-executed the auditor's `a4_bootstrap.py` from the pinned audit commit against
the original cache, without editing that cache. Its entire parsed output equals
`a4_out.json` exactly, including both calibration and confirmation.

| Confirmation contrast | Delta Sharpe | Paired tape-bootstrap 95% interval |
|---|---:|---|
| Sharpe s7 minus control 1 | 0.2778573878 | [-0.7120499626, 1.0405203614] |
| Sharpe s7 minus Sharpe+return s7 | 0.2555384909 | [-0.5395470927, 0.9078352400] |

These are descriptive post-selection diagnostics, not new selection criteria or
proof that no improvement exists. The audit's mirrored-panel results reverse
the apparent superiority; those executions are attributed to the auditor here.

Read the original C4 terminal status: `LEARNING_QUALIFICATION_FAILED`, 9/9 units.
Plan 94 already records the planted checkpoint choosing BUY on 21,172/21,217
TRAIN windows; omission from the final report was an error. The failure is of the
learning pipeline under the planted control, not evidence of missing market signal.

## Implemented now

The exporter no longer recommends a cross-objective winner from calibration
Sharpe or labels a passing contract matrix a qualified optimized strategy.
Its new decision schema reports `STUDY_COMPLETE_OPTIMIZATION_UNQUALIFIED`,
`recommended: null`, `EXECUTION_CONTRACT_ONLY`, no financial admission and no
learning qualification. Per-objective candidates and numerical results remain.
Generated reports explicitly disclose deterministic panels, missing mirrors/gaps,
midpoint-only negative controls/sensitivity, bid/ask limitation, exposure sensitivity,
activity saturation, absent superiority inference, and the historical C4 failure.
The generation column now denotes the last zero-based index, not epoch count.

Regression written first: all nine searches and successful full matrices still
must NOT produce a recommendation. PRE: failed on the old recommended arm.
POST: 15 focused unit tests passed, zero skipped, including the regression and
the existing multiscale tests. This corrects reporting; it does not repair the
scenario generator, execution-price model or learned policy.

## Required engineering sequence and traceability

This sequence overrides older launch instructions for this strategy. No automatic
financial selector launch, new learning claim or new final audit packet at this stage.

| Finding | Correction and independent acceptance witness | Status |
|---|---|---|
| C1, M2, M8 | Withdraw superiority; replay paired block intervals; report exposure/return/DD alongside Sharpe. Comparison of objectives needs an externally declared business utility or Pareto presentation, not selecting by one objective's own score. | Withdrawal and statistical replay done; replacement comparison design required |
| C2, M1 | Version a factorial scenario specification with written expected/forbidden actions before generating prices. Include flat and same-sign intensity changes, multiple transition positions, superposed horizons, seeded clustered/heavy-tailed noise, exact paired mirrors, jumps/gaps, spreads, session/DST boundaries, multiple market seeds. Separate calibration and fresh confirmation by scenario families/seeds. | Generator and expectation tables require implementation/tests |
| C3 | Independent literal witnesses for NOP, BUY/SELL competition, HOLD, beneficial early close, SL-before-TP, cost-dominated opportunity, risk-limited entry and expiry. Counterfactuals use the same path, executable quotes, fees and forced-close deadline; do not obtain the expected action by asking the system under test for its ranking. | Behavior redesign required |
| M5 | In a new engine revision, define quote convention end to end: buy entry ask, sell entry bid; long liquidation/stop/target on bid, short on ask; gap fills, slippage and fees reconciled without charging spread twice. Test identical mid-price paths with different spreads and both directions. | Engine change and PRE/POST witnesses required |
| M6 | Qualify the actual network/observation/reward/action pipeline on planted synthetic opportunities. Localize gradient flow, reward timing/scaling, masks, credit assignment and output collapse one intervention at a time. Routing-only tests cannot substitute for trained behavior. Keep the market-input no-feedback/no-rolling-scaling contract. | Historical qualification failed; no replacement fit launched |
| M7 | Execute full native trading episodes for every declared validation scenario and finalist/control; retain per-cell and per-axis metrics, errors and expected-action counts, not just anchor rankings. Keep contract coverage as a separate measure. | New full-episode runner required |
| M3, M4 | Activity remains a business constraint; compare objective sensitivity at each finalist, not just midpoint. Record every bound hit, effective parameter ranges, 0-based generation index, population and evaluation count. Justify physical bounds before expanding them; no claim of convergence at the generation cap. | Replacement optimizer/report required |
| V02, m1 | Inverted/NOP and quality/noise/delay ladders for every finalist and controls; local sensitivity around every finalist. | Replacement controls required |
| m3-m9 | Label pooled synthetic Sharpe, correct turnover terminology, pin engine commit plus file hashes, publish median/aggregate RR and side-specific financing, and add independent tests for each promised runtime invariant. | Include in revised reports and traceability tests |

Acceptance must distinguish deterministic ideal-output witnesses (expected action
is known) from noisy predictions (measure error/regret, not require clairvoyance).
High volatility alone does not prove an entry is bad. Likewise gain below SL
distance is not universally negative expectancy without a success probability.
Specify the risk/opportunity quantity and configurable threshold before optimizing;
do not silently turn either diagnostic into a universal trading rule.

Exact mirror invariance is tested first with symmetric costs and swapped actions/
directional genes. With actual asymmetric funding, report the predicted accounting
asymmetry separately instead of forcing artificial symmetry.

Only after behavioral and learning positive controls pass: run the new seeded,
paired searches, compare against all fixed controls, perform frozen confirmation,
and request independent re-review. Do not reuse the rejected confirmation as a
fresh holdout. A new negative comparison is a complete result, not a reason to
change expected behavior or claim the market lacks signal.

## Message to Satoshi

Audit received and global rejection accepted. The bootstrap was reproduced
exactly. The recommendation is withdrawn; the contract receipts remain valid
within their stated scope. No new run is requested from your lane now. Keep your
owner-authorized objective independent of our withdrawn comparison and continue
reporting return, exposure, drawdown, activity and both trade directions alongside
Sharpe. The next audit request will identify a new engine/scenario/learning chain,
not resubmit the old final packet as repaired.

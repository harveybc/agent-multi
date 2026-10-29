# Crossed account v2: completed results

Independent review received: see [disposition143](143_CROSSED_V2_AUDIT_DISPOSITION.md)
for Holm-corrected contrasts, fixed-position early-close regret and causal-report
qualifications. The original run and numerical results below are unchanged.

Date: 2026-10-29. Execution code `e889a2d4`, native engine `e0a1c2b`.
Run identity: `4415bd3fdab307bf0869051a37f8a69035d1a727a6b642cb9afd880aa0725455`.
Design: [140](140_CROSSED_PANEL_AUTOMATION_PLAN.md); execution: [141](141_CROSSED_PANEL_EXECUTION.md).

## Outcome and scope

The complete frozen queue finished at 18:29:52 UTC (13:29:52 Bogota), in
8h31m48s: 35 calibration control jobs, three sequential DEAP searches of
20 individuals x 300 generations, 17,028 distinct evaluated candidates,
199 nondominated policies / 180 distinct objective vectors, and 75 final
confirmation jobs (27,000 native accounts). The service exited successfully.
There is no new model training, financial validation, feature selection or
deployment in these results.

P1, the high-return endpoint frozen on calibration, improves mean return and
worst drawdown against most controls on the correlated-error confirmation bank.
It does not dominate every control or every quality rung. Confidently incorrect
reports which omit reversion remain a concrete failure mode. This is a finished
synthetic experiment, not a claim that a learned reporter is qualified.

## Frozen experimental contract

Each bank has 30 independent draws (10 seeds x 3 price families), expanded to
360 paired cells by three volatility envelopes, two spreads and two mirrors.
Families: trend, two-reversal and superposed 12/36/96-bar oscillations. Random
strengths, innovations, reversal times and signed gaps are present. Calibration
seeds are 50100..50109; confirmation 60100..60109. Smoke banks are separate.
These are one-week accounts, not 360 independent markets or one continuous year.

Only correlated-error calibration enters fitness. Objectives are mean account
return and negative worst mark-to-market drawdown, subject to mean activity of
at least three orders per week. Five representatives are selected at fixed ranks
of distinct calibration objective vectors; duplicate vectors use lexicographic
genome ordering. Confirmation does not choose or retune a representative.
Seven controls and three size=1 diagnostic variants complete the 15 policies.

Leverage is 1:50; margin is `q * 0.1 * current_balance`, with 0.1 used once.
One position at a time, no daily entry cap, 12-hour opening delay and forced
flat 12 hours before weekly close. Spread, slippage, commission and financing
are charged by the native engine. Missing reports prevent entry and discretionary
close, but never disable native SL/TP or calendar enforcement.

## Correlated-error confirmation

Return is the mean of independent-account returns. DD is the worst individual
account drawdown, not the mean. Counts pool 360 accounts. Win rates are net of
the configured costs. P1..P5 are ordered by frozen calibration return rank.

| Policy | Return % | Worst DD % | Orders | Buy / sell | Win % | Buy win % | Sell win % | Early closes |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| Guarded control | 1.92120 | 1.87997 | 2185 | 1085 / 1100 | 65.26 | 65.07 | 65.45 | 1241 |
| Greedy control | 2.38541 | 2.40398 | 4206 | 2070 / 2136 | 57.87 | 57.73 | 58.01 | 3485 |
| Only 3h | 2.54494 | 4.28602 | 6327 | 3134 / 3193 | 53.09 | 52.62 | 53.55 | 4736 |
| Only 6h | 2.66711 | 4.11816 | 5769 | 2845 / 2924 | 54.05 | 53.64 | 54.45 | 4529 |
| Only 12h | 2.47196 | 2.27475 | 4321 | 2136 / 2185 | 58.39 | 58.47 | 58.31 | 3671 |
| Only 24h | 2.07555 | 1.20191 | 3494 | 1730 / 1764 | 61.79 | 61.85 | 61.73 | 2991 |
| P1 | 3.32469 | 1.77887 | 3156 | 1512 / 1644 | 72.66 | 72.02 | 73.24 | 2103 |
| P2 | 2.69332 | 1.67053 | 2446 | 1137 / 1309 | 75.76 | 77.04 | 74.64 | 1955 |
| P3 | 2.00459 | 1.33597 | 2005 | 949 / 1056 | 79.45 | 80.40 | 78.60 | 1536 |
| P4 | 1.12877 | 0.75286 | 1957 | 900 / 1057 | 80.84 | 82.44 | 79.47 | 1498 |
| P5 | 0.54552 | 0.42100 | 1614 | 620 / 994 | 80.61 | 83.39 | 78.87 | 1198 |

P1 has 8.77 orders/week and 66.63% early closes, with both directions active.
An early close is not automatically an optimal or beneficial close; these counts
are not a counterfactual close-regret certificate. Native short-week Sharpe is
retained in every replay, but is not used to rank these policies or presented
as an annual financial-validation estimate.

Paired return differences versus Greedy are +0.93928 percentage points
(descriptive cluster-bootstrap95% interval [0.76558,1.12975]); versus Only6,
+0.65759 points [0.38868,0.91819]. Positive cluster differences occur in30/30
and24/30 respectively. These intervals concern return, not the worst-DD statistic.

### Horizon concentration and same-parameter diagnostic

P1's correlated orders are3090 at6h,65 at3h,1 at12h and0 at24h:97.91% use6h.
After discovering this in the finished run, a separate360-account diagnostic
kept every P1 coefficient and allowed only6h entries. It returned3.31947%,
worstDD1.77887%,3140orders and2096early closes. FullP1 minus this diagnostic
is only+0.00523 percentage points, interval[-0.00943,0.02070],15/30positive
clusters. This supports coefficient tuning as the main source of improvement;
it does not demonstrate a multihorizon benefit. This post hoc analysis neither
changes the frozen representatives nor provides a new selection criterion.

P1's native exit reasons are2103agent closes,858TP,132SL and63forced flat.
There are201one-bar trades, mean holding time7.47hourly bars, and no zero-order
or under-three-order accounts in this rung. Mean costs per account are
USD64.316spread,13.159slippage,25.708commission and2.547financing.
The721negative-net agent closes need counterfactual analysis to distinguish
loss prevention from destructive exit; sign alone does not answer that question.

## Quality sensitivity of P1

| Report quality | Return % | Worst DD % | Orders | Early closes |
| --- | ---: | ---: | ---: | ---: |
| Ideal future | 3.94665 | 1.25487 | 1552 | 353 |
| Correlated errors | 3.32469 | 1.77887 | 3156 | 2103 |
| Delayed forecast shape | 2.31790 | 2.70654 | 1685 | 221 |
| Omitted reversion | -1.81228 | 15.12537 | 3313 | 1277 |
| Missing 20% of reports | 3.12751 | 1.74212 | 2784 | 1756 |

Ideal, correlated, delayed and available portions of missing reports use future
information by construction. They test strategy response, not causal forecasting.
The omitted-reversion arm extrapolates a recent prefix with narrow dispersion;
all active finalists and controls lose on average there, while NOP returns zero.
This identifies a vulnerability to confidently wrong signals, not proof that
market inputs cannot support learning. A raw dispersion estimate is not proof
of calibrated uncertainty. The stress generator is declared, not fitted to a
real model's errors.

There is no universal winner: in ideal confirmation, Only3 returns 5.12503%
with 1.00663% worst DD, beating P1 on both objectives. Under correlated errors,
Only24 retains lower worst DD than P1. Some conservative finalists fail the
mean-activity threshold on ideal/delayed rungs; eligibility is recorded per rung,
not repaired by retuning after confirmation.

## Exposure is part of the frontier

| Policy, correlated bank | Original size gene | Return % | DD % | Size=1 return % | Size=1 DD % |
| --- | ---: | ---: | ---: | ---: | ---: |
| P3 | 0.798 | 2.00459 | 1.33597 | 2.52069 | 1.67053 |
| P4 | 0.448 | 1.12877 | 0.75286 | 2.54398 | 1.67053 |
| P5 | 0.250 | 0.54552 | 0.42100 | 2.20585 | 1.67049 |

P1 and P2 already use size=1. Lower-risk points must not be described as superior
signal selection simply because they allocate less margin. Full action-trace
equality for these size contrasts is checked separately for every quality rung
in the analysis artifact; cash and units are expected to change.

## Parameters and immutable evidence

The five exact genomes, their calibration scores and keys are in
[FRONTIER.json](../audits/evidence/crossed_account_final_20261029/FRONTIER.json).
Gene order is fixed in [MANIFEST.json](../audits/evidence/crossed_account_final_20261029/MANIFEST.json):
four horizon weights, trend weight, uncertainty penalty, time weight, entry
threshold, size multiplier, close margin, close persistence. The uncertainty
penalty is shared by entry and close; it is not separately optimized.
P1: `[0.704, 0.978, 0.436, 0.291, -0.079, 0.334, 0.658, 0.968, 1, 0.006, 1]`.

[ANALYSIS.json](../audits/evidence/crossed_account_final_20261029/ANALYSIS.json)
contains the artifact inventory, paired cluster-bootstrap comparisons against
controls, exit reasons, horizon mix, holding durations, mean cost components,
activity distributions and size-trace checks. Bootstrap intervals are descriptive
over 30 independent draw clusters, not 360 independent samples; there is no
multiple-comparison adjustment and no inference to a live market distribution.
[RESULTS.md](../audits/evidence/crossed_account_final_20261029/RESULTS.md)
contains every one of the 75 final summaries, including failures.

Verification authenticates all receipts, checks 18,000 population fitness entries
against them, independently reconstructs the two-objective skyline, checks the
27,000 archived accounts and re-executes P1 on all five rungs (1,800 accounts).
Fresh replays are exact on this environment. The remaining policies' archives
are authenticated and checked, not all independently re-executed by this verifier.
The original preflight has 162 passing tests; the full focused suite plus four
analysis regressions now passes166/166, zero skips (2.70 seconds). Ruff checks
and git whitespace checks pass. No claim of bitwise portability to a different
numerical environment is made.

## Interpretation and next work

The experiment improves on the predecessor's coverage and supplies a concrete
conditional advantage over controls. It does not solve a new Nash game, prove
global Pareto optimality, or test learned horizon-specific reports. The current
3/6/12/24 labels choose risk profiles; all receive residual-week path reports.
The policy chooses actions from those paths, not trained per-horizon forecasts.

Next is independent review of this complete evidence, followed by a separately
specified contrast of uncertainty calibration and abstention under biased reports.
That contrast must preserve comparable support and report activity, return and
drawdown together. A learned report adapter and its positive controls must be
qualified before using this strategy in model-specific floating selection.
No new numerical thresholds or trading rules were slipped into this run during
analysis. Its completed artifacts and predecessors remain unchanged.

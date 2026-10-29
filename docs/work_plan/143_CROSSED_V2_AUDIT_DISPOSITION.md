# Crossed v2: independent audit disposition

Date: 2026-10-29. Audit: Satoshi `bc5e4e94`, on producer `5ef458ec`.
This disposition supplements [142](142_CROSSED_ACCOUNT_V2_RESULTS.md); it does
not replace or alter the completed experiment, its receipts or its source pin.
[Unmodified verdict](../audits/SATOSHI_AUDIT_VERDICT_MUSASHI_CROSSED_ACCOUNT_V2_2026_10_29.md).
Selected auditor JSON outputs are imported byte-identically from that commit
under `docs/audits/evidence/satoshi_audit_crossed_account_v2_20261029/out/`.
The complete scripts and other outputs remain available on the auditor branch.

Follow-up verification:170 focused tests pass, zero skips (3.04seconds),
including four new native boundary regressions. The eight imported objects
match the auditor's git blobs byte for byte. [Verification receipt](../audits/evidence/crossed_v2_audit_followup_20261029/VERIFICATION.json).
No production formula, completed search or old receipt was changed.

## Accepted results and corrected interpretation

The auditor independently reproduced27,360 confirmation/diagnostic accounts
with zero differences, checked2,736 additional cold accounts, all three saved
DEAP histories, the199-policy/180-vector frontier and all published summaries.
Execution and search do not require a blanket restart. The learned-model
interface is **not yet qualified**, and a new reporter/closer experiment is needed.

All seven correlated-rung return comparisons survive Holm correction. Nine of
the35 contrasts across all five rungs do not; this does not invalidate the
seven correlated comparisons. Neither the bootstrap nor Holm turns a future-aware
report into a causal one. Only the median-drift extrapolator is causal in v2,
and all active policies lose on it. Preserve the stored `omitted_reversal` key
for reproducibility, but describe it as **causal eight-bar median-drift
extrapolation with narrow uncalibrated dispersion** in new interpretation.

| Correlated P1 contrast | Return delta, pp | Holm p | Worst-DD delta, pp |
| --- | ---: | ---: | ---: |
| vs Only6 | +0.6576 | 0.0011 | -2.339 |
| vs Greedy | +0.9393 | <0.0001 | -0.625 |
| vs Only3 | +0.7798 | 0.0003 | -2.507 |
| vs Only24 | +1.2491 | <0.0001 | +0.577 |

Values are auditor estimates; complete35-comparison output:
[C_paired.json](../audits/evidence/satoshi_audit_crossed_account_v2_20261029/out/C_paired.json).
Mean correlated returns by family are overlap6.62%, reversal1.72%, trend1.63%.
The overlap family supplies roughly two thirds of mean return, and most net
fixed-position close benefit. It also drives the causal extrapolator's losses.

## Early close is the primary repair target

The audit clones the identical open position at the executed decision, compares
native passive TP/SL/forced-flat with every executable later close, and defines
O* as their maximum. Gap protection still preempts discretionary actions.
This is an **exit** optimum for an already-open order, not a new-entry optimum.

| P1 report | Early closes | Helpful | Destructive | Neutral | Mean regret / entry cash risk |
| --- | ---: | ---: | ---: | ---: | ---: |
| Ideal | 353 | 351 | 0 | 2 | 0 |
| Correlated | 2103 | 1220 | 867 | 16 | 0.586 |
| Causal median drift | 1277 | 787 | 489 | 1 | 0.802 |

Helpful/destructive means net difference versus passive hold outside a
0.02*sigma*units neutral band. The band is an auditor classification tolerance,
not a newly authorized strategy threshold. On correlated reports, all997
SL-destined closes help;594 of599 TP-destined closes harm. Fixed-position gains
are+1.783percentage points for SL-destined orders and-1.072 for TP-destined
orders; aggregate+0.678. The causal median-drift arm instead loses-0.519points
from closing. These sums are **not** full-account counterfactual returns:
changing occupancy changes subsequent entries, balance and sizing.

We will not reward a large early-close percentage, or force it toward72%.
The target is low regret and beneficial executable decisions, including HOLD
when closing destroys value. Both directions, SL-bound and TP-bound orders,
gaps and calendar liquidation must appear in the acceptance corpus.

## Disposition of every audit item

| Item | Decision and concrete scope |
| --- | --- |
| CX-01..04 | Accept independent reproduction/accounting. Preserve v2 unchanged; no new search needed for these conclusions. |
| CX-05 | Added native long/short regression with spread changing exactly at elapsed-hour12 forced flat. The wrong last-bar spread produces a different cash result. Original panel is not relabeled as covering this boundary. |
| CX-06 | Accept latent cache/config and cached-value/report binding risks. Harden separately before introducing mutable costs/report adapters; v2 immutable worlds were not affected. Independent-clock values still require a trustworthy caller; the API cannot prove their provenance. |
| CX-07 | Added exact archived P1 regressions for both mirrors: decision26 requests CLOSE, native gap SL executes at27. Full account dictionaries match the archive. |
| CX-08 | Correct wording:30 independent price draws,90 report-error streams; spread and mirror transforms share errors deliberately. Missing masks the same correlated reports. Retain pairing for one-factor contrasts. |
| CX-09..10 | Keep future-shaped reporters as diagnostic controls only. New causal calibration, separate shape and dispersion factors, new seeds and identity; no retroactive replacement of v2 fitness. |
| CX-11 | No calibrated probability claim from three paths. Define nominal quantiles, horizon and coverage unit before measuring calibration; a larger sample count alone is not a repair. |
| CX-12 | Prototype causal report-error history using sealed issued reports and only matured observations. Do not feed position/P&L state to the neural network. Gate benefit must be tested on new worlds, not fitted to these confirmation losses. |
| CX-13 | Publish Holm results and DD intervals with the descriptive comparisons. All seven correlated contrasts survive. |
| CX-14 | Accept6h concentration and inert genes. Current same-genes ablation does not demonstrate multihorizon value. Freeze a discriminating horizon-specific control before claiming it. |
| CX-15 | Publish family-stratified values and family-cluster inference, not only a pooled average. |
| CX-16 | Treat fixed-position regret, destructive-close rate and passive outcome as mandatory metrics for every finalist and rung; ideal zero-regret control is preserved. New closer experiments must reduce harm without sacrificing SL avoidance. |
| CX-17 | Separate signal/decision search at fixed size from subsequent exposure tuning. Do not reinterpret lower q as improved information processing. |
| CX-18..19 | Accept generator identity and limited optimization claims. No global Pareto or Nash certificate. |
| CX-20 | Add separately identified full-trajectory no-voluntary-close and causal-closer contrasts. Fixed-position E is not a substitute for those account replays. |

## Qualifications to the proposed acceptance rules

1. The auditor's coverage is **min/max of three samples**, not a declared90%
   interval. For three independent samples and one exchangeable continuous
   outcome, rank symmetry gives50% expected min/max coverage. This does not
   validate v2's sampler, but35--55% cannot by itself be called a failed90%
   interval. Nominal quantiles and marginal versus joint-path coverage must be
   explicit. Undercoverage of the causal extrapolator remains a concrete issue.
2. The reported loss-prediction AUC is0.747 for median drift and0.768 for delayed
   reports, but0.510 for correlated reports. It is measured on P1-selected
   entries, not every decision or a new held-out reporter. Use it as a hypothesis,
   not a ready threshold or a guarantee of successful abstention. The next AUC
   implementation must give ties average ranks.
3. Better marginal MAE/AUC need not monotonically improve each finite trading
   account; changes in serial dependence, calibration and action thresholds
   matter. PC2 needs paired, specified perturbations and uncertainty intervals,
   not a forced strict ordering on every realization.
4. A calibrated climatology can contain drift or risk-premium information;
   PC3 needs a genuinely no-edge synthetic null and cost-aware expectations.
   No universal zero-profit property follows from the word "climatology".
5. A market-only closer removes dependency on a path-report channel, not all
   miscalibration. Its own output and errors still need the same counterfactual
   evaluation. No automatic superiority is assumed.

These qualifications refine the future protocol before it runs. They do not
change the audit's measured failures or relax an already-executed test.
Next implementation sequence: [144](144_CAUSAL_REPORT_CLOSE_REPAIR_PLAN.md).

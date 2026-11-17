# 369 - Frozen Actor Critic Repair

## Requirements And Test Contract

Owner approved the diagnostic and conditional repair after368. Preserve all
running GPU workers, source299,71 inputs,28groups and market-only observations.
No live work, STL, new feature selection, TEST inference or automatic promotion.

This is not a rediscovery of316: that panel already found3/16 same-input state
conflicts.317 also found critic fitting could improve FIT while worsening held
ranking. The new question is whether the restored original critic supports the
deployed greedy actor versus the stochastic soft SAC objective after368's drift.

| Requirement | Test before execution |
|---|---|
| Exact source and data | Pinned model/support hashes; immutable source and actor |
| Counterfactuals use real TRAIN only | Native clone, first action NOP/BUY/SELL, same-state repeat, source-account reconciliation |
| No target leakage into selection | Select timestamp/state strata before computing outcomes; chronological FIT2020-22 / monitor2023, monitor windows entirely within2023 |
| No false Q comparison | Separate greedy reward target from stochastic soft target; four fixed common-random-number tapes, publish dispersion |
| State conflicts | Same market window after reachable flat/long/short forks; no state features added |
| Paired calibration | Retained vs fresh online critic heads, both with fresh calibration Adam; frozen actor/encoders/temperature, identical batches and budget |
| Ranking gate | Monitor regret strictly lower than source Q and no worse than source actor/constant NOP; pairwise accuracy not lower; RMSE not higher; require both seeds |
| Bounded actor trial | Only a passing soft-target calibration; free vs KL-constrained32 head updates, same batches; no critic or encoder changes |
| Failure/restart | Identity-bound cached panel units and completed arms; atomic output; lock; failure record and supervised retry |

## Frozen Design

CPU-only finite panel: up to4 anchors per year per state kind (flat permitted,
flat restricted,long,short), deterministic seed369. This stratified panel is
not a population estimate. Continue to the end of FIT2020-22 or monitor2023;
these diagnostic episode boundaries are explicit and do not alter GPU runs.
Record unavailable strata, never manufacture positions or fill missing data.
One reachable same-window fork per year where flat opening is permitted.

The four stochastic continuation seeds are101,202,303,404; the first action
is forced and subsequent actions follow the frozen categorical policy using
the same random tape for all three alternatives. Discount uses native elapsed
bars. Soft labels add future entropy only, not entropy of the forced action.
Four paths are a finite Monte Carlo estimate, not exact expected Q. Greedy
labels have no entropy. Position-specific conflict witnesses stay diagnostic
and are not used as unweighted calibration samples.

Calibration: seeds7,19 x retained/fresh heads x greedy/soft labels,4096 updates,
batch128, original LR3e-5, smooth-L1 on both twins' three actions. Same FIT
sampling schedule across arms. Monitor checkpoints0,64,256,1024,2048,4096;
only endpoint4096 enters the gate, avoiding retrospective best-point selection.
Sync target critics after calibration. Compare each arm to both target types
and report flat/open-long/open-short/restricted groups separately.

Conditional actor trial uses only a soft-target head configuration passing in
both seeds. It tests free versus constrained head-only SAC actor-objective
updates with LR6e-6 and fixed critics/alpha. Maximum mean KL(reference||candidate)
is1e-3 on all FIT panel windows, a declared diagnostic trust budget, not a
business constant or learned optimum. Line search halves step displacement up
to12 times and restores weights and Adam state if none passes. Report actual
KL, acceptance, action changes and native TRAIN accounts at1/8/32 attempts.
No2024 selection or evaluation is performed here. If no arm passes, record
REPAIR_NOT_QUALIFIED and end successfully, not an endless retry or a new actor fit.

Old inspected TRAIN panels mean this is a development diagnostic, not fresh
out-of-sample proof. Results cannot certify a market-wide repair.

## Prelaunch Verification

27 focused CPU tests passed: target-independent panel selection, purged monitor
context, qualification rejecting RMSE-only improvements, same-state native
forks, paired-action labels, fresh/retained head calibration with exact frozen
actor and encoder checks, KL acceptance, complete Adam/weight restoration on
rejection, native actor-head gradients, and cached evidence tamper rejection.
Correctness lint passed. No running training process was modified.
